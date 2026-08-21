use std::collections::VecDeque;
use crate::block::{BlockAllocator, BlockTable};
use crate::paged_attn::KVPool;
use crate::sampling::{Params, Sampler};
use crate::stop::{StopReason, Stopper};
use tokio::sync::mpsc;



pub struct Request {
    pub prompt: String,
    pub max_tokens: usize,
    pub params: crate::sampling::Params,
    pub stop: Vec<String>,
}

pub enum Event {
    Token(String),
    Done { reason: StopReason, prompt_tokens: usize, completion_tokens: usize },
    Error(String),
}

pub struct Job {
    pub req: Request,
    pub tx: tokio::sync::mpsc::UnboundedSender<Event>,
}


enum State{
    Waiting, 
    Running,
    Preempted,
    Finished
}

struct Sequence{
    id:u64, 
    tokens:Vec<u32>,
    pos:usize,
    table:BlockTable,
    sampler:Sampler,
    stopper:Stopper,
    emitted: usize,
    tx: mpsc::UnboundedSender<Event>,
    state:State,
}

impl Sequence{
    fn new(job: Job, block_size:usize)->Self{
        let id=job.req.id;
        let tokens=job.req.input_ids;
        let pos=0;
        let table=BlockTable::new(block_size);
        let sampler=Sampler::new(&job.req.sampling_params);
        let stopper=Stopper::new(&job.req.stop_params);
        let detok=Detokenizer::new();
        let tx=job.tx;
        let state=State::Waiting;
        Self{
            id, tokens, pos, table, sampler, stopper, detok, tx, state
        }

    }

    fn ensure_blocks(&mut self, alloc:&mut BlockAllocator, needed:usize)->bool{
        while self.table.capacity()<needed{
            match alloc.allocate(){
                Some(id)=>self.table.append_block(id),
                None=>return false,
            }
        }
        true
    }

    fn store<'a>(&'a self, pool:&'a KVPool)->PagedStore<'a>{
        PagedStore::new(pool, &self.table)
    }

    fn newly_decoded(&mut self, tok:&tokenizers::Tokenizer)->Result<String>{
        let text=tok
                .decode(self.tokens[self.pos..].to_vec(), false)
                .map_err(anyhow::Error::from_boxed)?;
        if text.ends_with("\u{FFD}"){
            return Ok(String::new());
        }
        let out=text[self.emitted..].to_string();
        self.emitted=text.len();
        Ok(out)
    }
    fn release(&mut self, alloc:&mut BlockAllocator){
        for id in self.table.take_blocks(){
            alloc.free_block(id);
        }
    }

    fn preempt(&mut self, alloc:&mut BlockAllocator){
        self.release(alloc);
        self.state=State::Waiting;
        self.tokens.truncate(self.prompt_len);
        self.pos=0;
        self.emitted=0;
    }

}


pub struct Scheduler{
    waiting:VecDeque<Sequence>,
    running:Vec<Sequence>,
    pool:KVPool,
    alloc:BlockAllocator,
    block_size:usize,
    next_id:u64,
}

impl Scheduler{
    pub fn new(pool:KVPool, block_size:usize)->Self{
        let alloc=BlockAllocator::new(pool.num_blocks());
        Self{
            waiting:VecDeque::new(),
            running:Vec::new(),
            pool,
            alloc,
            block_size,
            next_id:0,
        }
    }
    pub fn admit(&mut self, job:Job, tok:&Tokenizer, vocab_size:usize, eos:usize)->Result<()>{
        let mut seq=Sequence::new(job, self.block_size);
        let id=self.next_id;
        self.next_id+=1;
        let tx=job.tx.clone();

        match seq{
            Ok(seq)=>{
                self.waiting.push_back(seq);
                Ok(id)
            }
            Err(e)=>{
                let _=tx.send(Event::Error(format!("Failed to create sequence: {e}")));
                Err(e)
            }
        }

    }

    pub fn check_invariant(&self){
        let held:usize=self.running.iter().map(|s| s.table.len_blocks())
        .sum::<usize>()
        +self.waiting.iter().map(|s| s.table.len_blocks())
        .sum::<usize>();

        debug_assert_eq!(
            self.alloc.free_count()+held,
            self.alloc.total_blocks(),
            "Block accounting broken"
        );
    }
}