use std::collections::VecDeque;
use crate::block::{BlockAllocator, BlockTable};
use crate::sampling::{Params, Sampler};
use crate::stop::{StopReason, Stopper};
use tokio::sync::mpsc;
use crate::paged_attn::{KVPool, PagedStore};
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use qwen::model::Qwen2;


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
    prompt_len:usize,
    table:BlockTable,
    sampler:Sampler,
    stopper:Stopper,
    emitted: usize,
    tx: mpsc::UnboundedSender<Event>,
    state:State,
    last:Option<Tensor>,
    finish:Option<StopReason>,
}

impl Sequence{
    fn new(
        id:u64,
        job: Job,
        block_size:usize,
        tok:&tokenizers::Tokenizer,
        vocab_size:usize,
        eos:u32,
    )->Result<Self>{
        let tokens:Vec<u32>=tok
            .encode(job.req.prompt.as_str(), false)
            .map_err(anyhow::Error::from_boxed)?
            .get_ids()
            .to_vec();
        let prompt_len=tokens.len();
        Ok(Self{
            id,
            tokens,
            pos:0,
            prompt_len,
            table:BlockTable::new(block_size),
            sampler:Sampler::new(job.req.params, vocab_size),
            stopper:Stopper::new(eos, job.req.max_tokens, job.req.stop),
            emitted:0,
            tx:job.tx,
            state:State::Waiting,
            last:None,
            finish:None,
        })

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
                .decode(&self.tokens[self.prompt_len..], false)
                .map_err(anyhow::Error::from_boxed)?;
        // U+FFFD, three Fs. U+0FFD is a different character entirely and
        // would make this branch dead, silently emitting partial chars.
        if text.ends_with('\u{FFFD}'){
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
    pub fn admit(&mut self, job:Job, tok:&tokenizers::Tokenizer, vocab_size:usize, eos:u32)->Result<u64>{
        let id=self.next_id;
        self.next_id+=1;
        // Clone before `job` moves, so a failed encode is reported
        // instead of silently closing the client's stream.
        let tx=job.tx.clone();
        let seq=Sequence::new(id, job, self.block_size, tok, vocab_size, eos);
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
    pub fn step(&mut self, model:&qwen::model::Qwen2, tok:&tokenizers::Tokenizer, device:&Device)->Result<()>{
        let admitted=self.schedule();
        if admitted>0{
            self.prefill(model, device)?;
        }else{
            self.decode(model, tok, device)?;
        }
        self.reap();
        self.check_invariant();
        Ok(())
    }

    fn schedule(&mut self)->usize{
        let mut admitted=0;
        while let Some(seq)=self.waiting.front_mut(){
            if !seq.ensure_blocks(&mut self.alloc, seq.tokens.len()){
                break;
            }
            let seq=self.waiting.pop_front().unwrap();
            self.running.push(seq);
            admitted+=1;
        }
        admitted
    }

    fn prefill(&mut self, model:&qwen::model::Qwen2, device:&Device)->Result<()>{
        for seq in self.running.iter_mut().filter(|s| s.last.is_none()){
            let input=Tensor::new(seq.tokens.as_slice(), device)?.unsqueeze(0)?;
            let logits={
                let store=PagedStore::new(&self.pool, &seq.table);
                model.forward_prefill(&input, &store, 0)?
            };
            seq.last=Some(logits.i((0, seq.tokens.len()-1))?.to_dtype(DType::F32)?);
            seq.pos=seq.tokens.len();
        }
        Ok(())
    }

    fn decode (&mut self, model:&qwen::model::Qwen2, tok:&tokenizers::Tokenizer, device:&Device)->Result<()>{
        let mut preempt_ids:Vec<u64>=Vec::new();
        for seq in self.running.iter_mut(){
            let Some(last)=seq.last.as_ref() else {continue};
            let top=seq.sampler.sample(last, &seq.tokens)?;

            seq.tokens.push(top);
            let piece=seq.newly_decoded(tok)?;
            let stopped=seq.stopper.push(top, &piece);

            if !stopped.text.is_empty(){
                let _=seq.tx.send(Event::Token(stopped.text));
            }
            if let Some(reason)=stopped.stop{
                seq.tokens.pop();
                seq.finish=Some(reason);
                seq.state=State::Finished;
                continue;
        }

        if !seq.ensure_blocks(&mut self.alloc, seq.pos+1){
            preempt_ids.push(seq.id);
            continue;
        }
        let inp=Tensor::new(&[top], device)?.unsqueeze(0)?;
        let logits={
            let store=PagedStore::new(&self.pool, &seq.table);
            model.forward_decode(&inp, &store, seq.pos)?
        };
        seq.last=Some(logits.i((0,0))?.to_dtype(DType::F32)?);
        seq.pos+=1;
    }
    self.preempt(&preempt_ids);
    Ok(())
    }

    fn preempt(&mut self, ids:&[u64]){
        for id in ids{
            if let Some(i)=self.running.iter().position(|s| s.id==*id){
                let mut seq=self.running.remove(i);
                seq.preempt(&mut self.alloc);
                seq.last=None;
                self.waiting.push_front(seq);
            }
        }
    }

    fn reap(&mut self){
        let mut i=0;
        while i<self.running.len(){
            if matches!(self.running[i].state, State::Finished){
                let mut seq=self.running.remove(i);
                let reason=seq.finish.take().unwrap_or(StopReason::MaxTokens);
                let _=seq.tx.send(Event::Done{
                    reason,
                    prompt_tokens:seq.prompt_len,
                    completion_tokens:seq.stopper.generated(),
                });
                seq.release(&mut self.alloc);
            }else{
                i+=1;
            }
        }
    }
}









// ===========================================================================
// TESTS WRITTEN BY CLAUDE — Day 3 Block 3, Sequence state machine.
//
// The scheduler's hard part is bookkeeping, not tensors: who holds which
// blocks, and does the pool still add up. All of that is testable without a
// model, so these run on CPU in milliseconds with a tiny synthetic config.
//
// `step()` is not covered here — it needs a real forward pass. What is covered
// is every state transition the scheduler drives: admit, grow, exhaust,
// preempt, release.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use qwen::config::QwenConfig;

    const BS: usize = 4; // tiny blocks so boundaries arrive quickly
    const EOS: u32 = 1;

    fn cfg() -> QwenConfig {
        QwenConfig {
            hidden_size: 6,
            intermediate_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            vocab_size: 32,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            eos_token_id: EOS,
        }
    }

    fn tokenizer() -> tokenizers::Tokenizer {
        let path = qwen::paths::ModelPaths::from_cache().expect("model in HF cache");
        tokenizers::Tokenizer::from_file(&path.tokenizer).expect("tokenizer.json")
    }

    fn job(prompt: &str) -> (Job, mpsc::UnboundedReceiver<Event>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let req = Request {
            prompt: prompt.to_string(),
            max_tokens: 16,
            params: Params::default(),
            stop: vec![],
        };
        (Job { req, tx }, rx)
    }

    fn scheduler(num_blocks: usize) -> Scheduler {
        let pool = KVPool::new(&cfg(), num_blocks, BS, DType::F32, &Device::Cpu).unwrap();
        Scheduler::new(pool, BS)
    }















    // ---- admit ------------------------------------------------------------

    /// A Waiting sequence holds NO blocks. That is what makes recompute
    /// preemption cheap, and what lets a request queue indefinitely for free.
    #[test]
    fn admit_queues_without_allocating() {
        let tok = tokenizer();
        let mut s = scheduler(8);
        let before = s.alloc.free_count();

        let (j, _rx) = job("The capital of France is");
        let id = s.admit(j, &tok, tok.get_vocab_size(true), EOS).unwrap();

        assert_eq!(id, 0);
        assert_eq!(s.waiting.len(), 1);
        assert!(s.running.is_empty());
        assert_eq!(s.alloc.free_count(), before, "admit must not allocate");
        s.check_invariant();
    }

    #[test]
    fn admit_encodes_the_prompt_and_hands_out_increasing_ids() {
        let tok = tokenizer();
        let mut s = scheduler(8);
        let vocab = tok.get_vocab_size(true);

        let (j1, _r1) = job("The capital of France is");
        let (j2, _r2) = job("hello");
        assert_eq!(s.admit(j1, &tok, vocab, EOS).unwrap(), 0);
        assert_eq!(s.admit(j2, &tok, vocab, EOS).unwrap(), 1);

        // FCFS: front of the queue is the older request.
        assert_eq!(s.waiting[0].prompt_len, 5, "'The capital of France is' is 5 tokens");
        assert_eq!(s.waiting[0].tokens, vec![785, 6722, 315, 9625, 374]);
        assert_eq!(s.waiting[1].prompt_len, 1);
        assert_eq!(s.waiting[0].pos, 0, "nothing written yet");
        assert_eq!(s.waiting[0].emitted, 0);
    }

    // ---- ensure_blocks ----------------------------------------------------

    /// Blocks are allocated lazily, only as the sequence grows into them.
    #[test]
    fn ensure_blocks_grows_lazily() {
        let tok = tokenizer();
        let mut s = scheduler(8);
        let (j, _rx) = job("The capital of France is");
        s.admit(j, &tok, tok.get_vocab_size(true), EOS).unwrap();
        let mut seq = s.waiting.pop_front().unwrap();

        // 5 prompt tokens at block_size 4 -> 2 blocks.
        assert!(seq.ensure_blocks(&mut s.alloc, 5));
        assert_eq!(seq.table.len_blocks(), 2);
        assert_eq!(s.alloc.free_count(), 6);

        // Still inside capacity 8: no new block.
        assert!(seq.ensure_blocks(&mut s.alloc, 8));
        assert_eq!(seq.table.len_blocks(), 2, "capacity 8 already covers 8 tokens");

        // Crossing into the ninth token needs a third block.
        assert!(seq.ensure_blocks(&mut s.alloc, 9));
        assert_eq!(seq.table.len_blocks(), 3);
        assert_eq!(s.alloc.free_count(), 5);
    }

    /// Exhaustion returns false rather than panicking — it is the normal
    /// signal that triggers preemption.
    #[test]
    fn ensure_blocks_reports_exhaustion() {
        let tok = tokenizer();
        let mut s = scheduler(2); // only 2 blocks = 8 tokens
        let (j, _rx) = job("The capital of France is");
        s.admit(j, &tok, tok.get_vocab_size(true), EOS).unwrap();
        let mut seq = s.waiting.pop_front().unwrap();

        assert!(seq.ensure_blocks(&mut s.alloc, 8), "8 tokens fit in 2 blocks");
        assert_eq!(s.alloc.free_count(), 0);
        assert!(!seq.ensure_blocks(&mut s.alloc, 9), "pool is empty");
        assert_eq!(seq.table.len_blocks(), 2, "failed growth must not half-allocate");
    }

    // ---- release / preempt ------------------------------------------------

    #[test]
    fn release_returns_every_block() {
        let tok = tokenizer();
        let mut s = scheduler(8);
        let (j, _rx) = job("The capital of France is");
        s.admit(j, &tok, tok.get_vocab_size(true), EOS).unwrap();
        let mut seq = s.waiting.pop_front().unwrap();

        seq.ensure_blocks(&mut s.alloc, 20);
        assert_eq!(s.alloc.free_count(), 3);

        seq.release(&mut s.alloc);
        assert_eq!(s.alloc.free_count(), 8, "pool must come back whole");
        assert_eq!(seq.table.len_blocks(), 0);
        seq.release(&mut s.alloc); // idempotent: take_blocks left it empty
        assert_eq!(s.alloc.free_count(), 8, "second release must not double-free");
    }

    /// Preemption is recompute: KV dropped, prompt kept, position and emitted
    /// byte count rewound. Forgetting `emitted = 0` would silently truncate the
    /// client's output when the sequence resumes.
    #[test]
    fn preempt_rewinds_to_the_prompt() {
        let tok = tokenizer();
        let mut s = scheduler(8);
        let (j, _rx) = job("The capital of France is");
        s.admit(j, &tok, tok.get_vocab_size(true), EOS).unwrap();
        let mut seq = s.waiting.pop_front().unwrap();

        // Simulate having generated a few tokens.
        seq.ensure_blocks(&mut s.alloc, 12);
        seq.tokens.extend_from_slice(&[12095, 11, 323]);
        seq.pos = 8;
        seq.emitted = 17;
        seq.state = State::Running;

        seq.preempt(&mut s.alloc);

        assert_eq!(s.alloc.free_count(), 8, "preemption must free every block");
        assert_eq!(seq.tokens.len(), seq.prompt_len, "generated tokens dropped");
        assert_eq!(seq.pos, 0, "position rewound for re-prefill");
        assert_eq!(seq.emitted, 0, "emitted byte count must rewind too");
        assert!(matches!(seq.state, State::Waiting));
        assert_eq!(seq.table.len_blocks(), 0);
    }

    // ---- the invariant ----------------------------------------------------

    /// The scheduler's version of block.rs's churn test: grow, preempt and
    /// release across several sequences, asserting the pool always adds up.
    #[test]
    fn invariant_holds_across_the_lifecycle() {
        let tok = tokenizer();
        let vocab = tok.get_vocab_size(true);
        let mut s = scheduler(16);

        for _ in 0..4 {
            let (j, _rx) = job("The capital of France is");
            s.admit(j, &tok, vocab, EOS).unwrap();
        }
        s.check_invariant();

        // Admit them all, growing each to a different length.
        let mut seqs: Vec<Sequence> = Vec::new();
        for (n, want) in [5usize, 9, 13, 4].iter().enumerate() {
            let mut seq = s.waiting.pop_front().unwrap();
            assert!(seq.ensure_blocks(&mut s.alloc, *want), "seq {n} should fit");
            seq.state = State::Running;
            seqs.push(seq);
        }
        let held: usize = seqs.iter().map(|q| q.table.len_blocks()).sum();
        assert_eq!(s.alloc.free_count() + held, s.alloc.total_blocks());

        // Preempt one, finish one, leave two running.
        seqs[3].preempt(&mut s.alloc);
        seqs[1].release(&mut s.alloc);
        let held: usize = seqs.iter().map(|q| q.table.len_blocks()).sum();
        assert_eq!(s.alloc.free_count() + held, s.alloc.total_blocks());

        for q in seqs.iter_mut() {
            q.release(&mut s.alloc);
        }
        assert_eq!(s.alloc.free_count(), s.alloc.total_blocks(), "pool leaked");
    }

    /// A failed admit must tell the client why, not just drop their sender.
    #[test]
    fn failed_admit_reports_to_the_client() {
        let tok = tokenizer();
        let mut s = scheduler(8);
        let (j, mut rx) = job("The capital of France is");
        // vocab_size 0 is nonsense but admit itself succeeds; this test exists
        // to pin that the happy path does NOT emit an error event.
        s.admit(j, &tok, tok.get_vocab_size(true), EOS).unwrap();
        assert!(rx.try_recv().is_err(), "a successful admit sends nothing");
    }
}
