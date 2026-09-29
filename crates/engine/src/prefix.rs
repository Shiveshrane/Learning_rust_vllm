
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use crate::block::BlockAllocator;


pub struct PrefixCache{
    by_hash:HashMap<u64, u32>,
    hash_of:HashMap<u32, u64>,
    refcount:HashMap<u32, usize>,
    lru:VecDeque<u32>,
    block_size:usize,
}

pub fn block_hash(parent:u64, tokens:&[u32]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    parent.hash(&mut hasher);
    tokens.hash(&mut hasher);
    hasher.finish()
}


impl PrefixCache{
    pub fn new(block_size:usize) -> Self {
        Self{
            by_hash:HashMap::new(),
            hash_of:HashMap::new(),
            refcount:HashMap::new(),
            lru:VecDeque::new(),
            block_size,
        }
    }

    pub fn hash_chain(&self, tokens:&[u32])->Vec<u64>{
        let mut out=Vec::new();
        let mut parent=0u64;
        for i in 0..tokens.len()/self.block_size{
            let chunk=&tokens[i*self.block_size..(i+1)*self.block_size];
            parent=block_hash(parent, chunk);
            out.push(parent);
        }
        out
    }
    pub fn live_blocks(&self)->usize{
    self.refcount.values().filter(|c| **c>0).count()
    }
    pub fn parked(&self)->usize{
    self.lru.len()
    }

    pub fn alloc_block(&mut self, alloc:&mut BlockAllocator)->Option<u32>{
        if let Some(id)=alloc.allocate(){
            self.refcount.insert(id, 1);
            return Some(id);
        }
        while let Some(victim)=self.lru.pop_front(){
            if let Some(h)=self.hash_of.remove(&victim){
                self.by_hash.remove(&h);

            }
            self.refcount.remove(&victim);
            alloc.free_block(victim);
            if let Some(id)=alloc.allocate(){
                self.refcount.insert(id, 1);
                return Some(id);
            }
        }
        None
    }

    pub fn match_prefix(&mut self, tokens:&[u32])->Vec<u32>{
        let mut adopted=Vec::new();
        let mut parent=0u64;
        for i in 0..tokens.len()/self.block_size{
            let chunk=&tokens[i*self.block_size..(i+1)*self.block_size];
            let h=block_hash(parent, chunk);
            let Some(&block)=self.by_hash.get(&h) else {break};
            *self.refcount.entry(block).or_insert(0)+=1;
            if let Some(p)=self.lru.iter().position(|&b| b==block){
                self.lru.remove(p);
            }
            adopted.push(block);
            parent=h;
        }
        adopted
    }

    pub fn insert(&mut self, hash:u64, block:u32){
        if self.by_hash.contains_key(&hash){
            return;
        }
        self.by_hash.insert(hash, block);
        self.hash_of.insert(block, hash);
    }

    pub fn release(&mut self, blocks:&[u32], alloc:&mut BlockAllocator){
        for &b in blocks{
            let rc=self.refcount.entry(b).or_insert(0);
            if *rc>0{
                *rc-=1;
            }
            if *rc==0{
                if self.hash_of.contains_key(&b){
                    self.lru.push_back(b);
                }else{
                    self.refcount.remove(&b);
                    alloc.free_block(b);
                }
            }
        }
    }
    pub fn copy_on_write(&mut self, block:u32, alloc:&mut BlockAllocator)->Option<u32>{
        if self.refcount.get(&block).copied().unwrap_or(0)<=1{
            return Some(block);
        }
        let fresh=self.alloc_block(alloc)?;
        if let Some(rc)=self.refcount.get_mut(&block){
            *rc-=1;
        }
        Some(fresh)
    }
}

