pub struct BlockAllocator{
    total:usize,
    free:Vec<u32>,
}

pub struct BlockTable{
    blocks:Vec<u32>,
    block_size:usize,
}


impl BlockAllocator{
    pub fn new(total:usize)->Self{
        let mut free=Vec::new();
        for i in 0..total as u32{
            free.push(i);
        }
        Self{total, free}
    }

    pub fn allocate(&mut self)->Option<u32>{
        self.free.pop()
    }
    pub fn free_block(&mut self, id:u32){
        self.free.push(id);
    }

    pub fn free_count(&self)->usize{
        self.free.len()
    }
    pub fn allocated_count(&self)->usize{
        self.total-self.free.len()
    }

    pub fn can_allocate(&self, n:usize)->bool{
        self.free.len()>=n
    }
}

impl BlockTable{
    pub fn new(block_size:usize)->Self{
        Self{blocks:Vec::new(), block_size}
    }

    pub fn locate(&self, pos:usize)->(u32, usize){
        debug_assert!(pos<self.capacity(), "position {} out of bounds for capacity {}", pos, self.capacity());
        (self.blocks[pos/self.block_size], pos%self.block_size)
    }

    pub fn blocks_needed(&self, len:usize)->usize{
        (len+self.block_size-1)/self.block_size
    }

    pub fn capacity(&self)->usize{
        self.blocks.len()*self.block_size
    }
    pub fn needs_block(&self, current_len:usize)->bool{
        current_len>=self.capacity()
    }

    pub fn len_blocks(&self)->usize{
        self.blocks.len()
    }
    pub fn blocks(&self)->&[u32]{
        &self.blocks
    }
    pub fn take_blocks(&mut self)->Vec<u32>{
        std::mem::take(&mut self.blocks)
    }
    pub fn append_block(&mut self, id:u32){
        self.blocks.push(id);
    }
}