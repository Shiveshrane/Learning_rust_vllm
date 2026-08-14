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

// ===========================================================================
// WRITTEN BY CLAUDE — Day 3 Block 1, block allocator and block table tests.
//
// Pure bookkeeping: no model, no device, no tensors. Every paging bug caught
// here would otherwise surface three layers downstream as garbled text, with
// the attention kernel as the prime suspect.
//
// Note on the invariant: `free_count() + allocated_count() == total` is a
// tautology, because `allocated_count()` is defined as `total - free.len()`.
// The meaningful check compares two INDEPENDENTLY maintained quantities — the
// allocator's free list against what the block tables actually hold. That is
// what `invariant_holds_under_random_churn` does, and it is the form the
// scheduler should assert every iteration.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    const BS: usize = 16;

    // ---- BlockAllocator ---------------------------------------------------

    #[test]
    fn new_pool_is_entirely_free() {
        let a = BlockAllocator::new(100);
        assert_eq!(a.free_count(), 100);
        assert_eq!(a.allocated_count(), 0);
    }

    #[test]
    fn allocate_hands_out_distinct_ids() {
        let mut a = BlockAllocator::new(64);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..64 {
            let id = a.allocate().expect("pool not exhausted yet");
            assert!(seen.insert(id), "block {id} handed out twice");
            assert!((id as usize) < 64, "block id {id} outside the pool");
        }
        assert_eq!(a.free_count(), 0);
        assert_eq!(a.allocated_count(), 64);
    }

    /// Exhaustion is a normal condition — it triggers preemption, so it must
    /// return None rather than panic.
    #[test]
    fn exhaustion_returns_none() {
        let mut a = BlockAllocator::new(3);
        for _ in 0..3 {
            assert!(a.allocate().is_some());
        }
        assert_eq!(a.allocate(), None);
        assert_eq!(a.allocate(), None, "still None on a second attempt");
    }

    #[test]
    fn freed_blocks_are_reusable() {
        let mut a = BlockAllocator::new(2);
        let x = a.allocate().unwrap();
        let y = a.allocate().unwrap();
        assert_eq!(a.allocate(), None);

        a.free_block(x);
        assert_eq!(a.free_count(), 1);
        let z = a.allocate().expect("the freed block should come back");
        assert_eq!(z, x);

        a.free_block(y);
        a.free_block(z);
        assert_eq!(a.free_count(), 2);
    }

    #[test]
    fn can_allocate_matches_free_count() {
        let mut a = BlockAllocator::new(4);
        assert!(a.can_allocate(4));
        assert!(!a.can_allocate(5));
        a.allocate().unwrap();
        assert!(a.can_allocate(3));
        assert!(!a.can_allocate(4));
        assert!(a.can_allocate(0), "asking for nothing always succeeds");
    }

    // ---- BlockTable -------------------------------------------------------

    /// Off-by-one here is the classic Day 3 bug. Note blocks_needed(0) == 0:
    /// an empty sequence must own no blocks, or the invariant drifts by one
    /// per sequence.
    #[test]
    fn blocks_needed_is_ceiling_division() {
        let t = BlockTable::new(BS);
        assert_eq!(t.blocks_needed(0), 0);
        assert_eq!(t.blocks_needed(1), 1);
        assert_eq!(t.blocks_needed(15), 1);
        assert_eq!(t.blocks_needed(16), 1);
        assert_eq!(t.blocks_needed(17), 2);
        assert_eq!(t.blocks_needed(32), 2);
        assert_eq!(t.blocks_needed(33), 3);
    }

    #[test]
    fn capacity_tracks_appended_blocks() {
        let mut t = BlockTable::new(BS);
        assert_eq!(t.capacity(), 0);
        t.append_block(7);
        assert_eq!(t.capacity(), 16);
        t.append_block(3);
        assert_eq!(t.capacity(), 32);
        assert_eq!(t.len_blocks(), 2);
    }

    /// The boundary the scheduler consults every iteration.
    #[test]
    fn needs_block_at_the_boundary() {
        let mut t = BlockTable::new(BS);
        assert!(t.needs_block(0), "an empty table needs a block for token 0");
        t.append_block(0);
        assert!(!t.needs_block(15), "token 15 still fits in the first block");
        assert!(t.needs_block(16), "token 16 crosses into a second block");
        t.append_block(1);
        assert!(!t.needs_block(31));
        assert!(t.needs_block(32));
    }

    /// Paging in one line: pos / block_size picks the block, pos % block_size
    /// picks the slot. Physical ids are deliberately non-contiguous here.
    #[test]
    fn locate_maps_position_to_block_and_slot() {
        let mut t = BlockTable::new(BS);
        t.append_block(42);
        t.append_block(7);
        assert_eq!(t.locate(0), (42, 0));
        assert_eq!(t.locate(15), (42, 15));
        assert_eq!(t.locate(16), (7, 0));
        assert_eq!(t.locate(31), (7, 15));
    }

    #[test]
    fn take_blocks_empties_the_table() {
        let mut t = BlockTable::new(BS);
        t.append_block(1);
        t.append_block(2);
        assert_eq!(t.blocks(), &[1, 2]);

        let taken = t.take_blocks();
        assert_eq!(taken, vec![1, 2]);
        assert_eq!(t.len_blocks(), 0, "table must be empty after take");
        assert_eq!(t.capacity(), 0);
        assert!(t.take_blocks().is_empty(), "a second take yields nothing");
    }

    // ---- the real invariant ----------------------------------------------

    /// Deterministic pseudo-random churn: grow and drop sequences, asserting
    /// after every operation that the allocator's free list and the blocks the
    /// tables actually hold still account for the whole pool.
    ///
    /// This catches leaks (a table dropped without freeing) and double-frees,
    /// neither of which the tautological form can see.
    #[test]
    fn invariant_holds_under_random_churn() {
        const TOTAL: usize = 64;
        let mut alloc = BlockAllocator::new(TOTAL);
        let mut seqs: Vec<BlockTable> = Vec::new();

        // xorshift, so the sequence is reproducible without a dependency.
        let mut rng: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        let held = |seqs: &Vec<BlockTable>| -> usize {
            seqs.iter().map(|t| t.len_blocks()).sum()
        };

        for step in 0..10_000 {
            match next() % 3 {
                // start a sequence
                0 => seqs.push(BlockTable::new(BS)),
                // grow one, if the pool allows
                1 if !seqs.is_empty() => {
                    let i = (next() as usize) % seqs.len();
                    if let Some(id) = alloc.allocate() {
                        seqs[i].append_block(id);
                    }
                }
                // finish one, returning its blocks
                2 if !seqs.is_empty() => {
                    let i = (next() as usize) % seqs.len();
                    let mut t = seqs.swap_remove(i);
                    for id in t.take_blocks() {
                        alloc.free_block(id);
                    }
                }
                _ => {}
            }

            assert_eq!(
                alloc.free_count() + held(&seqs),
                TOTAL,
                "invariant broken at step {step}: free={} held={}",
                alloc.free_count(),
                held(&seqs)
            );
        }

        // Drain everything; the pool must come back whole. This is the Day 3
        // gate item "after all requests finish, free_blocks == total_blocks".
        for mut t in seqs.drain(..) {
            for id in t.take_blocks() {
                alloc.free_block(id);
            }
        }
        assert_eq!(alloc.free_count(), TOTAL, "pool leaked");
        assert_eq!(alloc.allocated_count(), 0);
    }

    /// No block may be held by two sequences at once.
    #[test]
    fn no_block_is_ever_double_assigned() {
        let mut alloc = BlockAllocator::new(32);
        let mut tables: Vec<BlockTable> = (0..4).map(|_| BlockTable::new(BS)).collect();

        for round in 0..8 {
            for t in tables.iter_mut() {
                if let Some(id) = alloc.allocate() {
                    t.append_block(id);
                }
            }
            let mut seen = std::collections::HashSet::new();
            for t in &tables {
                for &id in t.blocks() {
                    assert!(seen.insert(id), "block {id} in two tables, round {round}");
                }
            }
        }
        assert_eq!(alloc.free_count(), 0, "all 32 blocks should be handed out");
    }
}