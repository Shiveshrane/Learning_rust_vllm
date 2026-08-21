use qwen::config::QwenConfig;
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
    pub fn total_blocks(&self)->usize{
        self.total
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


pub fn blocks_for_budget(budget_bytes:usize, cfg:&QwenConfig, block_size:usize, dtype_bytes:usize)->usize{

    budget_bytes/(cfg.kv_bytes_per_token(dtype_bytes)*block_size)
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

    // ---- pool sizing ------------------------------------------------------

    // WRITTEN BY CLAUDE — Day 3 Block 1, blocks_for_budget tests.
    //
    // Needs a real QwenConfig, so it reads config.json out of the HF cache.
    // No weights are loaded — this stays a milliseconds-scale test.

    fn cfg() -> QwenConfig {
        let path = qwen::paths::ModelPaths::from_cache().expect("model in HF cache");
        QwenConfig::from_path(&path.config).expect("config.json")
    }

    /// The sizing arithmetic the pool is built from. F32 KV costs 57,344
    /// bytes/token on this checkpoint — double the 28,672 the README quotes,
    /// because that figure assumes bf16.
    #[test]
    fn budget_divides_into_blocks() {
        let c = cfg();
        assert_eq!(c.kv_bytes_per_token(4), 57_344, "F32 KV per token");
        assert_eq!(c.kv_bytes_per_token(2), 28_672, "bf16 KV per token");

        // One block of 16 tokens, across all 28 layers.
        let f32_block = 57_344 * 16;
        assert_eq!(f32_block, 917_504);

        let budget = 15_000_000_000usize; // ~15 GB of KV headroom
        assert_eq!(blocks_for_budget(budget, &c, 16, 4), budget / f32_block);
        assert_eq!(blocks_for_budget(budget, &c, 16, 4), 16_348);
    }

    /// Halving the element width doubles the pool. This is the whole argument
    /// for bf16 KV on Day 4, in one assertion.
    #[test]
    fn halving_dtype_width_doubles_the_pool() {
        let c = cfg();
        let budget = 15_000_000_000usize;
        let wide = blocks_for_budget(budget, &c, 16, 4);
        let narrow = blocks_for_budget(budget, &c, 16, 2);
        // Not exactly 2x: integer division truncates, and the wider dtype
        // throws away more of the remainder. Within one block.
        assert!(narrow >= wide * 2, "bf16 must hold at least twice as much");
        assert!(narrow - wide * 2 <= 1, "and no more than that, give or take truncation");
    }

    /// Bigger blocks mean fewer of them, and the product stays put: the pool
    /// holds the same number of TOKENS either way. Block size trades internal
    /// fragmentation against block-table length, not capacity.
    #[test]
    fn block_size_trades_count_for_size_not_capacity() {
        let c = cfg();
        let budget = 15_000_000_000usize;
        let tokens = |bs: usize| blocks_for_budget(budget, &c, bs, 4) * bs;

        assert_eq!(blocks_for_budget(budget, &c, 32, 4), blocks_for_budget(budget, &c, 16, 4) / 2);
        // Equal to within one block's worth of truncation.
        assert!(tokens(16).abs_diff(tokens(32)) <= 32);
        assert!(tokens(16).abs_diff(tokens(8)) <= 32);
    }

    /// The comparison Day 3 exists to make. Day 2 pinned max_seq=4096 per
    /// sequence; paging hands out 16-token blocks on demand instead.
    #[test]
    fn paging_beats_preallocation_by_two_orders_of_magnitude() {
        let c = cfg();
        let budget = 15_000_000_000usize;

        let day2_sequences = budget / (c.kv_bytes_per_token(4) * 4096);
        let day3_tokens = blocks_for_budget(budget, &c, 16, 4) * 16;

        assert_eq!(day2_sequences, 63, "Day 2: whole sequences, most of it wasted");
        assert!(
            day3_tokens > day2_sequences * 4096,
            "paging must not hold fewer tokens than preallocation"
        );
        // A 50-token request costs 4 blocks instead of 4096 tokens of reservation.
        assert_eq!(BlockTable::new(BS).blocks_needed(50), 4);
    }

    #[test]
    fn tiny_budget_yields_no_blocks() {
        let c = cfg();
        assert_eq!(blocks_for_budget(0, &c, 16, 4), 0);
        assert_eq!(blocks_for_budget(917_503, &c, 16, 4), 0, "one byte short of a block");
        assert_eq!(blocks_for_budget(917_504, &c, 16, 4), 1);
    }
}
