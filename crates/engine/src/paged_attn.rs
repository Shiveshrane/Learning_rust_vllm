use qwen::config::QwenConfig;
use crate::block::BlockTable;
use candle_core::{DType, Device, Tensor};
use anyhow::Result;
use qwen::cache::KVStore;
pub struct KVPool{
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    block_size: usize,
    num_blocks: usize,
    device: Device,
}


impl KVPool{
    pub fn new(cfg:&QwenConfig, num_blocks:usize, block_size:usize, dtype: DType, device: &Device)->Result<Self>{
        let mut keys=Vec::with_capacity(cfg.num_hidden_layers);
        let mut values=Vec::with_capacity(cfg.num_hidden_layers);
        let slots=num_blocks*block_size;
        let kv_heads=cfg.num_key_value_heads;
        let head_dim=cfg.head_dim();
        for _ in 0..cfg.num_hidden_layers{
            let k=Tensor::zeros(&[slots, kv_heads, head_dim], dtype, device)?;
            let v=Tensor::zeros(&[slots, kv_heads, head_dim], dtype, device)?;
            keys.push(k);
            values.push(v);
        }
        Ok(Self { keys, values, block_size, num_blocks, device: device.clone() })
    }
    fn slot_of(&self, table:&BlockTable, pos:usize)->usize{
        let (block, slot)=table.locate(pos);
        block as usize*self.block_size+slot as usize
    }

    pub fn write(&self, layer:usize, table:&BlockTable, start_pos:usize, k:&Tensor, v:&Tensor)->Result<()>{
        let s=k.dim(2)?;
        let k3=k.squeeze(0)?.transpose(0,1)?.contiguous()?;
        let v3=v.squeeze(0)?.transpose(0,1)?.contiguous()?;
        for i in 0..s{
            let dst=self.slot_of(table, start_pos+i);
            self.keys[layer].slice_set(&k3.narrow(0,i,1)?.contiguous()?,0, dst)?;
            self.values[layer].slice_set(&v3.narrow(0,i,1)?.contiguous()?,0, dst)?;
        }
        Ok(())
    }
    pub fn gather(&self, layer:usize, table:&BlockTable, len:usize)->Result<(Tensor, Tensor)>{
        let idx:Vec<u32>=(0..len).map(|i| self.slot_of(table, i) as u32).collect();
        let idx=Tensor::new(idx.as_slice(), &self.device)?;

        let pick=|pool:&Tensor|->Result<Tensor>{
            Ok(pool
            .index_select(&idx, 0)?
            .transpose(0,1)?
            .contiguous()?
            .unsqueeze(0)?)
        };
        Ok((pick(&self.keys[layer])?, pick(&self.values[layer])?))
    }
    pub fn num_blocks(&self)->usize{
        self.num_blocks
    }
    pub fn block_size(&self)->usize{
        self.block_size
    }
}

pub struct PagedStore<'a>{
    pool:&'a KVPool,
    table:&'a BlockTable,
}
impl<'a> PagedStore<'a>{
    pub fn new(pool:&'a KVPool, table:&'a BlockTable)->Self{
        Self{pool, table}
    }
}

impl KVStore for PagedStore<'_>{
    fn write(&self, layer:usize, start_pos:usize, keys:&Tensor, values:&Tensor)->Result<()>{
        self.pool.write(layer, self.table, start_pos, keys, values)
    }
    fn gather(&self, layer:usize, len:usize)->Result<(Tensor, Tensor)>{
        self.pool.gather(layer, self.table, len)
    }
}






// ===========================================================================
// WRITTEN BY CLAUDE — Day 3 Block 2, KVPool round-trip tests.
//
// `write` scatters through two transposes into flat slots; `gather` pulls back
// with index_select and two more transposes. A shape or ordering bug compiles
// fine and only shows up much later as garbled text, so these pin the layout
// directly.
//
// Every test allocates blocks OUT OF ORDER. A pool that assumes
// block_id == logical_index passes a tidy test and fails the moment the
// allocator recycles a block, which is the normal case under load.
//
// CPU device and a tiny synthetic config: no Metal, no weights, milliseconds.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    const BS: usize = 4; // small block size so boundaries are easy to hit
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 3;

    /// Two layers, 2 KV heads, head_dim 3. Small enough to reason about by hand.
    fn cfg() -> QwenConfig {
        QwenConfig {
            hidden_size: 6, // / num_attention_heads(2) = head_dim 3
            intermediate_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: KV_HEADS,
            vocab_size: 32,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            eos_token_id: 1,
        }
    }

    /// A distinctive value per (position, head, dim), so a transposed or
    /// misordered result cannot accidentally compare equal.
    fn val(pos: usize, head: usize, dim: usize) -> f32 {
        (pos * 100 + head * 10 + dim) as f32
    }

    /// [1, kv_heads, s, head_dim] for logical positions start..start+s,
    /// matching what Attention::forward hands to `write`.
    fn kv_for(start: usize, s: usize) -> Tensor {
        let mut data = Vec::with_capacity(KV_HEADS * s * HEAD_DIM);
        for h in 0..KV_HEADS {
            for p in 0..s {
                for d in 0..HEAD_DIM {
                    data.push(val(start + p, h, d));
                }
            }
        }
        Tensor::from_vec(data, (1, KV_HEADS, s, HEAD_DIM), &Device::Cpu).unwrap()
    }

    fn table_with(blocks: &[u32]) -> BlockTable {
        let mut t = BlockTable::new(BS);
        for &b in blocks {
            t.append_block(b);
        }
        t
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn pool() -> KVPool {
        KVPool::new(&cfg(), 12, BS, DType::F32, &Device::Cpu).unwrap()
    }

    /// The core claim: whatever physical blocks a sequence owns, gather returns
    /// its tokens in LOGICAL order.
    #[test]
    fn round_trip_preserves_logical_order() {
        let p = pool();
        // Deliberately scattered and descending — nothing about these ids
        // matches the logical index.
        let table = table_with(&[5, 2, 9]);
        let len = 3 * BS;

        let k = kv_for(0, len);
        p.write(0, &table, 0, &k, &k).unwrap();

        let (gk, gv) = p.gather(0, &table, len).unwrap();
        assert_eq!(flat(&gk), flat(&k), "keys came back reordered");
        assert_eq!(flat(&gv), flat(&k), "values came back reordered");
    }

    /// Exactly the shape sdpa wants, and the shape Day 2's KVCache returned.
    #[test]
    fn gather_shape_matches_sdpa_expectations() {
        let p = pool();
        let table = table_with(&[7, 1]);
        let len = 6;
        p.write(0, &table, 0, &kv_for(0, len), &kv_for(0, len)).unwrap();

        let (gk, _) = p.gather(0, &table, len).unwrap();
        assert_eq!(gk.dims(), &[1, KV_HEADS, len, HEAD_DIM]);
    }

    /// A sequence rarely ends on a block boundary. The partly-filled tail of
    /// the last block must not leak into the gather.
    #[test]
    fn partial_last_block_is_not_gathered() {
        let p = pool();
        let table = table_with(&[3, 8]); // capacity 8
        let len = 6; // last block only half used

        p.write(0, &table, 0, &kv_for(0, len), &kv_for(0, len)).unwrap();
        let (gk, _) = p.gather(0, &table, len).unwrap();

        assert_eq!(gk.dims()[2], len, "gathered more positions than exist");
        assert_eq!(flat(&gk), flat(&kv_for(0, len)));
    }

    /// Prefill writes many positions at once, then each decode step writes one.
    /// `start_pos` is what stitches those together.
    #[test]
    fn incremental_writes_match_one_big_write() {
        let table = table_with(&[4, 0]);
        let len = 5;

        let batched = pool();
        batched.write(0, &table, 0, &kv_for(0, len), &kv_for(0, len)).unwrap();

        // Same data, written as a 3-token prefill then two decode steps.
        let stepped = pool();
        stepped.write(0, &table, 0, &kv_for(0, 3), &kv_for(0, 3)).unwrap();
        for pos in 3..len {
            let one = kv_for(pos, 1);
            stepped.write(0, &table, pos, &one, &one).unwrap();
        }

        let (a, _) = batched.gather(0, &table, len).unwrap();
        let (b, _) = stepped.gather(0, &table, len).unwrap();
        assert_eq!(flat(&a), flat(&b), "incremental writes diverged");
    }

    /// Two sequences sharing one pool must not see each other's KV. This is the
    /// failure that corrupts output with no crash.
    #[test]
    fn sequences_with_disjoint_blocks_do_not_alias() {
        let p = pool();
        let a = table_with(&[6, 11]);
        let b = table_with(&[0, 3]);
        let len = 5;

        let ka = kv_for(0, len);
        let kb = kv_for(50, len); // clearly different values
        p.write(0, &a, 0, &ka, &ka).unwrap();
        p.write(0, &b, 0, &kb, &kb).unwrap();

        assert_eq!(flat(&p.gather(0, &a, len).unwrap().0), flat(&ka));
        assert_eq!(flat(&p.gather(0, &b, len).unwrap().0), flat(&kb));
    }

    /// Layers share block tables but must not share storage.
    #[test]
    fn layers_are_independent() {
        let p = pool();
        let table = table_with(&[2, 5]);
        let len = 4;

        let l0 = kv_for(0, len);
        let l1 = kv_for(30, len);
        p.write(0, &table, 0, &l0, &l0).unwrap();
        p.write(1, &table, 0, &l1, &l1).unwrap();

        assert_eq!(flat(&p.gather(0, &table, len).unwrap().0), flat(&l0));
        assert_eq!(flat(&p.gather(1, &table, len).unwrap().0), flat(&l1));
    }

    /// K and V occupy separate storage; writing different data to each must
    /// survive the round trip.
    #[test]
    fn keys_and_values_do_not_share_storage() {
        let p = pool();
        let table = table_with(&[1]);
        let len = 4;

        let k = kv_for(0, len);
        let v = kv_for(70, len);
        p.write(0, &table, 0, &k, &v).unwrap();

        let (gk, gv) = p.gather(0, &table, len).unwrap();
        assert_eq!(flat(&gk), flat(&k));
        assert_eq!(flat(&gv), flat(&v));
    }

    /// Recycling a block must hand back the NEW occupant's data, not a ghost of
    /// the old one. Blocks are reused constantly once the pool is under load.
    #[test]
    fn reused_block_returns_fresh_data() {
        let p = pool();
        let table = table_with(&[10]);
        let len = 4;

        let old = kv_for(0, len);
        p.write(0, &table, 0, &old, &old).unwrap();

        // Same physical block, new sequence, new data.
        let new = kv_for(80, len);
        p.write(0, &table, 0, &new, &new).unwrap();

        assert_eq!(flat(&p.gather(0, &table, len).unwrap().0), flat(&new));
    }
}
