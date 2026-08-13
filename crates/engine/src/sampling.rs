use anyhow::Result;
use candle_core::Tensor;
use rand::prelude::*;
use std::collections::HashSet;

pub struct Params{
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_prob: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
}

impl Default for Params{
    /// Every knob off: greedy, no truncation, no penalty.
    fn default()->Self{
        Self{
            temperature: 0.0,
            top_k: None,
            top_p: None,
            min_prob: None,
            repetition_penalty: None,
            seed: None,
        }
    }
}


pub struct Sampler{
    params: Params,
    rng: StdRng,
    vocab_size: usize,
}



fn apply_repetition_penalty(logits: &mut [f32], prev: &[u32], penalty: f32){
    if penalty == 1.0{
        return;
    }
    let seen: HashSet<u32> = prev.iter().copied().collect();
    for id in seen{
        let i = id as usize;
        if i >= logits.len(){
            continue;
        }
        logits[i] = if logits[i] > 0.0{
            logits[i] / penalty
        } else {
            logits[i] * penalty
        };
    }
}

fn apply_temperature(logits: &mut [f32], temperature:f32){
    if temperature == 1.0{
        return;
    }
    for l in logits.iter_mut() {
        *l /= temperature;
    }
}

fn apply_top_k(logits: &mut [f32], top_k:usize){
    if top_k==0 || top_k >= logits.len(){
        return; 
    }
    let mut vals:Vec<f32> = logits.to_vec();
    let (_,&mut kth,_)=vals.select_nth_unstable_by(top_k-1, |a,b| b.total_cmp(a)); //QuickSelect
    for l in logits.iter_mut(){
        if *l < kth{
            *l = f32::NEG_INFINITY;
        }
    }
}

fn apply_top_p(logits: &mut [f32], top_p:f32){
    if top_p>=1.0 || top_p <= 0.0{
        return;
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a , &b| logits[b].total_cmp(&logits[a]));
    let max=logits[order[0]];
    let exps:Vec<f32>=order.iter().map(|&i| (logits[i]-max).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let mut cumulative_sum = 0.0;
    let mut cutoff_index = order.len();
    for (rank, e) in exps.iter().enumerate(){
        cumulative_sum+=e/sum;
        if cumulative_sum >=top_p{
            cutoff_index = rank+1;
            break;
        }
        
    }
    for &i in &order[cutoff_index..]{
        logits[i] = f32::NEG_INFINITY;
    }
}


fn apply_min_prob(logits:&mut [f32], min_prob:f32){
    if min_prob <= 0.0{
        return; 
    }
    let max=logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let threshold=min_prob.ln() + max;
    for l in logits.iter_mut(){
        if *l < threshold{
            *l = f32::NEG_INFINITY;
        }
    }
    
}

impl Sampler{
    
    pub fn new(params:Params, vocab_size:usize)->Self{
        let rng=match params.seed{
            Some(seed)=>StdRng::seed_from_u64(seed),
            None=>StdRng::from_entropy(),
        };
        Self{
            params,
            rng,
            vocab_size,
        }
    }

    pub fn argmax(logits:&[f32])->Result<usize>{
        logits.iter().enumerate().max_by(|(_,a),(_,b)| a.total_cmp(b)).map(|(i,_)| i).ok_or_else(|| anyhow::anyhow!("Logits is empty"))
    }

    pub fn sample_from(logits:&[f32],  rng: &mut StdRng)->u32{
        let max=logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exps=Vec::with_capacity(logits.len());
        let mut total=0.0f32;
        for &l in logits{
            let e=(l-max).exp();
            total+=e;
            exps.push(e);
        }
        let r=rng.gen::<f32>()*total;
        let mut cumulative=0.0f32;
        for (i,&e) in exps.iter().enumerate(){
            cumulative+=e;
            if cumulative >= r{
                return i as u32;
            }
        }
        logits.iter().rposition(|&l| l>f32::NEG_INFINITY).unwrap() as u32
    }


    pub fn sample(&mut self, logits: &Tensor, prev : &[u32])->Result<u32>{
        let mut v=logits.flatten_all()?.to_vec1::<f32>()?;
        v.truncate(self.vocab_size);
        if let Some(p)=self.params.repetition_penalty{
            apply_repetition_penalty(&mut v, prev, p);
        }
        if self.params.temperature==0.0{
            return Self::argmax(&v).map(|i| i as u32);
        }
        apply_temperature(&mut v, self.params.temperature);
        if let Some(k)=self.params.top_k{
            apply_top_k(&mut v, k);
        }
        if let Some(p)=self.params.top_p{
            apply_top_p(&mut v, p);
        }
        if let Some(m)=self.params.min_prob{
            apply_min_prob(&mut v, m);
        }
        Ok(Self::sample_from(&v, &mut self.rng))

    }


}

// ===========================================================================
// WRITTEN BY CLAUDE — unit tests for the sampling knobs.
//
// These are pure functions over a Vec<f32>: no model, no device, no RNG except
// where seeding is the thing under test. They run in milliseconds and cover the
// paths a generation run never exercises.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Count entries that survived truncation.
    fn kept(v: &[f32]) -> usize {
        v.iter().filter(|x| x.is_finite()).count()
    }

    #[test]
    fn top_k_1_equals_argmax() {
        let mut v = vec![1.0, 5.0, 3.0, -2.0];
        apply_top_k(&mut v, 1);
        assert_eq!(kept(&v), 1);
        assert_eq!(v[1], 5.0, "the surviving entry must be the max");
    }

    #[test]
    fn top_k_keeps_exactly_k() {
        let mut v = vec![1.0, 5.0, 3.0, -2.0, 4.0];
        apply_top_k(&mut v, 3);
        assert_eq!(kept(&v), 3);
        // 5.0, 4.0, 3.0 survive; 1.0 and -2.0 do not.
        assert!(v[0].is_infinite() && v[3].is_infinite());
    }

    #[test]
    fn top_k_disabled_is_a_noop() {
        let orig = vec![1.0, 5.0, 3.0];
        for k in [0, 3, 99] {
            let mut v = orig.clone();
            apply_top_k(&mut v, k);
            assert_eq!(v, orig, "top_k={k} should not truncate");
        }
    }

    #[test]
    fn top_p_keeps_the_token_that_crosses() {
        // softmax([3,2,1,0]) ~ [0.644, 0.237, 0.087, 0.032]
        // cumulative:          [0.644, 0.881, 0.968, 1.000]
        let mut v = vec![3.0, 2.0, 1.0, 0.0];
        apply_top_p(&mut v, 0.8);
        // 0.644 < 0.8, so the second token crosses the line and is INCLUDED.
        assert_eq!(kept(&v), 2, "must include the token that crosses p");
    }

    #[test]
    fn top_p_always_keeps_at_least_one() {
        let mut v = vec![10.0, 0.0, 0.0];
        apply_top_p(&mut v, 0.1); // top token alone already exceeds p
        assert_eq!(kept(&v), 1);
    }

    #[test]
    fn top_p_disabled_is_a_noop() {
        let orig = vec![3.0, 2.0, 1.0];
        for p in [1.0, 1.5, 0.0, -1.0] {
            let mut v = orig.clone();
            apply_top_p(&mut v, p);
            assert_eq!(v, orig, "top_p={p} should not truncate");
        }
    }

    #[test]
    fn min_prob_thresholds_relative_to_the_max() {
        // ln(0.5) + 3.0 = 2.307, so only logits >= 2.307 survive.
        let mut v = vec![3.0, 2.5, 2.0, 0.0];
        apply_min_prob(&mut v, 0.5);
        assert_eq!(kept(&v), 2);
        assert!(v[2].is_infinite() && v[3].is_infinite());
    }

    #[test]
    fn min_prob_zero_is_a_noop() {
        let orig = vec![3.0, 2.0, 1.0];
        let mut v = orig.clone();
        apply_min_prob(&mut v, 0.0);
        assert_eq!(v, orig);
    }

    /// The trap: dividing a NEGATIVE logit makes it less negative, i.e. more
    /// likely. A penalty must always move a logit down.
    #[test]
    fn repetition_penalty_pushes_both_signs_down() {
        let mut v = vec![4.0, -4.0, 1.0];
        apply_repetition_penalty(&mut v, &[0, 1], 2.0);
        assert_eq!(v[0], 2.0, "positive logit divided");
        assert_eq!(v[1], -8.0, "negative logit multiplied");
        assert_eq!(v[2], 1.0, "unseen token untouched");
        assert!(v[0] < 4.0 && v[1] < -4.0, "both must move DOWN");
    }

    #[test]
    fn repetition_penalty_applies_once_per_unique_token() {
        let mut v = vec![4.0, 0.0];
        apply_repetition_penalty(&mut v, &[0, 0, 0, 0], 2.0);
        assert_eq!(v[0], 2.0, "repeats must not compound");
    }

    #[test]
    fn repetition_penalty_ignores_ids_past_the_vocab() {
        let mut v = vec![4.0, 1.0];
        apply_repetition_penalty(&mut v, &[0, 999_999], 2.0); // must not panic
        assert_eq!(v[0], 2.0);
    }

    #[test]
    fn temperature_scales_and_one_is_a_noop() {
        let mut v = vec![2.0, 4.0];
        apply_temperature(&mut v, 2.0);
        assert_eq!(v, vec![1.0, 2.0]);

        let orig = vec![2.0, 4.0];
        let mut v = orig.clone();
        apply_temperature(&mut v, 1.0);
        assert_eq!(v, orig);
    }

    #[test]
    fn sample_from_never_draws_a_masked_token() {
        let mut rng = StdRng::seed_from_u64(7);
        let v = vec![f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY];
        for _ in 0..200 {
            assert_eq!(Self_sample(&v, &mut rng), 1);
        }
    }

    #[test]
    fn sample_from_respects_the_distribution() {
        let mut rng = StdRng::seed_from_u64(7);
        // logit 100 vs 0: the second token is ~e^-100, effectively unreachable.
        let v = vec![100.0, 0.0];
        for _ in 0..200 {
            assert_eq!(Self_sample(&v, &mut rng), 0);
        }
    }

    /// The gate item, at the unit level: same seed => same stream.
    #[test]
    fn same_seed_gives_the_same_draws() {
        let v = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let draw = |seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..50).map(|_| Self_sample(&v, &mut rng)).collect::<Vec<_>>()
        };
        assert_eq!(draw(42), draw(42), "same seed must replay");
        assert_ne!(draw(42), draw(43), "different seeds must diverge");
    }

    // `sample_from` is an associated fn on Sampler; this is a shorthand.
    fn Self_sample(v: &[f32], rng: &mut StdRng) -> u32 {
        Sampler::sample_from(v, rng)
    }
}