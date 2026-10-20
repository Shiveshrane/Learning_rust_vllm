use std::f64::consts::PI;
use crate::config::{QwenConfig, RopeScaling};


pub fn find_correction_dim(rotations:f64, dim:usize, base:f64, max_pos:usize)->f64{
    (dim as f64*(max_pos as f64/(rotations*2.0*PI)).ln())/(2.0*base.ln())
}

pub fn find_correction_range(
    beta_fast:f64, beta_slow:f64, dim:usize, base:f64, max_pos:usize
)->(usize, usize){
   let low=find_correction_dim(beta_fast, dim, base, max_pos).floor();
   let high=find_correction_dim(beta_slow, dim, base, max_pos).ceil();
   let top=(dim-1) as f64;
   (low.clamp(0.0, top) as usize, high.clamp(0.0, top) as usize)
}


pub fn linear_ramp_mask(low:f64, high:f64, n:usize)->Vec<f64>{
    let high=if high==low{high+0.001} else{high};
    (0..n).map(
        |i| (
            (i as f64-low)/(high-low)
        ).clamp(0.0, 1.0)
    ).collect()
}


pub fn get_mscale(s:f64)->f64{
    if s<=1.0{
        1.0
    }
    else{
        0.1*s.ln()+1.0
    }
}

pub fn scaled_inv_freq(cfg:&QwenConfig)->(Vec<f32>, f32){
    let dim=cfg.head_dim();
    let theta=cfg.rope_theta as f32;
    let base:Vec<f32>=(0..dim/2)
        .map(|i| 1f32/theta.powf(2.0*(i as f32)/dim as f32))
        .collect();
    match &cfg.rope_scaling{
        None=>(base, 1.0),
        Some(RopeScaling::Pi{factor})=>{
            let s=*factor as f32;
            (base.iter().map(|f| f/s).collect(), 1.0)
        },
        Some(RopeScaling::NtkAware{factor})=>{
            let nb=(cfg.rope_theta*factor.powf(dim as f64/(dim as f64-2.0))) as f32;
            ((0..dim/2).map(|i| 1f32/nb.powf(2.0*i as f32/dim as f32)).collect(), 1.0)
        },

        Some(RopeScaling::Yarn{factor, original_max_position_embeddings, beta_fast, beta_slow})=>{
            let (low, high)=find_correction_range(
                *beta_fast, *beta_slow, dim, cfg.rope_theta, *original_max_position_embeddings
            );
            let ramp=linear_ramp_mask(low as f64, high as f64, dim/2);
            let s=*factor as f32;
            let inv=base.iter().zip(&ramp).map(|(f,r)| {
                let r=*r as f32;
                f*(1.0-r)+f/s*r
            }).collect();
            (inv, get_mscale(*factor) as f32)
        }
    }
}

// ===========================================================================
// TESTS WRITTEN BY CLAUDE — Day 5 Block 1, RoPE scaling.
//
// Pure maths: no model, no device, milliseconds. The target numbers were
// worked out independently (Python) for THIS model: head_dim 128, rope_theta
// 10000, and an original training window of 4096 tokens.
//
// The ramp DIRECTION gets the most attention. An inverted ramp slows the fast
// pairs and leaves the slow ones alone. Nothing errors; YaRN just ends up worse
// than no scaling at all. It is the most likely bug of the day.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 128;
    const BASE: f64 = 10000.0;
    const L: usize = 4096;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// A config shaped like the real checkpoint (head_dim = 1536 / 12 = 128).
    fn cfg(scaling: Option<RopeScaling>) -> QwenConfig {
        QwenConfig {
            hidden_size: 1536,
            intermediate_size: 8960,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            vocab_size: 151936,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: BASE,
            tie_word_embeddings: false,
            eos_token_id: 151643,
            rope_scaling: scaling,
        }
    }

    /// Day 1's speeds, computed exactly as model.rs line 62 does.
    fn day1_speeds() -> Vec<f32> {
        let theta = BASE as f32;
        (0..D / 2)
            .map(|i| 1f32 / theta.powf(2.0 * i as f32 / D as f32))
            .collect()
    }

    // ---- the four building blocks ------------------------------------------

    /// Pair ~21 completes 32 full turns within 4096 tokens.
    #[test]
    fn correction_dim_matches_hand_computed_value() {
        let i = find_correction_dim(32.0, D, BASE, L);
        assert!(close(i, 20.944, 0.001), "got {i}");
        let j = find_correction_dim(1.0, D, BASE, L);
        assert!(close(j, 45.027, 0.001), "got {j}");
    }

    /// Fewer rotations means a slower pair, and slower pairs have HIGHER index.
    #[test]
    fn fewer_rotations_means_a_higher_pair_index() {
        let fast = find_correction_dim(32.0, D, BASE, L);
        let slow = find_correction_dim(1.0, D, BASE, L);
        assert!(fast < slow, "beta_fast must give the LOW index: {fast} vs {slow}");
    }

    /// low is floored, high is ceiled: the blend zone is never narrower than
    /// the maths says.
    #[test]
    fn correction_range_for_this_model_is_20_to_46() {
        assert_eq!(find_correction_range(32.0, 1.0, D, BASE, L), (20, 46));
    }

    #[test]
    fn correction_range_is_clamped_to_valid_pairs() {
        // Absurd rotation counts would land outside 0..D; clamp keeps them in.
        let (lo, hi) = find_correction_range(1e12, 1e-12, D, BASE, L);
        assert!(lo <= D - 1 && hi <= D - 1, "({lo}, {hi})");
    }

    #[test]
    fn ramp_values_at_known_pairs() {
        let r = linear_ramp_mask(20.0, 46.0, D / 2);
        assert_eq!(r.len(), 64);
        assert!(close(r[30], 0.3846, 0.001), "r[30] = {}", r[30]);
        assert!(close(r[40], 0.7692, 0.001), "r[40] = {}", r[40]);
    }

    /// THE direction check: fast (low) pairs 0, slow (high) pairs 1, and never
    /// decreasing in between.
    #[test]
    fn ramp_goes_from_untouched_to_fully_scaled() {
        let r = linear_ramp_mask(20.0, 46.0, D / 2);
        assert!(r[..=20].iter().all(|v| *v == 0.0), "pairs 0..=20 must be 0");
        assert!(r[46..].iter().all(|v| *v == 1.0), "pairs 46.. must be 1");
        assert!(r.windows(2).all(|w| w[0] <= w[1]), "ramp must never go down");
    }

    /// Guard against high == low: the division would be by zero.
    #[test]
    fn ramp_survives_low_equal_to_high() {
        let r = linear_ramp_mask(5.0, 5.0, 10);
        assert!(r.iter().all(|v| v.is_finite()), "{r:?}");
        assert!(r.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn mscale_is_one_without_extension() {
        assert_eq!(get_mscale(1.0), 1.0);
        assert_eq!(get_mscale(0.5), 1.0);
    }

    #[test]
    fn mscale_grows_with_the_scale_factor() {
        assert!(close(get_mscale(2.0), 1.0693, 0.0001));
        assert!(close(get_mscale(4.0), 1.1386, 0.0001));
        assert!(close(get_mscale(8.0), 1.2079, 0.0001));
    }

    // ---- scaled_inv_freq ---------------------------------------------------

    /// The gate's first item: with no scaling, the speeds are BIT-identical to
    /// Day 1. Anything else would shift the golden logits.
    #[test]
    fn no_scaling_reproduces_day1_exactly() {
        let (inv, mscale) = scaled_inv_freq(&cfg(None));
        assert_eq!(inv, day1_speeds(), "must be bit-identical to model.rs line 62");
        assert_eq!(mscale, 1.0);
    }

    /// PI divides EVERY speed — including the fast pairs, which is its flaw.
    #[test]
    fn pi_slows_every_pair_by_the_factor() {
        let (inv, mscale) = scaled_inv_freq(&cfg(Some(RopeScaling::Pi { factor: 4.0 })));
        let base = day1_speeds();
        for (i, (new, old)) in inv.iter().zip(&base).enumerate() {
            assert_eq!(*new, old / 4.0, "pair {i}");
        }
        assert_eq!(mscale, 1.0);
    }

    /// NTK-aware: a bigger base leaves pair 0 exactly alone, and slows the
    /// slowest pair by almost exactly the factor.
    #[test]
    fn ntk_leaves_fast_pairs_and_slows_slow_ones() {
        let s = 4.0;
        let (inv, mscale) = scaled_inv_freq(&cfg(Some(RopeScaling::NtkAware { factor: s })));
        let base = day1_speeds();
        assert_eq!(inv[0], base[0], "pair 0 has speed 1 under any base");
        let last = (base[63] / inv[63]) as f64;
        assert!(close(last, s, 0.01), "slowest pair should slow by ~{s}, got {last}");
        // In between, the slowdown grows steadily with the pair index.
        let ratios: Vec<f32> = base.iter().zip(&inv).map(|(b, n)| b / n).collect();
        assert!(ratios.windows(2).all(|w| w[0] <= w[1] + 1e-4));
        assert_eq!(mscale, 1.0);
    }

    fn yarn(s: f64) -> RopeScaling {
        RopeScaling::Yarn {
            factor: s,
            original_max_position_embeddings: L,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }

    /// YaRN per zone: 0..=20 untouched, 46.. divided by s, blended between.
    #[test]
    fn yarn_treats_each_zone_correctly() {
        let s = 4.0f32;
        let (inv, _) = scaled_inv_freq(&cfg(Some(yarn(4.0))));
        let base = day1_speeds();

        for i in 0..=20 {
            assert_eq!(inv[i], base[i], "pair {i} spins fast enough: must be untouched");
        }
        for i in 46..64 {
            assert_eq!(inv[i], base[i] / s, "pair {i} never completes a turn: fully scaled");
        }
        for i in 21..46 {
            assert!(
                inv[i] < base[i] && inv[i] > base[i] / s,
                "pair {i} is in the blend zone and must sit strictly between"
            );
        }
    }

    /// The slowdown must INCREASE with the pair index. An inverted ramp would
    /// make it decrease — slowing the fast pairs and sparing the slow ones.
    #[test]
    fn yarn_slows_slow_pairs_more_than_fast_ones() {
        let (inv, _) = scaled_inv_freq(&cfg(Some(yarn(4.0))));
        let base = day1_speeds();
        let ratios: Vec<f32> = base.iter().zip(&inv).map(|(b, n)| b / n).collect();
        assert_eq!(ratios[0], 1.0, "fastest pair untouched");
        assert!(close(ratios[63] as f64, 4.0, 1e-5), "slowest pair fully scaled");
        assert!(ratios.windows(2).all(|w| w[0] <= w[1] + 1e-6), "ramp inverted?");
    }

    #[test]
    fn yarn_returns_the_attention_sharpening() {
        let (_, mscale) = scaled_inv_freq(&cfg(Some(yarn(4.0))));
        assert!(close(mscale as f64, 1.1386, 0.0001), "got {mscale}");
    }

    // ---- config parsing ----------------------------------------------------

    /// The HuggingFace shape parses, and beta defaults to the paper's 32 / 1.
    #[test]
    fn hf_style_rope_scaling_block_parses() {
        let json = r#"{"type":"yarn","factor":4.0,"original_max_position_embeddings":4096}"#;
        let r: RopeScaling = serde_json::from_str(json).unwrap();
        match r {
            RopeScaling::Yarn { factor, original_max_position_embeddings, beta_fast, beta_slow } => {
                assert_eq!(factor, 4.0);
                assert_eq!(original_max_position_embeddings, 4096);
                assert_eq!((beta_fast, beta_slow), (32.0, 1.0));
            }
            other => panic!("parsed as the wrong variant: {other:?}"),
        }
        let pi: RopeScaling = serde_json::from_str(r#"{"type":"linear","factor":2.0}"#).unwrap();
        assert!(matches!(pi, RopeScaling::Pi { factor } if factor == 2.0));
    }
}
