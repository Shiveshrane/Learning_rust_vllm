use candle_core::{DType, Tensor, D};
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVDType{
    F32,
    Int8,
}


const MIN_SCALE:f64=1e-12;


pub fn quantize_token(x:&Tensor)->Result<(Tensor, Tensor)>{
    let scale=x.abs()?.max_keepdim(D::Minus1)?
    .affine(1.0/127.0, 0.0)?
    .clamp(MIN_SCALE, f64::INFINITY)?;

    let codes=x.broadcast_div(&scale)?
    .round()?
    .clamp(-127.0, 127.0)?
    .affine(1.0, 128.0)?
    .to_dtype(DType::U8)?;

    Ok((codes, scale))
}

pub fn dequantize_token(codes:&Tensor, scale:&Tensor)->Result<Tensor>{
    let x=codes.to_dtype(DType::F32)?
    .affine(1.0, -128.0)?
    .broadcast_mul(scale)?;
    Ok(x)
}





// ===========================================================================
// TESTS WRITTEN BY CLAUDE — Day 4 Block 1, symmetric int8 quantization.
//
// Pure tensor maths on CPU: no model, no pool, milliseconds. The important one
// is `a_single_outlier_destroys_its_group` — that is the entire argument for
// caring about grouping, and it is what will explain the number moving when K
// switches from per-token to per-channel.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn t(v: Vec<f32>, shape: (usize, usize)) -> Tensor {
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    fn flat(x: &Tensor) -> Vec<f32> {
        x.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The definition of the quantizer working: every value lands within half a
    /// step of where it started. Not "looks close" — a bound derived from the
    /// scale, asserted per element.
    #[test]
    fn round_trip_error_is_within_half_a_step() {
        let row = vec![3.0, -7.5, 0.25, 7.5, -0.1, 2.2, -4.4, 6.0];
        let x = t(row.clone(), (1, 8));
        let (q, s) = quantize_token(&x).unwrap();
        let back = dequantize_token(&q, &s).unwrap();

        let scale = flat(&s)[0];
        let half_step = scale / 2.0 + 1e-6;
        for (a, b) in row.iter().zip(flat(&back)) {
            assert!(
                (a - b).abs() <= half_step,
                "{a} -> {b}, error {} exceeds half-step {half_step}",
                (a - b).abs()
            );
        }
        // scale is set by the largest magnitude in the group
        assert!((scale - 7.5 / 127.0).abs() < 1e-9);
    }

    /// Why grouping is the whole question. One value 100x its neighbours sets
    /// the scale for everyone, and the small values collapse into the same code.
    #[test]
    fn a_single_outlier_destroys_its_group() {
        let clean = t(vec![1.0, 2.0, 3.0, 4.0], (1, 4));
        let dirty = t(vec![400.0, 1.0, 2.0, 3.0], (1, 4));

        let (qc, sc) = quantize_token(&clean).unwrap();
        let (qd, sd) = quantize_token(&dirty).unwrap();
        let bc = flat(&dequantize_token(&qc, &sc).unwrap());
        let bd = flat(&dequantize_token(&qd, &sd).unwrap());

        // Clean group: the small values survive.
        let clean_err: f32 = [1.0, 2.0, 3.0]
            .iter()
            .enumerate()
            .map(|(i, v)| (v - bc[i]).abs())
            .fold(0.0, f32::max);

        // Poisoned group: the same magnitudes, now sharing a scale 100x larger.
        let dirty_err: f32 = [1.0, 2.0, 3.0]
            .iter()
            .enumerate()
            .map(|(i, v)| (v - bd[i + 1]).abs())
            .fold(0.0, f32::max);

        println!("\n  clean group max err: {clean_err:.6}");
        println!("  outlier group max err: {dirty_err:.6}  ({:.0}x worse)", dirty_err / clean_err.max(1e-9));
        assert!(
            dirty_err > clean_err * 50.0,
            "outlier should wreck its group: {dirty_err} vs {clean_err}"
        );
        // The outlier itself is still fine — it set the scale.
        assert!((400.0 - bd[0]).abs() < 4.0);
    }

    /// scale = 0 would make x/scale NaN, and NaN propagates silently through
    /// attention. MIN_SCALE is the guard.
    #[test]
    fn all_zero_group_does_not_produce_nan() {
        let x = t(vec![0.0; 6], (1, 6));
        let (q, s) = quantize_token(&x).unwrap();
        let back = flat(&dequantize_token(&q, &s).unwrap());
        assert!(back.iter().all(|v| v.is_finite()), "NaN escaped: {back:?}");
        assert!(back.iter().all(|v| *v == 0.0));
    }

    /// Codes must stay inside u8 after the +128 shift; anything outside means
    /// a clamp is missing and values wrapped.
    #[test]
    fn codes_stay_inside_u8() {
        let x = t(vec![-9.0, 9.0, 0.0, 4.5, -4.5, 1.0], (1, 6));
        let (q, _) = quantize_token(&x).unwrap();
        assert_eq!(q.dtype(), DType::U8);
        let codes = q.flatten_all().unwrap().to_vec1::<u8>().unwrap();
        assert!(codes.iter().all(|c| *c >= 1), "code wrapped below 1: {codes:?}");
        // +-max maps to the ends, zero maps to the midpoint.
        assert_eq!(codes[0], 1);
        assert_eq!(codes[1], 255);
        assert_eq!(codes[2], 128);
    }

    /// Each row gets its own scale, so a quiet row is not punished for a loud
    /// neighbour. This is what "per-token" buys.
    #[test]
    fn each_row_gets_its_own_scale() {
        let x = t(vec![1.0, 2.0, 3.0, 4.0, 400.0, 800.0, 1200.0, 1600.0], (2, 4));
        let (q, s) = quantize_token(&x).unwrap();
        assert_eq!(s.dims(), &[2, 1], "one scale per row, trailing 1 for broadcast");

        let scales = flat(&s);
        assert!(scales[1] > scales[0] * 100.0, "loud row must not set the quiet row's scale");

        let back = flat(&dequantize_token(&q, &s).unwrap());
        assert!((back[0] - 1.0).abs() < 0.02, "quiet row survived: {}", back[0]);
    }

    /// Shapes must survive so the pool can slice_set codes and scales in step.
    #[test]
    fn shapes_round_trip_for_pool_layout() {
        // [slots, kv_heads, head_dim] as the pool stores it
        let x = Tensor::from_vec(
            (0..2 * 3 * 4).map(|i| i as f32 - 12.0).collect::<Vec<f32>>(),
            (2, 3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let (q, s) = quantize_token(&x).unwrap();
        assert_eq!(q.dims(), &[2, 3, 4]);
        assert_eq!(s.dims(), &[2, 3, 1], "one scale per (slot, head)");
        assert_eq!(dequantize_token(&q, &s).unwrap().dims(), &[2, 3, 4]);
    }
}
