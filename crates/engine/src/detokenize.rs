use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::DecodeStream;

type Stream<'a>=DecodeStream<'a, ModelWrapper, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper>;

pub struct Detokenizer<'tok>{
    stream:Stream<'tok>,
}

impl <'tok> Detokenizer<'tok>{
    pub fn new(tok: &'tok tokenizers::Tokenizer)->Self{
        let stream=tok.decode_stream(false);
        Self{stream}
    }

    pub fn push(&mut self, token:u32)->anyhow::Result<String>{
        match self.stream.step(token).map_err(anyhow::Error::from_boxed)?{
            Some(s)=>Ok(s),
            None=>Ok(String::new()),
        }
    }
}

// ===========================================================================
// WRITTEN BY CLAUDE — Day 2 Block 4, incremental detokenization tests.
//
// Qwen's vocab is large enough that common CJK and popular emoji are SINGLE
// tokens, so they survive naive per-token decoding. The cases that break are
// rarer characters, ZWJ sequences and non-Latin scripts, which split into
// several byte-level tokens. `naive_per_token_decode_is_broken` pins that so
// the tests below are demonstrably testing something real.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use qwen::paths::ModelPaths;

    /// Characters that split across multiple tokens in this checkpoint.
    const SPLIT: &[&str] = &[
        "🫎",        // 3 tokens
        "🧑\u{200d}🚀", // 4 tokens — ZWJ, mangling it yields two separate emoji
        "ᚠᚢᚦ",      // 6 tokens
        "न्य",        // 3 tokens
        "𓀀",        // 3 tokens
    ];

    fn tokenizer() -> tokenizers::Tokenizer {
        let path = ModelPaths::from_cache().expect("model in HF cache");
        tokenizers::Tokenizer::from_file(&path.tokenizer).expect("tokenizer.json")
    }

    fn ids_of(tok: &tokenizers::Tokenizer, s: &str) -> Vec<u32> {
        tok.encode(s, false).expect("encode").get_ids().to_vec()
    }

    /// Feed ids one at a time; return the concatenated output and how many
    /// pushes actually produced text.
    fn stream(tok: &tokenizers::Tokenizer, ids: &[u32]) -> (String, usize) {
        let mut d = Detokenizer::new(tok);
        let (mut out, mut chunks) = (String::new(), 0);
        for &id in ids {
            let piece = d.push(id).expect("push");
            if !piece.is_empty() {
                chunks += 1;
            }
            out.push_str(&piece);
        }
        (out, chunks)
    }

    /// The bug this module exists to fix. If this ever starts passing, the
    /// tokenizer changed and the tests below stop proving anything.
    #[test]
    fn naive_per_token_decode_is_broken() {
        let tok = tokenizer();
        for s in SPLIT {
            let ids = ids_of(&tok, s);
            assert!(ids.len() > 1, "{s:?} should split into several tokens");
            let naive: String = ids
                .iter()
                .map(|&i| tok.decode(&[i], false).unwrap())
                .collect();
            assert_ne!(naive, *s, "{s:?} unexpectedly survived naive decoding");
            assert!(naive.contains('\u{FFFD}'), "{s:?} should show replacement chars");
        }
    }

    #[test]
    fn split_characters_round_trip() {
        let tok = tokenizer();
        for s in SPLIT {
            let (out, _) = stream(&tok, &ids_of(&tok, s));
            assert_eq!(out, *s, "streamed output differs for {s:?}");
            assert!(!out.contains('\u{FFFD}'), "replacement char in {s:?}");
        }
    }

    /// The control. A detokenizer that simply buffers everything would pass
    /// every test above and destroy streaming — ASCII must arrive in pieces.
    #[test]
    fn ascii_still_streams_incrementally() {
        let tok = tokenizer();
        let text = "The capital of France is Paris";
        let ids = ids_of(&tok, text);
        let (out, chunks) = stream(&tok, &ids);
        assert_eq!(out, text);
        assert!(
            chunks >= ids.len() - 1,
            "expected ~{} chunks, got {chunks} — is it buffering?",
            ids.len()
        );
    }

    /// Mixed content: single-token CJK next to a split emoji.
    #[test]
    fn mixed_scripts_round_trip() {
        let tok = tokenizer();
        let text = "你好 hello 🫎 world ᚠ";
        let (out, _) = stream(&tok, &ids_of(&tok, text));
        assert_eq!(out, text);
        assert!(!out.contains('\u{FFFD}'));
    }
}
