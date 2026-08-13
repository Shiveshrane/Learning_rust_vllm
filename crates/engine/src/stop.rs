// ===========================================================================
// WRITTEN BY CLAUDE — Day 2 Block 3, stop conditions.
//
// EOS and max_tokens are trivial. Stop strings are not: you cannot emit a token
// until you know it is not the beginning of one. `"</"` might be the start of
// `"</s>"` or might be literal text, and you only find out on the next token.
//
// So this holds back the longest suffix of the generated text that could still
// grow into a stop string, and releases it once that becomes impossible. On a
// real match the stop string is truncated out of the output rather than shipped.
// ===========================================================================

/// Why generation ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    MaxTokens,
    StopString(String),
}

/// The result of feeding one token in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    /// Text that is now safe to emit. May be empty while a partial match is held.
    pub text: String,
    /// `Some` if generation should end here.
    pub stop: Option<StopReason>,
}

pub struct Stopper {
    eos: u32,
    max_tokens: usize,
    stop_strings: Vec<String>,
    generated: usize,
    /// Everything decoded so far (generated portion only, not the prompt).
    text: String,
    /// Byte offset of `text` already released to the caller.
    emitted: usize,
}

impl Stopper {
    pub fn new(eos: u32, max_tokens: usize, stop_strings: Vec<String>) -> Self {
        Self {
            eos,
            max_tokens,
            // An empty stop string would match everywhere; drop them up front.
            stop_strings: stop_strings.into_iter().filter(|s| !s.is_empty()).collect(),
            generated: 0,
            text: String::new(),
            emitted: 0,
        }
    }

    /// Feed one generated token plus its decoded text.
    ///
    /// `piece` is what this token decodes to on its own. For byte-level BPE that
    /// can be an incomplete character — see `server`'s incremental
    /// detokenization, which is the layer that should feed this valid UTF-8.
    pub fn push(&mut self, token: u32, piece: &str) -> Step {
        // EOS: nothing further can arrive, so any held-back tail can never grow
        // into a stop string. Flush it.
        if token == self.eos {
            return Step {
                text: self.flush(),
                stop: Some(StopReason::Eos),
            };
        }

        self.generated += 1;
        self.text.push_str(piece);

        // A completed stop string wins: emit up to it, discard it and anything
        // after, and end.
        if let Some((idx, matched)) = self.earliest_match() {
            let upto = idx.max(self.emitted);
            let out = self.text[self.emitted..upto].to_string();
            self.emitted = upto;
            return Step {
                text: out,
                stop: Some(StopReason::StopString(matched)),
            };
        }

        // Otherwise release everything that cannot be part of a future match.
        let hold = self.held_back_len();
        let mut safe = self.text.len() - hold;
        while safe > self.emitted && !self.text.is_char_boundary(safe) {
            safe -= 1; // never split a multi-byte character
        }
        let out = if safe > self.emitted {
            let s = self.text[self.emitted..safe].to_string();
            self.emitted = safe;
            s
        } else {
            String::new()
        };

        if self.generated >= self.max_tokens {
            let mut out = out;
            out.push_str(&self.flush());
            return Step {
                text: out,
                stop: Some(StopReason::MaxTokens),
            };
        }

        Step { text: out, stop: None }
    }

    /// Release everything still held back. Call when generation ends for a
    /// reason this type did not decide (e.g. the caller's own break).
    pub fn flush(&mut self) -> String {
        let out = self.text[self.emitted..].to_string();
        self.emitted = self.text.len();
        out
    }

    /// The full generated text, minus any truncated stop string.
    pub fn text(&self) -> &str {
        &self.text[..self.emitted]
    }

    pub fn generated(&self) -> usize {
        self.generated
    }

    /// Byte index of the earliest completed stop string, with the string itself.
    fn earliest_match(&self) -> Option<(usize, String)> {
        self.stop_strings
            .iter()
            .filter_map(|s| self.text.find(s.as_str()).map(|i| (i, s.clone())))
            .min_by_key(|(i, _)| *i)
    }

    /// Length in bytes of the longest suffix of `text` that is a strict prefix
    /// of some stop string — i.e. the part that might still become a match.
    fn held_back_len(&self) -> usize {
        let mut hold = 0;
        for s in &self.stop_strings {
            // Strict prefix only: a full match was already handled above.
            let max_n = (s.len() - 1).min(self.text.len());
            for n in (hold + 1..=max_n).rev() {
                let start = self.text.len() - n;
                if !self.text.is_char_boundary(start) {
                    continue;
                }
                if s.as_bytes().starts_with(&self.text.as_bytes()[start..]) {
                    hold = n;
                    break;
                }
            }
        }
        hold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: u32 = 151643;

    fn stopper(stops: &[&str], max: usize) -> Stopper {
        Stopper::new(EOS, max, stops.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn eos_stops_and_flushes() {
        let mut s = stopper(&[], 100);
        assert_eq!(s.push(1, "hi").stop, None);
        let step = s.push(EOS, "");
        assert_eq!(step.stop, Some(StopReason::Eos));
    }

    #[test]
    fn max_tokens_stops() {
        let mut s = stopper(&[], 2);
        assert_eq!(s.push(1, "a").stop, None);
        assert_eq!(s.push(2, "b").stop, Some(StopReason::MaxTokens));
    }

    #[test]
    fn plain_text_flows_straight_through() {
        let mut s = stopper(&["</s>"], 100);
        assert_eq!(s.push(1, "hello ").text, "hello ");
        assert_eq!(s.push(2, "world").text, "world");
    }

    /// The reason this type exists: `"</"` must be withheld until the next
    /// token proves it is not the start of `"</s>"`.
    #[test]
    fn partial_match_is_held_then_released() {
        let mut s = stopper(&["</s>"], 100);
        assert_eq!(s.push(1, "ok").text, "ok");
        assert_eq!(s.push(2, "</").text, "", "must hold a possible prefix");
        assert_eq!(s.push(3, "x").text, "</x", "released once impossible");
    }

    #[test]
    fn completed_stop_string_is_truncated_out() {
        let mut s = stopper(&["</s>"], 100);
        s.push(1, "answer");
        let step = s.push(2, "</s>");
        assert_eq!(step.stop, Some(StopReason::StopString("</s>".into())));
        assert_eq!(s.text(), "answer", "stop string must not be shipped");
    }

    #[test]
    fn stop_string_split_across_tokens() {
        let mut s = stopper(&["STOP"], 100);
        assert_eq!(s.push(1, "go ST").text, "go ");
        assert_eq!(s.push(2, "O").text, "");
        let step = s.push(3, "P");
        assert_eq!(step.stop, Some(StopReason::StopString("STOP".into())));
        assert_eq!(s.text(), "go ");
    }

    #[test]
    fn earliest_of_several_stop_strings_wins() {
        let mut s = stopper(&["END", "X"], 100);
        let step = s.push(1, "aXbEND");
        assert_eq!(step.stop, Some(StopReason::StopString("X".into())));
        assert_eq!(s.text(), "a");
    }

    /// Held-back text must never split a multi-byte character.
    #[test]
    fn multibyte_is_not_split() {
        let mut s = stopper(&["世界"], 100);
        let step = s.push(1, "你好世");
        assert_eq!(step.text, "你好", "hold the partial match, keep chars whole");
        let step = s.push(2, "界");
        assert_eq!(step.stop, Some(StopReason::StopString("世界".into())));
        assert_eq!(s.text(), "你好");
    }

    #[test]
    fn empty_stop_strings_are_ignored() {
        let mut s = stopper(&[""], 100);
        assert_eq!(s.push(1, "abc").text, "abc");
    }
}
