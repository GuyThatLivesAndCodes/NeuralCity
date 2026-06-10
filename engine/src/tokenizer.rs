//! Tokenization for next-token-prediction networks.
//!
//! Four modes, each a strict superset of the previous one:
//!
//! - **Char**: vocab is the set of unique characters in the corpus.
//! - **Subword**: vocab is Char + a Byte-Pair-Encoding merge sequence learned
//!   from the corpus, up to a user budget.
//! - **Word**: vocab is Subword + the top-N whitespace-delimited words in
//!   the corpus (also indexed by frequency).
//! - **Advanced**: vocab is whatever the user supplies verbatim. Reserved
//!   special tokens are inserted if missing.
//!
//! Encoding is greedy longest-match for every mode (Char devolves to one-char
//! tokens because the vocab only contains chars). Characters that aren't in
//! the vocab become `<unk>`.
//!
//! Reserved indices:
//!   0 = `<pad>`, 1 = `<unk>`, 2 = `<bos>`, 3 = `<eos>`,
//!   4 = `<user>`, 5 = `<assistant>`
//!
//! `<user>` / `<assistant>` mark conversation turns for fine-tuned chat
//! models. The fine-tuning encoding is:
//!   `<user> {prompt} <eos> <assistant> {reply} <eos>`

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;
pub const BOS_ID: u32 = 2;
pub const EOS_ID: u32 = 3;
pub const USER_ID: u32 = 4;
pub const ASSISTANT_ID: u32 = 5;
pub const RESERVED: usize = 6;

const PAD: &str = "<pad>";
const UNK: &str = "<unk>";
const BOS: &str = "<bos>";
const EOS: &str = "<eos>";
const USER: &str = "<user>";
const ASSISTANT: &str = "<assistant>";

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerMode {
    Char,
    Subword,
    Word,
    Advanced,
}

impl TokenizerMode {
    pub fn name(&self) -> &'static str {
        match self {
            TokenizerMode::Char     => "Char",
            TokenizerMode::Subword  => "Subword",
            TokenizerMode::Word     => "Word",
            TokenizerMode::Advanced => "Advanced",
        }
    }
}

/// Tuning knobs for vocabulary construction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VocabularyOptions {
    /// Subword/Word: how many BPE merges to learn. 0 disables BPE.
    pub subword_merges: usize,
    /// Word: how many distinct words to add (by descending frequency).
    pub word_top_n: usize,
}

impl Default for VocabularyOptions {
    fn default() -> Self { Self { subword_merges: 200, word_top_n: 500 } }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vocabulary {
    pub mode: TokenizerMode,
    /// Index → token string. Indices 0..RESERVED are reserved special tokens.
    pub tokens: Vec<String>,
    /// Token string → index lookup. Rebuilt by `rebuild_index` after deser.
    #[serde(skip)]
    index: HashMap<String, u32>,
    /// Tokens sorted by length descending — accelerates greedy match.
    #[serde(skip)]
    sorted_lengths: Vec<usize>,
}

impl Vocabulary {
    fn empty(mode: TokenizerMode) -> Self {
        let tokens = vec![
            PAD.into(), UNK.into(), BOS.into(), EOS.into(),
            USER.into(), ASSISTANT.into(),
        ];
        let mut v = Self { mode, tokens, index: HashMap::new(), sorted_lengths: Vec::new() };
        v.rebuild_index();
        v
    }

    pub fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, t) in self.tokens.iter().enumerate() {
            self.index.insert(t.clone(), i as u32);
        }
        let mut lens: Vec<usize> = self.tokens.iter()
            .enumerate()
            .filter(|(i, _)| *i >= RESERVED) // never match a reserved literal in user text
            .map(|(_, t)| t.chars().count())
            .collect();
        lens.sort_unstable_by(|a, b| b.cmp(a));
        lens.dedup();
        self.sorted_lengths = lens;
    }

    pub fn size(&self) -> usize { self.tokens.len() }
    pub fn id_of(&self, token: &str) -> u32 { self.index.get(token).copied().unwrap_or(UNK_ID) }
    pub fn token_of(&self, id: u32) -> &str {
        self.tokens.get(id as usize).map(String::as_str).unwrap_or(UNK)
    }

    fn add(&mut self, token: String) -> u32 {
        if let Some(&id) = self.index.get(&token) { return id; }
        let id = self.tokens.len() as u32;
        self.index.insert(token.clone(), id);
        self.tokens.push(token);
        id
    }

    /// Build a vocabulary from a corpus and explicit mode/options.
    pub fn build(mode: TokenizerMode, corpus_texts: &[&str], opts: &VocabularyOptions) -> Self {
        let mut vocab = Vocabulary::empty(mode);

        // Always add all unique characters (Char mode floor).
        for text in corpus_texts {
            for c in text.chars() {
                vocab.add(c.to_string());
            }
        }
        if mode == TokenizerMode::Char {
            vocab.rebuild_index();
            return vocab;
        }

        // Subword: learn BPE merges over the concatenated corpus.
        if opts.subword_merges > 0 {
            let merges = learn_bpe(corpus_texts, opts.subword_merges);
            for m in merges {
                if !vocab.index.contains_key(&m) {
                    vocab.add(m);
                }
            }
        }
        if mode == TokenizerMode::Subword {
            vocab.rebuild_index();
            return vocab;
        }

        // Word: add the top-N whole words.
        if opts.word_top_n > 0 {
            let mut counts: HashMap<String, usize> = HashMap::new();
            for text in corpus_texts {
                for w in split_words(text) {
                    *counts.entry(w).or_insert(0) += 1;
                }
            }
            let mut by_freq: Vec<(String, usize)> = counts.into_iter().collect();
            by_freq.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for (w, _) in by_freq.into_iter().take(opts.word_top_n) {
                if !vocab.index.contains_key(&w) {
                    vocab.add(w);
                }
            }
        }
        vocab.rebuild_index();
        vocab
    }

    /// Build an Advanced vocab from explicit tokens. Reserved tokens are
    /// inserted at the front if not already present.
    pub fn build_advanced(user_tokens: &[String]) -> Self {
        let mut vocab = Vocabulary::empty(TokenizerMode::Advanced);
        for t in user_tokens {
            if t.is_empty() { continue; }
            // Skip reserved literals if user re-supplied them.
            if matches!(t.as_str(), PAD | UNK | BOS | EOS | USER | ASSISTANT) { continue; }
            if !vocab.index.contains_key(t) {
                vocab.add(t.clone());
            }
        }
        vocab.rebuild_index();
        vocab
    }

    /// Greedy longest-match tokenization. Characters with no matching vocab
    /// entry are emitted as `<unk>`.
    ///
    /// Candidate substrings are taken as zero-allocation `&str` slices of the
    /// input via precomputed char byte-offsets, instead of collecting a fresh
    /// `String` for every (position, length) pair. Since `HashMap<String, _>`
    /// can be queried with a `&str` (via `Borrow<str>`), each candidate is a
    /// plain slice + hash lookup with no heap traffic. This is the hot path for
    /// turning a whole corpus into training sequences.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Byte offset of every char start, plus a trailing sentinel == text.len()
        // so `text[offsets[i]..offsets[i + len]]` is always a valid char-aligned
        // slice for any `i + len <= n_chars`.
        let mut offsets: Vec<usize> = Vec::with_capacity(text.len() + 1);
        for (b, _) in text.char_indices() { offsets.push(b); }
        offsets.push(text.len());
        let n_chars = offsets.len() - 1;

        let mut out = Vec::with_capacity(n_chars);
        let mut i = 0;
        while i < n_chars {
            let mut matched = false;
            let remaining = n_chars - i;
            for &len in &self.sorted_lengths {
                // `sorted_lengths` is descending, so once a length fits, longer
                // ones were already skipped; we keep scanning down for a match.
                if len == 0 || len > remaining { continue; }
                let candidate = &text[offsets[i]..offsets[i + len]];
                if let Some(&id) = self.index.get(candidate) {
                    out.push(id);
                    i += len;
                    matched = true;
                    break;
                }
            }
            if !matched {
                // Try a single-character lookup (Char-mode fallback / any mode).
                let single = &text[offsets[i]..offsets[i + 1]];
                if let Some(&id) = self.index.get(single) {
                    out.push(id);
                } else {
                    out.push(UNK_ID);
                }
                i += 1;
            }
        }
        out
    }

    /// Decode token IDs back to text. Reserved tokens are dropped.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if (id as usize) < RESERVED { continue; }
            out.push_str(self.token_of(id));
        }
        out
    }
}

// ─── Word splitter (whitespace + punctuation kept inline) ───────────────────

fn split_words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_whitespace() || c.is_ascii_punctuation() {
            if !current.is_empty() { out.push(std::mem::take(&mut current)); }
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() { out.push(current); }
    out
}

// ─── Byte-pair encoding (over chars, not bytes — same algorithm) ────────────

/// Learn a sequence of `n_merges` BPE merges on the concatenated corpus.
/// Returns the merged token strings (in the order they were learned).
fn learn_bpe(corpus_texts: &[&str], n_merges: usize) -> Vec<String> {
    if n_merges == 0 { return Vec::new(); }

    // Represent each word as a vector of "symbols" (initially single chars).
    // A word boundary token isn't needed — we operate within each word so the
    // merges remain interpretable as substrings.
    let mut word_freq: HashMap<Vec<String>, usize> = HashMap::new();
    for text in corpus_texts {
        for w in split_words(text) {
            let symbols: Vec<String> = w.chars().map(|c| c.to_string()).collect();
            if symbols.is_empty() { continue; }
            *word_freq.entry(symbols).or_insert(0) += 1;
        }
    }

    let mut merges: Vec<String> = Vec::with_capacity(n_merges);
    for _ in 0..n_merges {
        // Count adjacent pair frequencies.
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        for (symbols, freq) in &word_freq {
            for window in symbols.windows(2) {
                let key = (window[0].clone(), window[1].clone());
                *pair_counts.entry(key).or_insert(0) += freq;
            }
        }
        // Pick the most frequent pair (deterministic tie-break on the pair).
        let best = pair_counts.into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)));
        let Some((pair, count)) = best else { break };
        if count < 2 { break; }
        let merged = format!("{}{}", pair.0, pair.1);
        merges.push(merged.clone());

        // Apply the merge to every word.
        let entries: Vec<(Vec<String>, usize)> = word_freq.drain().collect();
        for (symbols, freq) in entries {
            let mut new_syms: Vec<String> = Vec::with_capacity(symbols.len());
            let mut i = 0;
            while i < symbols.len() {
                if i + 1 < symbols.len()
                    && symbols[i] == pair.0
                    && symbols[i + 1] == pair.1
                {
                    new_syms.push(merged.clone());
                    i += 2;
                } else {
                    new_syms.push(symbols[i].clone());
                    i += 1;
                }
            }
            *word_freq.entry(new_syms).or_insert(0) += freq;
        }
    }
    merges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_mode_contains_only_chars() {
        let v = Vocabulary::build(TokenizerMode::Char, &["abca"], &VocabularyOptions::default());
        // reserved + a, b, c
        assert_eq!(v.size(), RESERVED + 3);
    }

    #[test]
    fn reserved_tokens_include_chat_markers() {
        let v = Vocabulary::build(TokenizerMode::Char, &["abc"], &VocabularyOptions::default());
        assert_eq!(v.token_of(USER_ID), "<user>");
        assert_eq!(v.token_of(ASSISTANT_ID), "<assistant>");
    }

    #[test]
    fn subword_mode_includes_chars() {
        let v = Vocabulary::build(
            TokenizerMode::Subword,
            &["the cat sat on the mat the cat"],
            &VocabularyOptions { subword_merges: 5, word_top_n: 0 },
        );
        // chars: 't','h','e',' ','c','a','s','o','n','m' = 10
        assert!(v.size() >= RESERVED + 10);
        // After merges, vocab should have learned at least one combo token.
        let merged_count = v.tokens.iter().filter(|t| t.chars().count() > 1).count();
        assert!(merged_count > 0, "subword mode should produce multi-char tokens");
    }

    #[test]
    fn word_mode_includes_whole_words() {
        let v = Vocabulary::build(
            TokenizerMode::Word,
            &["the cat sat the cat sat the cat"],
            &VocabularyOptions { subword_merges: 0, word_top_n: 10 },
        );
        // Whole words "the", "cat", "sat" should be present.
        assert!(v.index.contains_key("the"));
        assert!(v.index.contains_key("cat"));
        assert!(v.index.contains_key("sat"));
    }

    #[test]
    fn greedy_match_prefers_longest() {
        let v = Vocabulary::build(
            TokenizerMode::Word,
            &["the cat the cat"],
            &VocabularyOptions { subword_merges: 0, word_top_n: 5 },
        );
        let ids = v.encode("the cat");
        // Should match "the" as one token (since it's in vocab), then " ", then "cat".
        // First and last should be the word tokens, not 3 chars each.
        assert_eq!(v.token_of(ids[0]), "the");
        assert_eq!(v.token_of(*ids.last().unwrap()), "cat");
    }

    #[test]
    fn unknown_char_becomes_unk() {
        let v = Vocabulary::build(TokenizerMode::Char, &["abc"], &VocabularyOptions::default());
        let ids = v.encode("abx");
        assert_eq!(ids[2], UNK_ID);
    }

    #[test]
    fn advanced_mode_takes_user_tokens() {
        let toks: Vec<String> = vec!["hello".into(), "world".into()];
        let v = Vocabulary::build_advanced(&toks);
        assert_eq!(v.size(), RESERVED + 2);
        let ids = v.encode("helloworld");
        assert_eq!(v.token_of(ids[0]), "hello");
        assert_eq!(v.token_of(ids[1]), "world");
    }

    #[test]
    fn encode_handles_multibyte_chars() {
        // Multi-byte UTF-8 chars must tokenize by character, not by byte: the
        // offset-based slicing in `encode` has to stay char-aligned.
        let v = Vocabulary::build(
            TokenizerMode::Word,
            &["héllo wörld héllo wörld"],
            &VocabularyOptions { subword_merges: 0, word_top_n: 5 },
        );
        let ids = v.encode("héllo wörld");
        // Greedy longest-match should recover the whole words verbatim.
        assert_eq!(v.token_of(ids[0]), "héllo");
        assert_eq!(v.token_of(*ids.last().unwrap()), "wörld");
        // Round-trip through decode reproduces the input exactly.
        assert_eq!(v.decode(&ids), "héllo wörld");
    }

    #[test]
    fn decode_roundtrip() {
        let v = Vocabulary::build(
            TokenizerMode::Word,
            &["the cat sat the cat sat"],
            &VocabularyOptions { subword_merges: 0, word_top_n: 10 },
        );
        let s = "the cat sat";
        let ids = v.encode(s);
        let back = v.decode(&ids);
        assert_eq!(back, s);
    }
}
