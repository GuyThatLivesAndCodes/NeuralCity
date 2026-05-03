//! Vocabulary management for next-token-generation networks.
//!
//! A vocabulary is just an ordered list of tokens; index 0 is reserved for
//! `<unk>`. The user can build the vocab manually, upload one, derive it from
//! the corpus text in several ways, or wipe it.

use std::collections::BTreeSet;
use std::fs;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VocabMode {
    Char,
    Subword,
    Word,
    Sentence,
    Bulk,
    Custom,
}

impl VocabMode {
    pub fn name(&self) -> &'static str {
        match self {
            VocabMode::Char => "Per-character",
            VocabMode::Subword => "Per-subword",
            VocabMode::Word => "Per-word",
            VocabMode::Sentence => "Per-sentence",
            VocabMode::Bulk => "Bulk (all, deduped)",
            VocabMode::Custom => "Custom (manual)",
        }
    }

    pub fn to_str(self) -> &'static str {
        match self {
            VocabMode::Char     => "Char",
            VocabMode::Subword  => "Subword",
            VocabMode::Word     => "Word",
            VocabMode::Sentence => "Sentence",
            VocabMode::Bulk     => "Bulk",
            VocabMode::Custom   => "Custom",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Char"     => Some(VocabMode::Char),
            "Subword"  => Some(VocabMode::Subword),
            "Word"     => Some(VocabMode::Word),
            "Sentence" => Some(VocabMode::Sentence),
            "Bulk"     => Some(VocabMode::Bulk),
            "Custom"   => Some(VocabMode::Custom),
            _          => None,
        }
    }

    pub fn all() -> &'static [VocabMode] {
        &[
            VocabMode::Char,
            VocabMode::Subword,
            VocabMode::Word,
            VocabMode::Sentence,
            VocabMode::Bulk,
            VocabMode::Custom,
        ]
    }
}

/// `tokens[0]` is always the `<unk>` placeholder.
#[derive(Clone, Debug)]
pub struct Vocab {
    pub tokens: Vec<String>,
    pub mode: VocabMode,
    pub message: Option<String>,
    pub upload_path: String,
    pub draft_token: String,
}

impl Default for Vocab {
    fn default() -> Self {
        Self {
            tokens: vec!["<unk>".into()],
            mode: VocabMode::Char,
            message: None,
            upload_path: String::new(),
            draft_token: String::new(),
        }
    }
}

impl Vocab {
    pub fn len(&self) -> usize { self.tokens.len() }
    pub fn is_empty(&self) -> bool { self.tokens.len() <= 1 }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.tokens.push("<unk>".into());
        self.message = Some("Vocab cleared (only <unk> remains).".into());
    }

    pub fn add_unique(&mut self, token: String) {
        if token.is_empty() { return; }
        if !self.tokens.iter().any(|t| t == &token) {
            self.tokens.push(token);
        }
    }

    pub fn id_of(&self, token: &str) -> usize {
        self.tokens.iter().position(|t| t == token).unwrap_or(0)
    }

    /// Encode a string to a token-id sequence according to the current mode.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self.mode {
            VocabMode::Char => text.chars().map(|c| self.id_of(&c.to_string())).collect(),
            VocabMode::Word => split_words(text).into_iter().map(|w| self.id_of(&w)).collect(),
            VocabMode::Sentence => split_sentences(text).into_iter().map(|s| self.id_of(&s)).collect(),
            VocabMode::Subword => encode_greedy(self, text),
            VocabMode::Bulk => encode_greedy(self, text),
            VocabMode::Custom => encode_greedy(self, text),
        }
    }

    /// Generate a vocab from `corpus_text` according to `mode`. Replaces the
    /// existing vocab (except for `<unk>`).
    pub fn auto_generate(&mut self, corpus_text: &str, mode: VocabMode) {
        self.mode = mode;
        let mut set: BTreeSet<String> = BTreeSet::new();
        match mode {
            VocabMode::Char => {
                for c in corpus_text.chars() { set.insert(c.to_string()); }
            }
            VocabMode::Word => {
                for w in split_words(corpus_text) { set.insert(w); }
            }
            VocabMode::Sentence => {
                for s in split_sentences(corpus_text) { set.insert(s); }
            }
            VocabMode::Subword => {
                for w in split_words(corpus_text) {
                    let chars: Vec<char> = w.chars().collect();
                    for i in 0..chars.len() {
                        for j in (i + 1..=chars.len()).take(4) {
                            let frag: String = chars[i..j].iter().collect();
                            if !frag.is_empty() { set.insert(frag); }
                        }
                    }
                }
            }
            VocabMode::Bulk => {
                for c in corpus_text.chars() { set.insert(c.to_string()); }
                for w in split_words(corpus_text) { set.insert(w); }
                for s in split_sentences(corpus_text) { set.insert(s); }
            }
            VocabMode::Custom => {
                self.message = Some("Custom mode: add tokens manually.".into());
                return;
            }
        }
        self.tokens = vec!["<unk>".into()];
        self.tokens.extend(set);
        self.message = Some(format!("Generated {} tokens ({}).", self.tokens.len() - 1, mode.name()));
    }

    /// Load a vocab from a file: one token per line. The first line is the
    /// `<unk>` placeholder if present, otherwise we add it.
    pub fn load_file(&mut self, path: &str) -> Result<(), String> {
        let raw = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut tokens: Vec<String> = raw.lines().map(|l| l.to_string()).filter(|l| !l.is_empty()).collect();
        if tokens.first().map(|s| s.as_str()) != Some("<unk>") {
            tokens.insert(0, "<unk>".into());
        }
        self.tokens = tokens;
        self.message = Some(format!("Loaded {} tokens from {path}.", self.tokens.len()));
        Ok(())
    }
}

fn split_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(|s| s.to_string()).collect()
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for c in text.chars() {
        buf.push(c);
        if matches!(c, '.' | '!' | '?' | '\n') {
            let trimmed = buf.trim().to_string();
            if !trimmed.is_empty() { out.push(trimmed); }
            buf.clear();
        }
    }
    let trimmed = buf.trim().to_string();
    if !trimmed.is_empty() { out.push(trimmed); }
    out
}

fn encode_greedy(vocab: &Vocab, text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let mut best_id = 0;
        let mut best_len = 0;
        for (id, tok) in vocab.tokens.iter().enumerate().skip(1) {
            let tb = tok.as_bytes();
            if !tb.is_empty() && i + tb.len() <= bytes.len() && &bytes[i..i + tb.len()] == tb && tb.len() > best_len {
                best_id = id;
                best_len = tb.len();
            }
        }
        if best_len == 0 {
            out.push(0);
            i += 1;
        } else {
            out.push(best_id);
            i += best_len;
        }
    }
    out
}
