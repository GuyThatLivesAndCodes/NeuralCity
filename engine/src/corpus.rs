//! Convert a text corpus into a real next-token-prediction training set.
//!
//! Strategy: sliding windows of `context_size` tokens predict the next token.
//! Inputs are flattened one-hot vectors of length `context_size * vocab_size`.
//! Targets are one-hot vectors of length `vocab_size`. This works directly with
//! the existing Linear + Activation layer stack — no new layer types required.
//!
//! For fine-tuning (input/output pairs), the same shape applies: each pair is
//! tokenized into a single sequence
//! `<user> input <eos> <assistant> output <eos>` and we emit one training
//! example per output token.

use crate::tensor::Tensor;
use crate::tokenizer::{Vocabulary, ASSISTANT_ID, EOS_ID, PAD_ID, USER_ID};

/// A single fine-tuning conversation pair.
#[derive(Clone, Debug)]
pub struct Pair {
    pub input: String,
    pub output: String,
}

/// Build (X, Y) tensors from raw text using sliding windows.
///
/// `X.shape == [n_examples, context_size * vocab_size]`
/// `Y.shape == [n_examples, vocab_size]`
///
/// Returns `None` if the corpus has fewer than `context_size + 1` tokens.
pub fn build_pretraining_tensors(
    text: &str,
    vocab: &Vocabulary,
    context_size: usize,
) -> Option<(Tensor, Tensor)> {
    assert!(context_size > 0, "context_size must be positive");
    let ids = vocab.encode(text);
    if ids.len() < context_size + 1 { return None; }

    let v = vocab.size();
    let n_examples = ids.len() - context_size;

    let mut x = vec![0.0_f32; n_examples * context_size * v];
    let mut y = vec![0.0_f32; n_examples * v];

    for i in 0..n_examples {
        let window = &ids[i..i + context_size];
        let target = ids[i + context_size];
        for (pos, &tok) in window.iter().enumerate() {
            x[i * context_size * v + pos * v + tok as usize] = 1.0;
        }
        y[i * v + target as usize] = 1.0;
    }

    Some((
        Tensor::new(vec![n_examples, context_size * v], x),
        Tensor::new(vec![n_examples, v], y),
    ))
}

/// Build (X, Y) tensors from input/output pairs for fine-tuning.
///
/// Each pair is encoded as `<user> input <eos> <assistant> output <eos>`.
/// Sliding windows then teach the model to predict each token of the output
/// (and the trailing <eos>) given the prior tokens, including the explicit
/// `<assistant>` turn marker.
///
/// If `mask_user_tokens` is true, only windows whose target falls strictly in
/// the assistant region (everything after `<assistant>`) are emitted — the
/// model is not penalised for failing to reproduce the user's input.
pub fn build_finetuning_tensors(
    pairs: &[Pair],
    vocab: &Vocabulary,
    context_size: usize,
    mask_user_tokens: bool,
) -> Option<(Tensor, Tensor)> {
    assert!(context_size > 0, "context_size must be positive");
    let v = vocab.size();
    let row_len = context_size * v;
    // Fill the one-hot feature matrix directly into a single flat buffer instead
    // of collecting a `Vec<Vec<f32>>` of per-row windows and copying them all
    // into a flat tensor afterwards. The old two-stage approach held every row
    // as its own heap allocation *and* a second full-size flat copy at the end,
    // roughly doubling peak memory for what is the largest tensor in next-token
    // training (n_examples × context_size × vocab_size floats).
    let mut x_flat: Vec<f32> = Vec::new();
    let mut y_ids: Vec<u32> = Vec::new();

    for pair in pairs {
        let mut seq: Vec<u32> = Vec::new();
        seq.push(USER_ID);
        seq.extend(vocab.encode(&pair.input));
        seq.push(EOS_ID);
        seq.push(ASSISTANT_ID);
        let output_start = seq.len();
        seq.extend(vocab.encode(&pair.output));
        seq.push(EOS_ID);

        if seq.len() < 2 { continue; }

        for i in 0..(seq.len() - 1) {
            let target = seq[i + 1];
            if mask_user_tokens && (i + 1) < output_start { continue; }

            // Append a fresh zeroed window, then set the one-hot bits in place.
            // Context: last `context_size` tokens up to and including `i`,
            // left-padded with PAD if shorter.
            let base = x_flat.len();
            x_flat.resize(base + row_len, 0.0);
            let start = (i + 1).saturating_sub(context_size);
            let prefix_len = (i + 1) - start;
            let pad_len = context_size - prefix_len;
            for p in 0..pad_len {
                x_flat[base + p * v + PAD_ID as usize] = 1.0;
            }
            for (offset, &tok) in seq[start..=i].iter().enumerate() {
                let pos = pad_len + offset;
                x_flat[base + pos * v + tok as usize] = 1.0;
            }
            y_ids.push(target);
        }
    }

    if y_ids.is_empty() { return None; }

    let n = y_ids.len();
    let mut y_flat = vec![0.0_f32; n * v];
    for (i, &t) in y_ids.iter().enumerate() {
        y_flat[i * v + t as usize] = 1.0;
    }

    Some((
        Tensor::new(vec![n, row_len], x_flat),
        Tensor::new(vec![n, v], y_flat),
    ))
}

/// Encode a single context window for inference. Returns a `[1, context_size * vocab_size]`
/// tensor with the most recent `context_size` tokens (left-padded with PAD).
pub fn encode_context(ids: &[u32], vocab: &Vocabulary, context_size: usize) -> Tensor {
    let v = vocab.size();
    let mut window = vec![0.0_f32; context_size * v];
    let start = ids.len().saturating_sub(context_size);
    let prefix_len = ids.len() - start;
    let pad_len = context_size - prefix_len;
    for p in 0..pad_len {
        window[p * v + PAD_ID as usize] = 1.0;
    }
    for (offset, &tok) in ids[start..].iter().enumerate() {
        let pos = pad_len + offset;
        window[pos * v + tok as usize] = 1.0;
    }
    Tensor::new(vec![1, context_size * v], window)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{TokenizerMode, VocabularyOptions};

    fn char_vocab(text: &str) -> Vocabulary {
        Vocabulary::build(TokenizerMode::Char, &[text], &VocabularyOptions::default())
    }

    #[test]
    fn pretraining_tensors_have_correct_shape() {
        let vocab = char_vocab("abcabc");
        let v = vocab.size();
        let (x, y) = build_pretraining_tensors("abcabc", &vocab, 2).unwrap();
        // 6 tokens, context 2 → 4 examples
        assert_eq!(x.shape, vec![4, 2 * v]);
        assert_eq!(y.shape, vec![4, v]);
        let target_id = vocab.id_of("c") as usize;
        assert_eq!(y.data[target_id], 1.0);
    }

    #[test]
    fn finetuning_emits_examples() {
        let vocab = char_vocab("hi there");
        let v = vocab.size();
        let pairs = vec![Pair { input: "hi".into(), output: "there".into() }];
        let (x, _y) = build_finetuning_tensors(&pairs, &vocab, 4, false).unwrap();
        // Sequence: <user> h i <eos> <assistant> t h e r e <eos>
        //         =  1     2 3 4     5           6 7 8 9 10 11  = 11 tokens
        // 10 transitions (predicting each subsequent token).
        assert_eq!(x.cols(), 4 * v);
        assert_eq!(x.rows(), 10);
    }

    #[test]
    fn finetuning_masks_user_region() {
        // With mask_user_tokens=true, only transitions producing tokens in
        // the assistant region should be emitted. Output sequence is
        // `<user> h i <eos> <assistant> y <eos>` — 7 tokens; the assistant
        // region starts at index 5 (the 'y'). Targets in that region are
        // 'y' (transition 5→6) and '<eos>' (transition 6→7), so 2 rows.
        let vocab = char_vocab("hi y");
        let pairs = vec![Pair { input: "hi".into(), output: "y".into() }];
        let (x, _) = build_finetuning_tensors(&pairs, &vocab, 3, true).unwrap();
        assert_eq!(x.rows(), 2);
    }

    #[test]
    fn encode_context_left_pads() {
        let vocab = char_vocab("ab");
        let v = vocab.size();
        let ids = vocab.encode("a");
        let ctx = encode_context(&ids, &vocab, 3);
        assert_eq!(ctx.shape, vec![1, 3 * v]);
        assert_eq!(ctx.data[PAD_ID as usize], 1.0);
        assert_eq!(ctx.data[v + PAD_ID as usize], 1.0);
        assert_eq!(ctx.data[2 * v + vocab.id_of("a") as usize], 1.0);
    }
}
