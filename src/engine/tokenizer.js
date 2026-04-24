'use strict';

// Internal sentinel used only as merge-key separator — not stored in tokens.
const SEP = '\x01';

// ─────────────────────────────────────────────────────────────────────────────
// CharTokenizer — one token per Unicode character (original behaviour)
// ─────────────────────────────────────────────────────────────────────────────
class CharTokenizer {
  constructor(chars) {
    this.chars = chars || [];
    this.stoi = new Map();
    this.itos = new Map();
    for (let i = 0; i < this.chars.length; i++) {
      this.stoi.set(this.chars[i], i);
      this.itos.set(i, this.chars[i]);
    }
  }
  get vocabSize() { return this.chars.length; }
  get kind() { return 'char'; }

  static fromCorpus(text) {
    const chars = Array.from(new Set(text)).sort();
    return new CharTokenizer(chars);
  }

  encode(text) {
    const out = new Array(text.length);
    for (let i = 0; i < text.length; i++) {
      const id = this.stoi.get(text[i]);
      if (id == null) throw new Error(`char not in vocab: ${JSON.stringify(text[i])}`);
      out[i] = id;
    }
    return out;
  }

  encodeSafe(text, fallback = null) {
    const out = [];
    for (let i = 0; i < text.length; i++) {
      const id = this.stoi.get(text[i]);
      if (id == null) { if (fallback != null) out.push(fallback); }
      else out.push(id);
    }
    return out;
  }

  decode(ids) {
    let s = '';
    for (const id of ids) s += this.itos.get(id) || '';
    return s;
  }

  toJSON() { return { kind: 'char', chars: this.chars }; }
  static fromJSON(o) { return new CharTokenizer(o.chars); }
}

// ─────────────────────────────────────────────────────────────────────────────
// WordPartTokenizer — BPE subword tokenization (Sennrich et al., 2016)
//
// Starts from a character vocabulary and iteratively merges the most frequent
// adjacent pair until targetVocabSize is reached. The default of 512 tokens
// gives a good balance between context efficiency and coverage for small
// corpora. Encoding is greedy: apply merges in priority order (lowest merge
// index = highest priority) until no more apply.
// ─────────────────────────────────────────────────────────────────────────────
class WordPartTokenizer {
  constructor(merges, vocab) {
    // merges: Array<[string, string]> — ordered list of BPE merge rules
    // vocab:  plain object  token → id  (not a Map, for easy JSON round-trip)
    this.merges = merges;
    this.vocab  = vocab;
    this._vocabSize = Object.keys(vocab).length;
    this.ivocab = Object.create(null);
    for (const [tok, id] of Object.entries(vocab)) this.ivocab[id] = tok;
  }

  get vocabSize() { return this._vocabSize; }
  get kind() { return 'wordpart'; }

  // Map-like interface so trainer.js pad-char logic works the same as for
  // CharTokenizer (which uses a real Map). Only `has` and `get` are needed.
  get stoi() {
    const v = this.vocab;
    return { has: (k) => Object.prototype.hasOwnProperty.call(v, k), get: (k) => v[k] };
  }

  // Apply BPE merge rules in priority order (rule 0 = highest priority).
  // Each rule is applied in one O(n) pass over the token sequence — this is
  // O(n × V) overall, not O(n²) — critical for large-corpus encoding.
  // Produces the same result as greedy best-first because BPE rule ordering
  // guarantees that rule i is always preferred over rule i+1.
  _applyMerges(chars) {
    let tokens = chars.slice();
    for (const [a, b] of this.merges) {
      if (tokens.length < 2) break;
      const next = [];
      let i = 0;
      while (i < tokens.length) {
        if (i + 1 < tokens.length && tokens[i] === a && tokens[i + 1] === b) {
          next.push(a + b); i += 2;
        } else {
          next.push(tokens[i]); i++;
        }
      }
      tokens = next;
    }
    return tokens;
  }

  // Async version: yields to the event loop every 32 merge rules so that IPC
  // and UI events are not starved during large-corpus encoding.
  async encodeAsync(text, yieldFn) {
    let tokens = Array.from(text);
    for (let ri = 0; ri < this.merges.length; ri++) {
      if (tokens.length < 2) break;
      const [a, b] = this.merges[ri];
      const next = [];
      let i = 0;
      while (i < tokens.length) {
        if (i + 1 < tokens.length && tokens[i] === a && tokens[i + 1] === b) {
          next.push(a + b); i += 2;
        } else {
          next.push(tokens[i]); i++;
        }
      }
      tokens = next;
      if (ri % 32 === 31 && yieldFn) await yieldFn();
    }
    return tokens.map(t => {
      const id = this.vocab[t];
      if (id == null) throw new Error(`token not in vocab: ${JSON.stringify(t)}`);
      return id;
    });
  }

  encode(text) {
    const tokens = this._applyMerges(Array.from(text));
    return tokens.map(t => {
      const id = this.vocab[t];
      if (id == null) throw new Error(`token not in vocab: ${JSON.stringify(t)}`);
      return id;
    });
  }

  encodeSafe(text, fallback = null) {
    const tokens = this._applyMerges(Array.from(text));
    const out = [];
    for (const t of tokens) {
      const id = this.vocab[t];
      if (id == null) { if (fallback != null) out.push(fallback); }
      else out.push(id);
    }
    return out;
  }

  decode(ids) { return ids.map(id => this.ivocab[id] || '').join(''); }

  static fromCorpus(text, targetVocabSize = 512) {
    // Drain the generator synchronously.
    const gen = WordPartTokenizer._buildBPEGen(text, targetVocabSize);
    let step;
    do { step = gen.next(); } while (!step.done);
    return step.value;
  }

  // Async version: yields to the event loop every 20 merge iterations so IPC
  // and UI events are not starved when building from a large corpus.
  static async fromCorpusAsync(text, targetVocabSize = 512, yieldFn) {
    const gen = WordPartTokenizer._buildBPEGen(text, targetVocabSize);
    let step;
    do {
      step = gen.next();
      if (!step.done && yieldFn) await yieldFn();
    } while (!step.done);
    return step.value;
  }

  // Generator that performs BPE training and yields every 20 merge steps.
  // The return value (when done === true) is the completed WordPartTokenizer.
  static *_buildBPEGen(text, targetVocabSize = 512) {
    const chars = Array.from(new Set(Array.from(text))).sort();
    const vocab = Object.create(null);
    let nextId = 0;
    for (const ch of chars) vocab[ch] = nextId++;

    const merges = [];
    let tokenSeq = Array.from(text);
    let vocabCount = chars.length;
    let itr = 0;

    while (vocabCount < targetVocabSize) {
      if (tokenSeq.length < 2) break;
      const counts = new Map();
      for (let i = 0; i < tokenSeq.length - 1; i++) {
        const k = tokenSeq[i] + SEP + tokenSeq[i + 1];
        counts.set(k, (counts.get(k) || 0) + 1);
      }
      if (!counts.size) break;
      let bestKey = null, bestCount = 0;
      for (const [k, c] of counts) { if (c > bestCount) { bestCount = c; bestKey = k; } }
      if (!bestKey || bestCount < 2) break;

      const si = bestKey.indexOf(SEP);
      const a = bestKey.slice(0, si), b = bestKey.slice(si + 1);
      const merged = a + b;
      vocab[merged] = nextId++;
      vocabCount++;
      merges.push([a, b]);

      const newSeq = [];
      let i = 0;
      while (i < tokenSeq.length) {
        if (i < tokenSeq.length - 1 && tokenSeq[i] === a && tokenSeq[i + 1] === b) {
          newSeq.push(merged); i += 2;
        } else { newSeq.push(tokenSeq[i]); i++; }
      }
      tokenSeq = newSeq;
      if (++itr % 20 === 0) yield;
    }

    return new WordPartTokenizer(merges, vocab);
  }

  toJSON() { return { kind: 'wordpart', merges: this.merges, vocab: this.vocab }; }
  static fromJSON(o) { return new WordPartTokenizer(o.merges, o.vocab); }
}

// ─────────────────────────────────────────────────────────────────────────────
// WordTokenizer — whitespace-split word-level tokenization
//
// Splits on /\S+|\s+/ so whitespace is preserved as tokens (allowing the model
// to learn spacing). Each distinct word/whitespace-run becomes one token.
// Context lengths are measured in words rather than characters.
// ─────────────────────────────────────────────────────────────────────────────
class WordTokenizer {
  constructor(words) {
    this.words = words;
    this.stoi  = new Map();
    this.itos  = new Map();
    for (let i = 0; i < words.length; i++) { this.stoi.set(words[i], i); this.itos.set(i, words[i]); }
  }

  get vocabSize() { return this.words.length; }
  get kind() { return 'word'; }

  _split(text) { return text.match(/\S+|\s+/g) || []; }

  static fromCorpus(text) {
    const tokens = text.match(/\S+|\s+/g) || [];
    const unique = Array.from(new Set(tokens)).sort();
    return new WordTokenizer(unique);
  }

  encode(text) {
    return this._split(text).map(t => {
      const id = this.stoi.get(t);
      if (id == null) throw new Error(`word not in vocab: ${JSON.stringify(t)}`);
      return id;
    });
  }

  encodeSafe(text, fallback = null) {
    const out = [];
    for (const t of this._split(text)) {
      const id = this.stoi.get(t);
      if (id == null) { if (fallback != null) out.push(fallback); }
      else out.push(id);
    }
    return out;
  }

  decode(ids) { return ids.map(id => this.itos.get(id) || '').join(''); }

  toJSON() { return { kind: 'word', words: this.words }; }
  static fromJSON(o) { return new WordTokenizer(o.words); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory helpers
// ─────────────────────────────────────────────────────────────────────────────

// Build a fresh tokenizer from corpus text.
// kind: 'char' | 'wordpart' | 'word'
// opts.vocabSize: target for BPE (wordpart only, default 512)
function buildTokenizer(text, kind, opts = {}) {
  switch (kind) {
    case 'wordpart': return WordPartTokenizer.fromCorpus(text, opts.vocabSize || 512);
    case 'word':     return WordTokenizer.fromCorpus(text);
    default:         return CharTokenizer.fromCorpus(text);
  }
}

// Restore a tokenizer from its toJSON() output.
// Handles old saves that have no `kind` field (treat as char).
function tokenizerFromJSON(o) {
  if (!o) return null;
  switch (o.kind) {
    case 'wordpart': return WordPartTokenizer.fromJSON(o);
    case 'word':     return WordTokenizer.fromJSON(o);
    default:         return CharTokenizer.fromJSON(o);
  }
}

module.exports = { CharTokenizer, WordPartTokenizer, WordTokenizer, buildTokenizer, tokenizerFromJSON };
