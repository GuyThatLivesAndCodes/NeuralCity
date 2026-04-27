'use strict';

// Chat formatting for character-level language models.
//
// We accept several JSON shapes and flatten each conversation into a single
// text stream with role tags the model can learn:
//
//   <|user|>Hello, I need help<|end|><|assistant|>Of course, I'm here<|end|>
//
// During inference we wrap the user's prompt the same way and stop generation
// when we hit the assistant end tag.

const USER_OPEN = '<|user|>';
const ASSISTANT_OPEN = '<|assistant|>';
const SYSTEM_OPEN = '<|system|>';
const END = '<|end|>';

// Public so callers (tests / docs) can reference them.
const TAGS = { USER_OPEN, ASSISTANT_OPEN, SYSTEM_OPEN, END };

// Detect whether a sample is a chat-style record. Returns the normalized form
// or null if not a chat sample.
//   { user, assistant }                      → one turn
//   { messages: [{role, content}, ...] }     → multi-turn (OpenAI-style)
//   { conversation: [{user|assistant, ...}]} → multi-turn alternating
function normalizeChatSample(s) {
  if (!s || typeof s !== 'object') return null;
  if (typeof s.user === 'string' && typeof s.assistant === 'string') {
    const turns = [{ role: 'user', content: s.user }, { role: 'assistant', content: s.assistant }];
    if (typeof s.system === 'string') turns.unshift({ role: 'system', content: s.system });
    return turns;
  }
  if (Array.isArray(s.messages) && s.messages.every(m => m && typeof m.role === 'string' && typeof m.content === 'string')) {
    return s.messages.map(m => ({ role: m.role, content: m.content }));
  }
  if (Array.isArray(s.conversation)) {
    const out = [];
    for (const turn of s.conversation) {
      if (!turn || typeof turn !== 'object') continue;
      if (typeof turn.user === 'string') out.push({ role: 'user', content: turn.user });
      if (typeof turn.assistant === 'string') out.push({ role: 'assistant', content: turn.assistant });
      if (typeof turn.system === 'string') out.push({ role: 'system', content: turn.system });
    }
    return out.length ? out : null;
  }
  return null;
}

function renderTurns(turns) {
  let out = '';
  for (const t of turns) {
    const open = t.role === 'assistant' ? ASSISTANT_OPEN : t.role === 'system' ? SYSTEM_OPEN : USER_OPEN;
    out += open + t.content + END;
  }
  return out;
}

// Take training data (whatever shape) and return:
//   { text: string, isChat: boolean, perSampleCount }
// Falls back to the existing plaintext behavior for non-chat samples.
function buildCorpus(data) {
  if (!data) return { text: '', isChat: false, perSampleCount: 0 };
  // Direct text wins
  if (typeof data.text === 'string' && (!data.samples || data.samples.length === 0)) {
    return { text: data.text, isChat: false, perSampleCount: 0 };
  }
  if (!Array.isArray(data.samples)) {
    if (typeof data.text === 'string') return { text: data.text, isChat: false, perSampleCount: 0 };
    return { text: '', isChat: false, perSampleCount: 0 };
  }
  // Look at the first non-null sample to decide
  const chats = [];
  const plains = [];
  for (const s of data.samples) {
    const turns = normalizeChatSample(s);
    if (turns) chats.push(turns);
    else if (typeof s?.text === 'string') plains.push(s.text);
  }
  if (chats.length > 0) {
    // Render every conversation, separated by a blank line for breathing room.
    const text = chats.map(renderTurns).join('\n');
    return { text, isChat: true, perSampleCount: chats.length };
  }
  if (plains.length > 0) {
    return { text: plains.join('\n'), isChat: false, perSampleCount: plains.length };
  }
  return { text: '', isChat: false, perSampleCount: 0 };
}

// Wrap a single user prompt for inference against a chat-trained model.
// The returned string ends right after <|assistant|> so the model continues
// from there.
function wrapPromptForChat(userPrompt, system) {
  let out = '';
  if (typeof system === 'string' && system.length > 0) out += SYSTEM_OPEN + system + END;
  out += USER_OPEN + userPrompt + END + ASSISTANT_OPEN;
  return out;
}

// Wrap a multi-turn conversation. `history` is the past turns (any of
// {role:'user'|'assistant'|'system', content:string}). `opts.userPrompt` is
// the *new* user message we're asking about; if present it's appended after
// the history (caller normally passes both — the running history and the new
// turn the user just typed). `opts.system`, if non-empty, is anchored at the
// very start so the model always sees it regardless of truncation.
//
// Output ends with <|assistant|> so the model continues the assistant span.
function wrapHistoryForChat(history, opts) {
  opts = opts || {};
  const turns = [];
  if (Array.isArray(history)) {
    for (const t of history) {
      if (!t || typeof t.content !== 'string') continue;
      const role = t.role === 'assistant' ? 'assistant' : t.role === 'system' ? 'system' : 'user';
      turns.push({ role, content: t.content });
    }
  }
  if (typeof opts.userPrompt === 'string' && opts.userPrompt.length > 0) {
    turns.push({ role: 'user', content: opts.userPrompt });
  }
  // Hoist the most recent system message (if any) to the front for stable anchoring.
  // We dedupe by removing any system turns that were inline and re-inserting at index 0.
  let system = typeof opts.system === 'string' && opts.system.length > 0 ? opts.system : null;
  const cleaned = [];
  for (const t of turns) {
    if (t.role === 'system') {
      // Inline system messages override the outer system if no explicit one was given.
      if (!system) system = t.content;
      continue;
    }
    cleaned.push(t);
  }
  let out = '';
  if (system) out += SYSTEM_OPEN + system + END;
  for (const t of cleaned) {
    const open = t.role === 'assistant' ? ASSISTANT_OPEN : USER_OPEN;
    out += open + t.content + END;
  }
  out += ASSISTANT_OPEN;
  return out;
}

// Truncate a wrapped chat prompt so its encoded length fits within `maxLen`
// tokens, while preserving:
//   - the system anchor (if present, always kept)
//   - the trailing <|assistant|> suffix (so the model still continues)
//   - whole turns (we never split mid-turn — that would make the model see a
//     dangling open-tag with no end)
// Drops the *oldest* user/assistant turns first.
function truncateWrappedToFit(wrapped, encodeFn, maxLen) {
  if (encodeFn(wrapped).length <= maxLen) return wrapped;

  // Split into the system prefix (if any) + ordered list of user/assistant turns + trailing assistant-open.
  const sysIdx = wrapped.startsWith(SYSTEM_OPEN) ? wrapped.indexOf(END) + END.length : 0;
  const sysPrefix = wrapped.slice(0, sysIdx);
  // The last token is the trailing ASSISTANT_OPEN (no content after).
  const trailingIdx = wrapped.lastIndexOf(ASSISTANT_OPEN);
  // Body between sysPrefix and trailingAssistantOpen.
  const body = wrapped.slice(sysIdx, trailingIdx);
  const trailing = wrapped.slice(trailingIdx); // always ASSISTANT_OPEN

  // Walk body, splitting into [openTag + content + END] chunks.
  const turns = [];
  let i = 0;
  while (i < body.length) {
    let open;
    if (body.startsWith(USER_OPEN, i)) open = USER_OPEN;
    else if (body.startsWith(ASSISTANT_OPEN, i)) open = ASSISTANT_OPEN;
    else if (body.startsWith(SYSTEM_OPEN, i)) open = SYSTEM_OPEN;
    else { i++; continue; }
    const endAt = body.indexOf(END, i + open.length);
    if (endAt === -1) break;
    turns.push(body.slice(i, endAt + END.length));
    i = endAt + END.length;
  }

  // Drop oldest turns until it fits. Always keep the most recent user turn
  // (the one we're asking about) — drop from the front.
  let kept = turns.slice();
  while (kept.length > 1) {
    const candidate = sysPrefix + kept.join('') + trailing;
    if (encodeFn(candidate).length <= maxLen) return candidate;
    kept.shift();
  }
  // Even with one turn it may still not fit. Return what we have — the
  // caller's existing window-padding logic will handle the final shape.
  return sysPrefix + kept.join('') + trailing;
}

// Strip role tags + everything after the first END tag in the assistant span.
// Used when post-processing generated text.
//
// Small charLMs frequently emit *corrupted* end-of-turn markers: instead of a
// clean '<|end|>' they'll produce '<aend|>', '<|en|>', '<|edn|>', etc. If we
// only cut on a perfect '<|end|>' we leak garbage and even the start of the
// next imagined conversation into the visible reply. So we also cut on the
// '<|' bigram, which is the unmistakable signature of *any* attempt at a
// role tag — whether well-formed or not.
function extractAssistantReply(generated) {
  // Generated is whatever came after wrapPromptForChat — i.e. assistant content
  // possibly followed by <|end|> and more turns.
  const cutPoints = [];
  const endIdx = generated.indexOf(END);
  if (endIdx !== -1) cutPoints.push(endIdx);
  const tagStartIdx = generated.indexOf('<|');
  if (tagStartIdx !== -1) cutPoints.push(tagStartIdx);
  // Also cut on the tail bigram '|>' — a small charLM can emit the closing half
  // of a role tag (e.g. "end|>", "user|>") without the leading "<|", bypassing
  // the tagStartIdx check above.
  const tagTailIdx = generated.indexOf('|>');
  if (tagTailIdx !== -1) cutPoints.push(tagTailIdx);
  if (cutPoints.length === 0) return generated;
  return generated.slice(0, Math.min(...cutPoints));
}

module.exports = { TAGS, normalizeChatSample, renderTurns, buildCorpus, wrapPromptForChat, wrapHistoryForChat, truncateWrappedToFit, extractAssistantReply };
