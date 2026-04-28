'use strict';

const http = require('http');
const crypto = require('crypto');
const os = require('os');
const { EventEmitter } = require('events');
const { infer } = require('../engine/trainer');

// Lazy-loaded plugin engines (only required when a plugin network is served).
let _T = null;
let _restoreFromState = null;
function getTensor() { return _T || (_T = require('../engine/tensor')); }
function getRestore() { return _restoreFromState || (_restoreFromState = require('../engine/model').restoreFromState); }

function _argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

// Run inference against a plugin network's saved state.
// Neuroevolution plugins save Population.toJSON() → individuals[0] is the elite.
// DQN plugins save DQNAgent.toJSON() → onlineState is the policy network.
function pluginPredict(net, payload) {
  const arch  = net.architecture;
  const state = net.state;
  const T     = getTensor();
  const restore = getRestore();

  const rawInputs = payload.inputs ?? payload.input;
  if (!rawInputs || !Array.isArray(rawInputs)) {
    throw new Error(`"inputs" is required — provide an array of ${arch.inputDim} numbers`);
  }

  let modelState, modelArch;
  if (Array.isArray(state.individuals)) {
    // Neuroevolution (self-driving-car, snake-neuro): elite is at index 0.
    modelState = state.individuals[0];
    modelArch  = state.arch || arch;
  } else if (state.onlineState) {
    // DQN (warehouse-robot): use the online policy network.
    modelState = state.onlineState;
    modelArch  = state.arch || arch;
  } else {
    throw new Error('Unrecognised plugin state format — ensure you have saved a trained model.');
  }

  if (rawInputs.length !== modelArch.inputDim) {
    throw new Error(`Expected ${modelArch.inputDim} inputs, got ${rawInputs.length}`);
  }

  const rng   = T.rngFromSeed(42);
  const model = restore(modelState, modelArch, rng);
  const x     = new T.Tensor([1, modelArch.inputDim], new Float32Array(rawInputs), false);
  const out   = model.forward(x, { training: false });
  const outputs = Array.from(out.data);

  const pluginKind = arch.pluginKind;
  if (pluginKind === 'self-driving-car') {
    return {
      outputs,
      steer:    Math.max(-1, Math.min(1, outputs[0])),
      throttle: Math.max(-1, Math.min(1, outputs[1])),
    };
  }
  if (pluginKind === 'snake-neuro') {
    const DIRS = ['up', 'right', 'down', 'left'];
    const dir  = _argmax(outputs);
    return { outputs, direction: dir, directionName: DIRS[dir] };
  }
  if (pluginKind === 'warehouse-robot') {
    const ACTIONS = ['up', 'down', 'left', 'right'];
    const action  = _argmax(outputs);
    return { outputs, action, actionName: ACTIONS[action] };
  }
  return { outputs };
}

function getHostIp() {
  const ifaces = os.networkInterfaces();
  for (const name of Object.keys(ifaces)) {
    for (const info of ifaces[name]) {
      if (info.family === 'IPv4' && !info.internal) return info.address;
    }
  }
  return '127.0.0.1';
}

const MAX_SESSION_TURNS = 64;
const MAX_SESSIONS_PER_MODEL = 256;
const SESSION_TTL_MS = 60 * 60 * 1000;

// ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter {
  // window: sliding window in ms; maxReqs: max requests per window
  constructor(maxReqs = 60, windowMs = 60_000) {
    this.maxReqs = maxReqs;
    this.windowMs = windowMs;
    this.buckets = new Map(); // ip → [timestamp, ...]
  }

  check(ip) {
    const now = Date.now();
    let timestamps = this.buckets.get(ip) || [];
    timestamps = timestamps.filter(t => now - t < this.windowMs);
    if (timestamps.length >= this.maxReqs) return false;
    timestamps.push(now);
    this.buckets.set(ip, timestamps);
    return true;
  }

  // Evict stale entries to prevent unbounded memory growth
  gc() {
    const now = Date.now();
    for (const [ip, ts] of this.buckets) {
      const fresh = ts.filter(t => now - t < this.windowMs);
      if (fresh.length === 0) this.buckets.delete(ip);
      else this.buckets.set(ip, fresh);
    }
  }
}

// ── Per-server metrics ────────────────────────────────────────────────────────

class ServerMetrics {
  constructor() {
    this.requestsTotal = 0;
    this.requestsByEndpoint = {};
    this.errorsTotal = 0;
    this.latencySumMs = 0;
    this.latencyCount = 0;
    this.startedAt = Date.now();
  }

  record(endpoint, durationMs, isError) {
    this.requestsTotal++;
    this.requestsByEndpoint[endpoint] = (this.requestsByEndpoint[endpoint] || 0) + 1;
    if (isError) this.errorsTotal++;
    this.latencySumMs += durationMs;
    this.latencyCount++;
  }

  toPrometheus(id) {
    const safe = id.replace(/[^a-zA-Z0-9_]/g, '_');
    const uptime = (Date.now() - this.startedAt) / 1000;
    const avgLatency = this.latencyCount > 0 ? this.latencySumMs / this.latencyCount : 0;
    const lines = [
      `# HELP neuralcabin_requests_total Total HTTP requests`,
      `# TYPE neuralcabin_requests_total counter`,
      `neuralcabin_requests_total{model="${safe}"} ${this.requestsTotal}`,
      `# HELP neuralcabin_errors_total Total HTTP errors`,
      `# TYPE neuralcabin_errors_total counter`,
      `neuralcabin_errors_total{model="${safe}"} ${this.errorsTotal}`,
      `# HELP neuralcabin_avg_latency_ms Average request latency`,
      `# TYPE neuralcabin_avg_latency_ms gauge`,
      `neuralcabin_avg_latency_ms{model="${safe}"} ${avgLatency.toFixed(2)}`,
      `# HELP neuralcabin_uptime_seconds Server uptime in seconds`,
      `# TYPE neuralcabin_uptime_seconds gauge`,
      `neuralcabin_uptime_seconds{model="${safe}"} ${uptime.toFixed(1)}`,
    ];
    for (const [ep, count] of Object.entries(this.requestsByEndpoint)) {
      lines.push(`neuralcabin_requests_by_endpoint{model="${safe}",endpoint="${ep}"} ${count}`);
    }
    return lines.join('\n') + '\n';
  }
}

// ── JWT-lite: HS256 API-key tokens ───────────────────────────────────────────
// Not full JWT — just a HMAC-SHA256 signed { sub, iat, exp } payload.
// Keeps the server dependency-free while still being verifiable.

function signToken(payload, secret) {
  const header = Buffer.from(JSON.stringify({ alg: 'HS256', typ: 'JWT' })).toString('base64url');
  const body = Buffer.from(JSON.stringify(payload)).toString('base64url');
  const sig = crypto
    .createHmac('sha256', secret)
    .update(`${header}.${body}`)
    .digest('base64url');
  return `${header}.${body}.${sig}`;
}

function verifyToken(token, secret) {
  if (!token || typeof token !== 'string') return null;
  const parts = token.split('.');
  if (parts.length !== 3) return null;
  const [header, body, sig] = parts;
  const expected = crypto.createHmac('sha256', secret).update(`${header}.${body}`).digest('base64url');
  if (!crypto.timingSafeEqual(Buffer.from(sig), Buffer.from(expected))) return null;
  try {
    const payload = JSON.parse(Buffer.from(body, 'base64url').toString('utf8'));
    if (payload.exp && Date.now() / 1000 > payload.exp) return null;
    return payload;
  } catch (_) {
    return null;
  }
}

// ── ApiServer ─────────────────────────────────────────────────────────────────

class ApiServer extends EventEmitter {
  constructor(storage) {
    super();
    this.storage = storage;
    this.servers = new Map(); // id → { server, port, startedAt, sessions, metrics, limiter, authSecret }
  }

  _getOrCreateSession(rec, sessionId) {
    if (!sessionId) sessionId = 'session-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);
    let s = rec.sessions.get(sessionId);
    if (!s) {
      if (rec.sessions.size >= MAX_SESSIONS_PER_MODEL) {
        let oldestKey = null, oldestAt = Infinity;
        for (const [k, v] of rec.sessions) if (v.lastSeen < oldestAt) { oldestAt = v.lastSeen; oldestKey = k; }
        if (oldestKey) rec.sessions.delete(oldestKey);
      }
      s = { history: [], system: '', createdAt: Date.now(), lastSeen: Date.now() };
      rec.sessions.set(sessionId, s);
    }
    s.lastSeen = Date.now();
    return { sessionId, session: s };
  }

  _gcSessions(rec) {
    const now = Date.now();
    for (const [k, v] of rec.sessions) if (now - v.lastSeen > SESSION_TTL_MS) rec.sessions.delete(k);
  }

  listAll() {
    const out = [];
    for (const [id, rec] of this.servers) {
      out.push({
        id, port: rec.port, startedAt: rec.startedAt,
        url: `http://${getHostIp()}:${rec.port}`,
        auth: !!rec.authSecret
      });
    }
    return out;
  }

  status(id) {
    const rec = this.servers.get(id);
    if (!rec) return { running: false };
    return { running: true, port: rec.port, url: `http://${getHostIp()}:${rec.port}` };
  }

  // opts = { port?, rateLimit?: number, authSecret?: string }
  start(id, port, opts = {}) {
    if (this.servers.has(id)) {
      const rec = this.servers.get(id);
      return { running: true, port: rec.port, url: `http://${getHostIp()}:${rec.port}` };
    }
    const net = this.storage.getNetwork(id);
    if (!net) throw new Error('Network not found');
    if (!net.state || net.stateLocked) throw new Error('Network has no trained state');
    const usePort = Number(port) || 0;

    const sessions = new Map();
    const metrics = new ServerMetrics();
    const limiter = new RateLimiter(opts.rateLimit ?? 120, 60_000);
    const authSecret = opts.authSecret || null;
    const recRefHolder = { sessions };

    // GC rate limiter buckets every 5 minutes.
    const gcTimer = setInterval(() => limiter.gc(), 5 * 60_000);
    gcTimer.unref && gcTimer.unref();

    const readBody = (req) => new Promise((resolve, reject) => {
      let body = '';
      req.on('data', c => { body += c; if (body.length > 1_000_000) reject(new Error('Request too large')); });
      req.on('end', () => resolve(body));
      req.on('error', reject);
    });

    const checkAuth = (req) => {
      if (!authSecret) return true;
      const header = req.headers['authorization'] || '';
      const token = header.startsWith('Bearer ') ? header.slice(7) : null;
      if (!token) return false;
      return verifyToken(token, authSecret) !== null;
    };

    const getClientIp = (req) => {
      return (req.headers['x-forwarded-for'] || req.socket.remoteAddress || '').split(',')[0].trim();
    };

    const server = http.createServer(async (req, res) => {
      const t0 = Date.now();
      const url = req.url || '/';
      let statusCode = 200;

      // Common headers
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('X-Content-Type-Options', 'nosniff');

      // Rate limiting (skip for /health and /metrics)
      if (url !== '/health' && url !== '/metrics') {
        const ip = getClientIp(req);
        if (!limiter.check(ip)) {
          res.statusCode = 429;
          res.end(JSON.stringify({ error: 'Too many requests — try again later.' }));
          metrics.record(url, Date.now() - t0, true);
          this.emit('log', { id, line: `429 rate-limited ${ip}` });
          return;
        }
      }

      // Auth check (skip for /health)
      if (url !== '/health' && !checkAuth(req)) {
        res.statusCode = 401;
        res.end(JSON.stringify({ error: 'Unauthorized — provide a valid Bearer token.' }));
        metrics.record(url, Date.now() - t0, true);
        return;
      }

      try {
        if (req.method === 'GET' && (url === '/' || url === '/info')) {
          const freshNet = this.storage.getNetwork(id);
          res.end(JSON.stringify({
            name: freshNet.name, id: freshNet.id,
            kind: freshNet.architecture.kind,
            description: freshNet.description,
            inputSpec: this._inputSpec(freshNet),
            chat: !!freshNet.architecture.isChat,
            auth: !!authSecret,
          }));
          this.emit('log', { id, line: `GET ${url}` });
          metrics.record(url, Date.now() - t0, false);
          return;
        }

        if (req.method === 'GET' && url === '/health') {
          res.end(JSON.stringify({ ok: true, uptime: process.uptime() }));
          return;
        }

        if (req.method === 'GET' && url === '/metrics') {
          res.setHeader('Content-Type', 'text/plain; version=0.0.4');
          res.end(metrics.toPrometheus(id));
          return;
        }

        if (req.method === 'POST' && url === '/predict') {
          const body = await readBody(req);
          const payload = body ? JSON.parse(body) : {};
          const freshNet = this.storage.getNetwork(id);
          if (!freshNet || !freshNet.state) throw new Error('Model no longer available');
          const result = freshNet.architecture && freshNet.architecture.pluginKind
            ? pluginPredict(freshNet, payload)
            : infer(freshNet, payload);
          res.end(JSON.stringify(result));
          this.emit('log', { id, line: `POST /predict ok` });
          metrics.record('/predict', Date.now() - t0, false);
          return;
        }

        if (req.method === 'POST' && url === '/chat') {
          const body = await readBody(req);
          const payload = body ? JSON.parse(body) : {};
          const freshNet = this.storage.getNetwork(id);
          if (!freshNet || !freshNet.state) throw new Error('Model no longer available');
          if (!freshNet.architecture.isChat) throw new Error('Model is not a chat model — train on chat samples or use /predict');
          this._gcSessions(recRefHolder);
          const { sessionId, session } = this._getOrCreateSession(recRefHolder, payload.sessionId);
          const message = String(payload.message ?? payload.prompt ?? '');
          if (!message) throw new Error('"message" is required');
          if (typeof payload.system === 'string') session.system = payload.system;
          const result = infer(freshNet, {
            history: session.history, prompt: message, system: session.system,
            maxTokens: payload.maxTokens, temperature: payload.temperature, topK: payload.topK
          });
          session.history.push({ role: 'user', content: message });
          session.history.push({ role: 'assistant', content: result.text });
          while (session.history.length > MAX_SESSION_TURNS) session.history.shift();
          res.end(JSON.stringify({ sessionId, reply: result.text, history: session.history.slice() }));
          this.emit('log', { id, line: `POST /chat ok (session=${sessionId.slice(0, 14)}…, turns=${session.history.length})` });
          metrics.record('/chat', Date.now() - t0, false);
          return;
        }

        if (req.method === 'POST' && url === '/chat/reset') {
          const body = await readBody(req);
          const payload = body ? JSON.parse(body) : {};
          if (payload.sessionId && recRefHolder.sessions.has(payload.sessionId)) {
            recRefHolder.sessions.delete(payload.sessionId);
            res.end(JSON.stringify({ ok: true, cleared: payload.sessionId }));
          } else {
            res.end(JSON.stringify({ ok: true, cleared: null }));
          }
          this.emit('log', { id, line: `POST /chat/reset` });
          metrics.record('/chat/reset', Date.now() - t0, false);
          return;
        }

        res.statusCode = 404;
        statusCode = 404;
        res.end(JSON.stringify({ error: 'Not found' }));

      } catch (e) {
        res.statusCode = 400;
        statusCode = 400;
        res.end(JSON.stringify({ error: e.message }));
        metrics.record(url, Date.now() - t0, true);
        this.emit('log', { id, line: `${req.method} ${url} ERROR: ${e.message}` });
        return;
      }

      if (statusCode >= 400) metrics.record(url, Date.now() - t0, true);
    });

    server.on('error', (e) => {
      this.emit('log', { id, line: `server error: ${e.message}` });
      this.servers.delete(id);
    });

    return new Promise((resolve, reject) => {
      server.listen(usePort, () => {
        const actualPort = server.address().port;
        this.servers.set(id, { server, port: actualPort, startedAt: Date.now(), sessions, metrics, limiter, authSecret, gcTimer });
        this.emit('log', { id, line: `listening on ${actualPort}${authSecret ? ' (auth enabled)' : ''}` });
        resolve({ running: true, port: actualPort, url: `http://${getHostIp()}:${actualPort}` });
      });
      server.once('error', reject);
    });
  }

  // Generate a signed API token for a given server (requires authSecret to have been set at start()).
  issueToken(id, expiresInSeconds = 86400) {
    const rec = this.servers.get(id);
    if (!rec || !rec.authSecret) throw new Error('Server not running or no authSecret configured');
    const now = Math.floor(Date.now() / 1000);
    return signToken({ sub: id, iat: now, exp: now + expiresInSeconds }, rec.authSecret);
  }

  stop(id) {
    const rec = this.servers.get(id);
    if (!rec) return { running: false };
    rec.server.close();
    if (rec.gcTimer) clearInterval(rec.gcTimer);
    this.servers.delete(id);
    this.emit('log', { id, line: 'stopped' });
    return { running: false };
  }

  stopAll() {
    for (const [id, rec] of this.servers) {
      try { rec.server.close(); } catch (_) {}
      if (rec.gcTimer) clearInterval(rec.gcTimer);
    }
    this.servers.clear();
  }

  getMetrics(id) {
    const rec = this.servers.get(id);
    if (!rec) return null;
    return rec.metrics.toPrometheus(id);
  }

  _inputSpec(net) {
    const a = net.architecture;
    if (a.pluginKind === 'self-driving-car') {
      return {
        type: 'plugin', pluginKind: 'self-driving-car',
        field: 'inputs', length: 11,
        inputDescription: [
          'ray[0..8]: 9 sensor distances (0=wall, 1=clear), angles −80° to +80°',
          'inputs[9]: normalised speed (0–1)',
          'inputs[10]: normalised heading angle (0–1)',
        ],
        outputFields: { steer: '−1 to 1', throttle: '−1 to 1', outputs: 'raw logits [2]' },
      };
    }
    if (a.pluginKind === 'snake-neuro') {
      return {
        type: 'plugin', pluginKind: 'snake-neuro',
        field: 'inputs', length: 255,
        inputDescription: [
          '15×17 grid (row-major): 0=empty, 1=apple, 0.5=head, 0.25=body, −0.5=tail',
        ],
        outputFields: { direction: '0–3 index', directionName: 'up|right|down|left', outputs: 'raw logits [4]' },
      };
    }
    if (a.pluginKind === 'warehouse-robot') {
      const nBoxes = net.state && net.state.arch
        ? Math.round((net.state.arch.inputDim - 5) / 6)
        : Math.round((a.inputDim - 5) / 6);
      return {
        type: 'plugin', pluginKind: 'warehouse-robot',
        field: 'inputs', length: a.inputDim,
        nBoxes,
        inputDescription: [
          'inputs[0..1]: robot [row, col] normalised 0–1',
          'inputs[2]: carrying flag (0 or 1)',
          'inputs[3..3+nBoxes*2−1]: box positions [row, col] × nBoxes',
          'inputs[next nBoxes*2]: target positions [row, col] × nBoxes',
          'inputs[next nBoxes*2]: box→target offsets × nBoxes',
          'inputs[last 2]: robot→nearest goal offset [dr, dc]',
        ],
        outputFields: { action: '0–3 index', actionName: 'up|down|left|right', outputs: 'raw Q-values [4]' },
      };
    }
    if (a.kind === 'classifier' || a.kind === 'mlp') return { type: 'vector', length: a.inputDim };
    if (a.kind === 'regressor') return { type: 'vector', length: a.inputDim, outputLength: a.outputDim };
    if (a.kind === 'charLM' || a.kind === 'gpt') {
      const base = { type: 'text', fields: ['prompt', 'maxTokens', 'temperature', 'topK'] };
      if (a.isChat) {
        base.chatFields = ['history', 'messages', 'system'];
        base.chatEndpoints = { stateful: '/chat', resetSession: '/chat/reset' };
      }
      return base;
    }
    return { type: 'unknown' };
  }
}

module.exports = { ApiServer };
