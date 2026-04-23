'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// JSON file store at <userData>/NeuralCity/networks/<id>.json
// Optional per-network encryption using AES-256-GCM with a passphrase-derived key.

function uuid() {
  return crypto.randomUUID();
}

function safeName(name) {
  return name.replace(/[^a-zA-Z0-9_\-]/g, '_').slice(0, 64);
}

class Storage {
  constructor(rootDir) {
    this.rootDir = rootDir;
    this.netDir = path.join(rootDir, 'networks');
    if (!fs.existsSync(this.netDir)) fs.mkdirSync(this.netDir, { recursive: true });
  }

  _netPath(id) { return path.join(this.netDir, `${id}.json`); }

  _readRaw(id) {
    const p = this._netPath(id);
    if (!fs.existsSync(p)) return null;
    return JSON.parse(fs.readFileSync(p, 'utf-8'));
  }

  _writeRaw(id, obj) {
    fs.writeFileSync(this._netPath(id), JSON.stringify(obj, null, 2));
  }

  // Encryption helpers
  _encryptPayload(payload, passphrase) {
    const salt = crypto.randomBytes(16);
    const key = crypto.scryptSync(passphrase, salt, 32);
    const iv = crypto.randomBytes(12);
    const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
    const plain = Buffer.from(JSON.stringify(payload), 'utf-8');
    const ct = Buffer.concat([cipher.update(plain), cipher.final()]);
    const tag = cipher.getAuthTag();
    return {
      alg: 'aes-256-gcm',
      salt: salt.toString('base64'),
      iv: iv.toString('base64'),
      tag: tag.toString('base64'),
      ct: ct.toString('base64')
    };
  }

  _decryptPayload(bundle, passphrase) {
    const salt = Buffer.from(bundle.salt, 'base64');
    const iv = Buffer.from(bundle.iv, 'base64');
    const tag = Buffer.from(bundle.tag, 'base64');
    const ct = Buffer.from(bundle.ct, 'base64');
    const key = crypto.scryptSync(passphrase, salt, 32);
    const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv);
    decipher.setAuthTag(tag);
    const pt = Buffer.concat([decipher.update(ct), decipher.final()]);
    return JSON.parse(pt.toString('utf-8'));
  }

  // list returns bare metadata (no weights) for UI listing
  listNetworks() {
    const files = fs.readdirSync(this.netDir).filter(f => f.endsWith('.json'));
    const out = [];
    for (const f of files) {
      try {
        const obj = JSON.parse(fs.readFileSync(path.join(this.netDir, f), 'utf-8'));
        out.push({
          id: obj.id,
          name: obj.name,
          kind: obj.architecture?.kind,
          createdAt: obj.createdAt,
          updatedAt: obj.updatedAt,
          encrypted: !!obj.encrypted,
          trained: !!obj.state,
          metricsTail: (obj.metrics || []).slice(-1)[0] || null
        });
      } catch (e) {
        // ignore broken
      }
    }
    out.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
    return out;
  }

  getNetwork(id, passphrase) {
    const obj = this._readRaw(id);
    if (!obj) return null;
    if (obj.encrypted && obj.encryptedState) {
      if (!passphrase) {
        // return without state — UI can prompt for decryption if needed
        const { encryptedState, ...rest } = obj;
        return { ...rest, stateLocked: true };
      }
      try {
        const payload = this._decryptPayload(obj.encryptedState, passphrase);
        return { ...obj, state: payload.state, tokenizer: payload.tokenizer };
      } catch (e) {
        return { ...obj, stateLocked: true, decryptError: e.message };
      }
    }
    return obj;
  }

  createNetwork(payload) {
    const now = Date.now();
    const id = uuid();
    const net = {
      id,
      name: payload.name || 'Untitled Network',
      description: payload.description || '',
      architecture: payload.architecture || { kind: 'classifier', inputDim: 2, outputDim: 2, hidden: [8], activation: 'relu', dropout: 0, classes: ['A', 'B'] },
      training: payload.training || { optimizer: 'adam', learningRate: 0.01, batchSize: 32, epochs: 20, seed: 42, workers: 0 },
      trainingData: payload.trainingData || { samples: [] },
      state: null,
      optimizerState: null,
      tokenizer: null,
      metrics: [],
      encrypted: false,
      script: payload.script || '',
      createdAt: now,
      updatedAt: now
    };
    this._writeRaw(id, net);
    return net;
  }

  updateNetwork(id, patch) {
    const existing = this._readRaw(id);
    if (!existing) throw new Error('Network not found');

    // Handle encryption transitions
    let merged = { ...existing, ...patch, id: existing.id, updatedAt: Date.now() };

    if (patch.encryptionIntent === 'enable' && patch.passphrase) {
      const payload = { state: merged.state, tokenizer: merged.tokenizer };
      merged.encryptedState = this._encryptPayload(payload, patch.passphrase);
      merged.state = null;
      merged.tokenizer = null;
      merged.encrypted = true;
      delete merged.passphrase;
      delete merged.encryptionIntent;
    }
    if (patch.encryptionIntent === 'disable' && existing.encrypted && patch.passphrase) {
      const payload = this._decryptPayload(existing.encryptedState, patch.passphrase);
      merged.state = payload.state;
      merged.tokenizer = payload.tokenizer;
      delete merged.encryptedState;
      merged.encrypted = false;
      delete merged.passphrase;
      delete merged.encryptionIntent;
    }
    // If encrypted, keep encryptedState; don't write decrypted state to disk.
    if (merged.encrypted && !merged.encryptedState && existing.encryptedState) {
      merged.encryptedState = existing.encryptedState;
    }
    // Re-encrypt on save when we have a state loaded
    if (merged.encrypted && merged.state && patch.passphrase) {
      merged.encryptedState = this._encryptPayload({ state: merged.state, tokenizer: merged.tokenizer }, patch.passphrase);
      merged.state = null;
      merged.tokenizer = null;
      delete merged.passphrase;
    }

    this._writeRaw(id, merged);
    return this.getNetwork(id);
  }

  deleteNetwork(id) {
    const p = this._netPath(id);
    if (fs.existsSync(p)) fs.unlinkSync(p);
    return true;
  }

  duplicateNetwork(id) {
    const existing = this._readRaw(id);
    if (!existing) throw new Error('Network not found');
    const copy = { ...existing, id: uuid(), name: existing.name + ' (copy)', createdAt: Date.now(), updatedAt: Date.now() };
    this._writeRaw(copy.id, copy);
    return copy;
  }

  importNetwork(data) {
    if (!data || !data.architecture) throw new Error('Invalid network file');
    const id = uuid();
    const now = Date.now();
    const net = { ...data, id, createdAt: now, updatedAt: now };
    this._writeRaw(id, net);
    return net;
  }

  // ── Backup API ─────────────────────────────────────────────────────────────

  _backupDir(netId) {
    const d = path.join(this.rootDir, 'backups', netId);
    if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
    return d;
  }

  listBackups(netId) {
    const d = path.join(this.rootDir, 'backups', netId);
    if (!fs.existsSync(d)) return [];
    const out = [];
    for (const f of fs.readdirSync(d).filter(f => f.endsWith('.json'))) {
      try {
        const obj = JSON.parse(fs.readFileSync(path.join(d, f), 'utf-8'));
        out.push({ id: obj.backupId, label: obj.label || '', createdAt: obj.createdAt, hasWeights: !!obj.network?.state });
      } catch {}
    }
    out.sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
    return out;
  }

  createBackup(netId, label) {
    const network = this._readRaw(netId);
    if (!network) throw new Error('Network not found');
    const backupId = uuid();
    const now = Date.now();
    const backup = {
      backupId,
      networkId: netId,
      label: label || `Backup — ${new Date(now).toLocaleString()}`,
      createdAt: now,
      network: { ...network }
    };
    fs.writeFileSync(path.join(this._backupDir(netId), `${backupId}.json`), JSON.stringify(backup, null, 2));
    return { id: backupId, label: backup.label, createdAt: now, hasWeights: !!network.state };
  }

  deleteBackup(netId, backupId) {
    const p = path.join(this.rootDir, 'backups', netId, `${backupId}.json`);
    if (fs.existsSync(p)) fs.unlinkSync(p);
    return true;
  }

  restoreBackup(netId, backupId) {
    const p = path.join(this.rootDir, 'backups', netId, `${backupId}.json`);
    if (!fs.existsSync(p)) throw new Error('Backup not found');
    const backup = JSON.parse(fs.readFileSync(p, 'utf-8'));
    const restored = { ...backup.network, id: netId, updatedAt: Date.now() };
    this._writeRaw(netId, restored);
    return this.getNetwork(netId);
  }

  getBackupPath(netId, backupId) {
    return path.join(this.rootDir, 'backups', netId, `${backupId}.json`);
  }

  // Used by trainer/API server to save back state after training (bypasses patch semantics).
  saveTrainedState(id, { state, optimizerState, tokenizer, metrics, architecture }) {
    const existing = this._readRaw(id);
    if (!existing) throw new Error('Network not found');
    existing.state = state;
    // Persist optimizer state so the next "Continue training" picks up where
    // we left off without an Adam warmup bump. Cleared when arch shape changes.
    if (optimizerState !== undefined) existing.optimizerState = optimizerState;
    if (tokenizer) existing.tokenizer = tokenizer;
    if (metrics) existing.metrics = (existing.metrics || []).concat(metrics);
    // The trainer may have mutated the architecture during training (most
    // commonly: setting vocabSize after the tokenizer was built from corpus,
    // or flipping isChat=true). Persist the trainer's view so the next
    // continue-training run sees a consistent (arch ↔ state ↔ tokenizer)
    // triple. Without this, a 0-vocab arch on disk + a 47-vocab saved state
    // would cause the trainer to detect a "mismatch" and blow away the model.
    if (architecture) existing.architecture = architecture;
    existing.updatedAt = Date.now();
    // If it was encrypted, require re-encryption before write — in practice we don't auto-encrypt after each train.
    // For simplicity, training on an encrypted net writes decrypted state; user is prompted elsewhere to re-encrypt.
    existing.encrypted = false;
    delete existing.encryptedState;
    this._writeRaw(id, existing);
    return existing;
  }
}

module.exports = { Storage };
