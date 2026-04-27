'use strict';

const fs = require('fs');
const path = require('path');

class PluginLoader {
  constructor(pluginDir) {
    this.pluginDir = pluginDir;
    this.loaded = []; // { id, manifest, rendererCode }
  }

  // Copy bundled example plugins into userData on first launch.
  seedBuiltins(builtinsDir) {
    if (!fs.existsSync(builtinsDir)) return;
    fs.mkdirSync(this.pluginDir, { recursive: true });
    for (const id of fs.readdirSync(builtinsDir)) {
      const src = path.join(builtinsDir, id);
      const dst = path.join(this.pluginDir, id);
      if (!fs.statSync(src).isDirectory()) continue;
      // Always overwrite builtin files so source changes are picked up on restart
      fs.mkdirSync(dst, { recursive: true });
      for (const file of fs.readdirSync(src)) {
        fs.copyFileSync(path.join(src, file), path.join(dst, file));
      }
    }
  }

  // Load all installed plugins and register their main-process IPC handlers.
  // storage is optional; passed to factory-style plugins (those that export a function).
  load(ipcMain, storage) {
    if (!fs.existsSync(this.pluginDir)) {
      fs.mkdirSync(this.pluginDir, { recursive: true });
      return;
    }
    for (const id of fs.readdirSync(this.pluginDir)) {
      const dir = path.join(this.pluginDir, id);
      try {
        if (!fs.statSync(dir).isDirectory()) continue;
        const manifestPath = path.join(dir, 'manifest.json');
        if (!fs.existsSync(manifestPath)) continue;
        const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));

        const rendererPath = path.join(dir, 'renderer.js');
        const rendererCode = fs.existsSync(rendererPath)
          ? fs.readFileSync(rendererPath, 'utf-8')
          : '';

        const mainPath = path.join(dir, 'main.js');
        if (fs.existsSync(mainPath)) {
          // Clear require cache so reinstalled plugins reload cleanly.
          delete require.cache[require.resolve(mainPath)];
          let pluginMain = require(mainPath);
          // Factory pattern: if the plugin exports a function, call it with context.
          if (typeof pluginMain === 'function') pluginMain = pluginMain({ storage });
          if (pluginMain && pluginMain.mainHandlers) {
            for (const [channel, handler] of Object.entries(pluginMain.mainHandlers)) {
              ipcMain.handle(`plugin:${id}:${channel}`, handler);
            }
          }
        }

        this.loaded.push({ id, manifest, rendererCode });
      } catch (e) {
        console.error(`[PluginLoader] Failed to load plugin "${id}":`, e.message);
      }
    }
  }

  // Return plugin metadata + renderer code for the renderer process.
  list() {
    return this.loaded.map(p => ({
      id: p.id,
      manifest: p.manifest,
      rendererCode: p.rendererCode
    }));
  }

  // Install a .nbpl file (JSON) into the plugins directory.
  install(ncplPath) {
    const raw = fs.readFileSync(ncplPath, 'utf-8');
    let plugin;
    try { plugin = JSON.parse(raw); }
    catch (e) { throw new Error('Invalid .nbpl file: not valid JSON'); }

    const { id, name, version, description, author, mainCode, rendererCode } = plugin;
    if (!id || typeof id !== 'string') throw new Error('.nbpl missing required "id" field');
    if (!/^[a-z0-9_-]+$/i.test(id)) throw new Error(`.nbpl id "${id}" contains invalid characters`);

    const dir = path.join(this.pluginDir, id);
    fs.mkdirSync(dir, { recursive: true });

    const manifest = { id, name: name || id, version: version || '0.0.0', description: description || '', author: author || '' };
    fs.writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2));
    if (mainCode) fs.writeFileSync(path.join(dir, 'main.js'), mainCode);
    if (rendererCode) fs.writeFileSync(path.join(dir, 'renderer.js'), rendererCode);

    return { id, name: manifest.name, needsRestart: true };
  }

  // Remove a plugin's directory.
  uninstall(id) {
    if (!id || !/^[a-z0-9_-]+$/i.test(id)) throw new Error('Invalid plugin id');
    const dir = path.join(this.pluginDir, id);
    if (!fs.existsSync(dir)) throw new Error(`Plugin "${id}" not found`);
    fs.rmSync(dir, { recursive: true, force: true });
    return { ok: true };
  }
}

module.exports = PluginLoader;
