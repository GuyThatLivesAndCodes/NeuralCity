const { app, BrowserWindow, ipcMain, dialog, Menu } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { Storage } = require('./storage');
const { ApiServer } = require('./api-server');
const { TrainingManager } = require('./training-manager');
const PluginLoader = require('./plugin-loader');

const isDev = process.argv.includes('--dev');
const userDataDir = path.join(app.getPath('userData'), 'NeuralCabin');
if (!fs.existsSync(userDataDir)) fs.mkdirSync(userDataDir, { recursive: true });

const storage = new Storage(userDataDir);
const apiServer = new ApiServer(storage);
const trainer = new TrainingManager(storage);

const pluginLoader = new PluginLoader(path.join(userDataDir, 'plugins'));
pluginLoader.seedBuiltins(path.join(__dirname, '..', 'plugins'));
pluginLoader.load(ipcMain, storage);

let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 650,
    backgroundColor: '#0a0a0a',
    title: 'NeuralCabin',
    autoHideMenuBar: true,
    icon: path.join(__dirname, '..', '..', 'assets', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
  if (isDev) mainWindow.webContents.openDevTools({ mode: 'detach' });

  trainer.on('progress', (payload) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('training:progress', payload);
    }
  });
  trainer.on('done', (payload) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('training:done', payload);
    }
  });
  trainer.on('error', (payload) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('training:error', payload);
    }
  });
  trainer.on('log', (payload) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('training:log', payload);
    }
  });

  apiServer.on('log', (line) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('api:log', line);
    }
  });
}

Menu.setApplicationMenu(null);

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  apiServer.stopAll();
  trainer.stopAll();
  if (process.platform !== 'darwin') app.quit();
});

// ---------- IPC ----------

ipcMain.handle('networks:list', () => storage.listNetworks());
ipcMain.handle('networks:get', (_, id) => storage.getNetwork(id));
ipcMain.handle('networks:create', (_, payload) => storage.createNetwork(payload));
ipcMain.handle('networks:update', (_, id, patch) => storage.updateNetwork(id, patch));
ipcMain.handle('networks:delete', (_, id) => {
  trainer.stop(id);
  apiServer.stop(id);
  return storage.deleteNetwork(id);
});
ipcMain.handle('networks:duplicate', (_, id) => storage.duplicateNetwork(id));

ipcMain.handle('networks:export', async (_, id) => {
  const net = storage.getNetwork(id);
  if (!net) throw new Error('Network not found');
  const result = await dialog.showSaveDialog(mainWindow, {
    title: 'Export Network',
    defaultPath: `${net.name.replace(/[^a-z0-9_\-]/gi, '_')}.nbnet.json`,
    filters: [{ name: 'NeuralCabin Network', extensions: ['json'] }]
  });
  if (result.canceled || !result.filePath) return null;
  fs.writeFileSync(result.filePath, JSON.stringify(net, null, 2));
  return result.filePath;
});

ipcMain.handle('networks:import', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Import Network',
    filters: [{ name: 'NeuralCabin Network', extensions: ['json'] }],
    properties: ['openFile']
  });
  if (result.canceled || !result.filePaths[0]) return null;
  const data = JSON.parse(fs.readFileSync(result.filePaths[0], 'utf-8'));
  return storage.importNetwork(data);
});

ipcMain.handle('backups:list',    (_, netId)           => storage.listBackups(netId));
ipcMain.handle('backups:create',  (_, netId, label)    => storage.createBackup(netId, label));
ipcMain.handle('backups:delete',  (_, netId, backupId) => storage.deleteBackup(netId, backupId));
ipcMain.handle('backups:restore', (_, netId, backupId) => storage.restoreBackup(netId, backupId));
ipcMain.handle('backups:download', async (_, netId, backupId) => {
  const srcPath = storage.getBackupPath(netId, backupId);
  if (!fs.existsSync(srcPath)) throw new Error('Backup not found');
  const backup = JSON.parse(fs.readFileSync(srcPath, 'utf-8'));
  const safeName = (backup.label || 'backup').replace(/[^a-z0-9_\-]/gi, '_').slice(0, 48);
  const result = await dialog.showSaveDialog(mainWindow, {
    title: 'Save Backup',
    defaultPath: `${safeName}.nbbackup.json`,
    filters: [{ name: 'NeuralCabin Backup', extensions: ['json'] }]
  });
  if (result.canceled || !result.filePath) return null;
  fs.copyFileSync(srcPath, result.filePath);
  return result.filePath;
});

ipcMain.handle('training:start', (_, id, opts) => trainer.start(id, opts));
ipcMain.handle('training:stop', (_, id) => trainer.stop(id));
ipcMain.handle('training:status', (_, id) => trainer.status(id));

ipcMain.handle('inference:run', async (_, id, input) => trainer.infer(id, input));

let _streamCancel = null;
ipcMain.handle('inference:stream-start', async (event, id, input) => {
  if (_streamCancel) _streamCancel.cancelled = true;
  const cancelRef = { cancelled: false };
  _streamCancel = cancelRef;
  try {
    return await trainer.inferStream(id, input, (chunk) => {
      if (!event.sender.isDestroyed()) event.sender.send('inference:token', chunk);
    }, cancelRef);
  } finally {
    if (_streamCancel === cancelRef) _streamCancel = null;
  }
});
ipcMain.on('inference:cancel', () => { if (_streamCancel) _streamCancel.cancelled = true; });

ipcMain.handle('plugins:list', () => pluginLoader.list());
ipcMain.handle('plugins:install', async () => {
  const res = await dialog.showOpenDialog(mainWindow, {
    title: 'Install Plugin',
    filters: [{ name: 'NeuralCabin Plugin', extensions: ['nbpl'] }],
    properties: ['openFile']
  });
  if (res.canceled || !res.filePaths[0]) return null;
  return pluginLoader.install(res.filePaths[0]);
});
ipcMain.handle('plugins:uninstall', (_, id) => pluginLoader.uninstall(id));

ipcMain.handle('api:start', (_, id, port, opts) => apiServer.start(id, port, opts || {}));
ipcMain.handle('api:stop', (_, id) => apiServer.stop(id));
ipcMain.handle('api:status', (_, id) => apiServer.status(id));
ipcMain.handle('api:list', () => apiServer.listAll());
ipcMain.handle('api:issue-token', (_, id, expiresIn) => apiServer.issueToken(id, expiresIn));
ipcMain.handle('api:metrics', (_, id) => apiServer.getMetrics(id));

ipcMain.handle('training:backend-info', () => trainer.backendInfo());

ipcMain.handle('script:run', (_, id, code) => trainer.runScript(id, code));

ipcMain.handle('system:info', () => ({
  platform: process.platform,
  arch: process.arch,
  cpus: os.cpus().length,
  mem: os.totalmem(),
  hostIp: getHostIp(),
  version: app.getVersion(),
  userDataDir
}));

ipcMain.handle('dialog:openFile', async (_, options) => {
  const result = await dialog.showOpenDialog(mainWindow, options || {});
  if (result.canceled) return null;
  return result.filePaths[0];
});

ipcMain.handle('dialog:readTextFile', async (_, options) => {
  const result = await dialog.showOpenDialog(mainWindow, options || { properties: ['openFile'] });
  if (result.canceled || !result.filePaths[0]) return null;
  return { path: result.filePaths[0], content: fs.readFileSync(result.filePaths[0], 'utf-8') };
});

ipcMain.handle('gpt:pickDocuments', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Select Documents',
    filters: [{ name: 'Documents', extensions: ['txt', 'md', 'html', 'htm', 'csv', 'json', 'ts', 'js', 'py', 'yaml', 'yml', 'xml', 'zip'] }],
    properties: ['openFile', 'multiSelections']
  });
  if (result.canceled || !result.filePaths.length) return [];
  const MAX = 500 * 1024;
  const TEXT_EXTS = new Set(['txt', 'md', 'html', 'htm', 'csv', 'json', 'ts', 'js', 'py', 'yaml', 'yml', 'xml']);
  const docs = [];
  for (const p of result.filePaths) {
    const ext = path.extname(p).slice(1).toLowerCase();
    if (ext === 'zip') {
      const extractZip = require('extract-zip');
      const extractDir = path.join(os.tmpdir(), `nb-zip-${Date.now()}`);
      fs.mkdirSync(extractDir, { recursive: true });
      try {
        await extractZip(p, { dir: extractDir });
        function walkDir(dir) {
          for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
            const full = path.join(dir, entry.name);
            if (entry.isDirectory()) { walkDir(full); continue; }
            const e = entry.name.split('.').pop().toLowerCase();
            if (!TEXT_EXTS.has(e)) continue;
            let content = '';
            try { content = fs.readFileSync(full, 'utf-8'); } catch {}
            const truncated = content.length > MAX;
            if (truncated) content = content.slice(0, MAX);
            docs.push({ name: entry.name, content, size: content.length, truncated });
          }
        }
        walkDir(extractDir);
      } finally {
        try { fs.rmSync(extractDir, { recursive: true, force: true }); } catch {}
      }
    } else {
      const name = path.basename(p);
      let content = '';
      try { content = fs.readFileSync(p, 'utf-8'); } catch {}
      const truncated = content.length > MAX;
      if (truncated) content = content.slice(0, MAX);
      docs.push({ name, content, size: content.length, truncated });
    }
  }
  return docs;
});

ipcMain.handle('app:restart', () => { app.relaunch(); app.exit(0); });

function getHostIp() {
  const ifaces = os.networkInterfaces();
  for (const name of Object.keys(ifaces)) {
    for (const info of ifaces[name]) {
      if (info.family === 'IPv4' && !info.internal) return info.address;
    }
  }
  return '127.0.0.1';
}
