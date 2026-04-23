const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('nc', {
  networks: {
    list: () => ipcRenderer.invoke('networks:list'),
    get: (id) => ipcRenderer.invoke('networks:get', id),
    create: (payload) => ipcRenderer.invoke('networks:create', payload),
    update: (id, patch) => ipcRenderer.invoke('networks:update', id, patch),
    delete: (id) => ipcRenderer.invoke('networks:delete', id),
    duplicate: (id) => ipcRenderer.invoke('networks:duplicate', id),
    exportNet: (id) => ipcRenderer.invoke('networks:export', id),
    importNet: () => ipcRenderer.invoke('networks:import')
  },
  training: {
    start: (id, opts) => ipcRenderer.invoke('training:start', id, opts),
    stop: (id) => ipcRenderer.invoke('training:stop', id),
    status: (id) => ipcRenderer.invoke('training:status', id),
    onProgress: (cb) => ipcRenderer.on('training:progress', (_, p) => cb(p)),
    onDone: (cb) => ipcRenderer.on('training:done', (_, p) => cb(p)),
    onError: (cb) => ipcRenderer.on('training:error', (_, p) => cb(p))
  },
  inference: {
    run: (id, input) => ipcRenderer.invoke('inference:run', id, input)
  },
  api: {
    start: (id, port) => ipcRenderer.invoke('api:start', id, port),
    stop: (id) => ipcRenderer.invoke('api:stop', id),
    status: (id) => ipcRenderer.invoke('api:status', id),
    list: () => ipcRenderer.invoke('api:list'),
    onLog: (cb) => ipcRenderer.on('api:log', (_, l) => cb(l))
  },
  backups: {
    list:     (netId)           => ipcRenderer.invoke('backups:list', netId),
    create:   (netId, label)    => ipcRenderer.invoke('backups:create', netId, label),
    delete:   (netId, backupId) => ipcRenderer.invoke('backups:delete', netId, backupId),
    restore:  (netId, backupId) => ipcRenderer.invoke('backups:restore', netId, backupId),
    download: (netId, backupId) => ipcRenderer.invoke('backups:download', netId, backupId)
  },
  script: {
    run: (id, code) => ipcRenderer.invoke('script:run', id, code)
  },
  system: {
    info: () => ipcRenderer.invoke('system:info')
  },
  dialog: {
    openFile: (opts) => ipcRenderer.invoke('dialog:openFile', opts),
    readTextFile: (opts) => ipcRenderer.invoke('dialog:readTextFile', opts)
  }
});
