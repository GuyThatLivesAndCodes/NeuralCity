'use strict';

// Single-session training worker.
//
// The entire training loop runs here so the Electron main-process event loop
// stays free for IPC — the UI never blocks waiting on a training step.
//
// Protocol
//   main → worker: { type: 'stop' }
//   worker → main: { type: 'progress', epoch, totalEpochs, step, totalSteps, loss, elapsedMs }
//   worker → main: { type: 'log', line, level }
//   worker → main: { type: 'done', result }    — result is JSON-serialisable
//   worker → main: { type: 'error', message, stack }

const { parentPort, workerData } = require('worker_threads');
const { trainNetwork } = require('./trainer');

let _stopRequested = false;
parentPort.on('message', (msg) => {
  if (msg.type === 'stop') _stopRequested = true;
});

const { network, fromScratch } = workerData;

trainNetwork(network, {
  fromScratch,
  onProgress: (p) => parentPort.postMessage({ type: 'progress', ...p }),
  shouldStop:  () => _stopRequested,
  log: (line, level = 'info') => parentPort.postMessage({ type: 'log', line, level }),
})
  .then((result) => parentPort.postMessage({ type: 'done', result }))
  .catch((e)     => parentPort.postMessage({ type: 'error', message: e.message, stack: e.stack }));
