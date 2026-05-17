/**
 * Minimal IndexedDB key/value store for plugin payloads too large for
 * localStorage (which is ~5 MB per origin). Used by the Image Classification
 * plugin to persist drawn/uploaded image samples — a 64×64 RGB sample is
 * already ~12k floats and a handful of them overflow localStorage's quota.
 *
 * One database, one object store, keyed by an arbitrary string. Values are
 * structured-cloneable, so plain objects, arrays, and typed arrays all work.
 */

const DB_NAME = 'neuralcabin'
const STORE = 'plugin_kv'
const VERSION = 1

let dbPromise: Promise<IDBDatabase> | null = null

function openDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE)
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error ?? new Error('IDB open failed'))
  })
  return dbPromise
}

async function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDB()
  return new Promise<T>((resolve, reject) => {
    const t = db.transaction(STORE, mode)
    const req = fn(t.objectStore(STORE))
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error ?? new Error('IDB tx failed'))
  })
}

export const idbKV = {
  get:    <T = unknown>(key: string) => tx<T>('readonly',  s => s.get(key) as IDBRequest<T>),
  set:    (key: string, value: unknown) => tx<IDBValidKey>('readwrite', s => s.put(value, key)),
  delete: (key: string) => tx<undefined>('readwrite', s => s.delete(key) as IDBRequest<undefined>),
}
