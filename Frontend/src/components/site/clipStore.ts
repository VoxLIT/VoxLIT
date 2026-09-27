/** Persists the home-page clip (and its transcript) in IndexedDB so a page
 *  refresh doesn't lose it. Browser-local only; every call fails soft. */

const DB_NAME = "voxlit-home";
const STORE = "clip";
const KEY = "current";

export interface StoredClip {
  name: string;
  type: string;
  blob: Blob;
  transcript?: { text: string; seconds: number };
}

const openDb = () =>
  new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });

const run = async <T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T | undefined> => {
  try {
    const db = await openDb();
    return await new Promise<T>((resolve, reject) => {
      const req = fn(db.transaction(STORE, mode).objectStore(STORE));
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    }).finally(() => db.close());
  } catch {
    return undefined;
  }
};

export const loadClip = () => run<StoredClip | undefined>("readonly", (s) => s.get(KEY));

export const saveClip = (file: File) =>
  run("readwrite", (s) => s.put({ name: file.name, type: file.type, blob: file } satisfies StoredClip, KEY));

export const saveTranscript = async (transcript: StoredClip["transcript"]) => {
  const current = await loadClip();
  if (current) await run("readwrite", (s) => s.put({ ...current, transcript } satisfies StoredClip, KEY));
};

export const clearClip = () => run("readwrite", (s) => s.delete(KEY));
