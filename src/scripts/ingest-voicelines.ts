/**
 * Ingest voicelines into SQLite with precomputed embeddings.
 *
 * Reads every voicelines/*.json file, embeds each line with
 * Xenova/all-MiniLM-L6-v2 (384-dim), and stores the result in the
 * `voicelines` table of the main squeebot DB.
 *
 * Safe to re-run — wipes and rebuilds the table each time.
 *
 * Usage (from repo root):  npm run ingest
 *
 * On a Raspberry Pi 4, expect 15-45 minutes for ~10k lines depending on
 * what else is using the CPU. Tip: stop the Terraria server with
 * `tmod-stop` to give the embedder full CPU.
 */

import { pipeline } from "@xenova/transformers";
import Database from "better-sqlite3";
import { readdirSync, readFileSync, mkdirSync } from "node:fs";
import { basename, join, dirname } from "node:path";

// Minimal shape for the feature-extraction pipeline callable. The transformers.js
// return type is a broad union — we narrow to what we actually use.
type FeatureExtractor = (
  texts: string | string[],
  opts: { pooling: "mean"; normalize: boolean }
) => Promise<{ data: Float32Array }>;

const VOICELINES_DIR = "./voicelines";
const DB_PATH = "./data/squeebot.db";
const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";
const DIM = 384;
const BATCH_SIZE = 16;

interface VoicelineEntry {
  line: string;
  tags: string[];
  source: string;
  category: string;
}

async function main(): Promise<void> {
  mkdirSync(dirname(DB_PATH), { recursive: true });

  console.log(`Loading embedding model: ${MODEL_NAME}`);
  console.log("(first run downloads ~25MB of model weights; subsequent runs use the HF cache)");
  const extractor = (await pipeline(
    "feature-extraction",
    MODEL_NAME
  )) as unknown as FeatureExtractor;
  console.log("Model loaded.\n");

  const db = new Database(DB_PATH);
  db.pragma("journal_mode = WAL");
  db.exec(`
    CREATE TABLE IF NOT EXISTS voicelines (
      id       INTEGER PRIMARY KEY AUTOINCREMENT,
      line     TEXT NOT NULL,
      tags     TEXT NOT NULL,
      category TEXT NOT NULL,
      source   TEXT NOT NULL,
      vec      BLOB NOT NULL
    );
  `);
  db.exec("DELETE FROM voicelines;");
  console.log("Existing voicelines cleared.");

  const entries: VoicelineEntry[] = [];
  const files = readdirSync(VOICELINES_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort();
  for (const file of files) {
    const category = basename(file, ".json");
    const items = JSON.parse(readFileSync(join(VOICELINES_DIR, file), "utf-8")) as Array<{
      line: string;
      tags?: string[];
      source?: string;
    }>;
    for (const item of items) {
      if (!item.line) continue;
      entries.push({
        line: item.line,
        tags: item.tags ?? [],
        source: item.source ?? "synthetic",
        category,
      });
    }
  }
  console.log(`Found ${entries.length} voicelines across ${files.length} categories.\n`);

  const insert = db.prepare(
    "INSERT INTO voicelines (line, tags, category, source, vec) VALUES (?, ?, ?, ?, ?)"
  );
  const insertBatch = db.transaction(
    (rows: Array<VoicelineEntry & { vec: Buffer }>): void => {
      for (const r of rows) {
        insert.run(r.line, JSON.stringify(r.tags), r.category, r.source, r.vec);
      }
    }
  );

  const startTime = Date.now();
  let lastLogTime = startTime;
  for (let i = 0; i < entries.length; i += BATCH_SIZE) {
    const batch = entries.slice(i, i + BATCH_SIZE);
    const texts = batch.map((e) => e.line);
    // Output is a Tensor of shape [batch_size, DIM]; .data is a flat Float32Array
    const output = await extractor(texts, { pooling: "mean", normalize: true });
    const flat = output.data;

    const rows = batch.map((entry, j) => {
      const vec = flat.slice(j * DIM, (j + 1) * DIM);
      return { ...entry, vec: Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength) };
    });
    insertBatch(rows);

    // Throttled progress line
    const now = Date.now();
    if (now - lastLogTime > 2000 || i + BATCH_SIZE >= entries.length) {
      const done = Math.min(i + BATCH_SIZE, entries.length);
      const pct = ((done / entries.length) * 100).toFixed(1);
      const elapsed = (now - startTime) / 1000;
      const rate = done / elapsed;
      const remainingSecs = Math.round((entries.length - done) / rate);
      const mm = Math.floor(remainingSecs / 60);
      const ss = remainingSecs % 60;
      console.log(
        `  [${pct.padStart(5)}%]  ${done}/${entries.length}  ` +
          `(${rate.toFixed(1)}/s, ETA ${mm}m${ss.toString().padStart(2, "0")}s)`
      );
      lastLogTime = now;
    }
  }

  const count = (db.prepare("SELECT COUNT(*) as c FROM voicelines").get() as { c: number }).c;
  db.close();

  const totalSecs = Math.round((Date.now() - startTime) / 1000);
  console.log(
    `\nDone. Ingested ${count} voicelines in ${Math.floor(totalSecs / 60)}m${totalSecs % 60}s.`
  );
}

main().catch((err) => {
  console.error("Ingestion failed:", err);
  process.exit(1);
});
