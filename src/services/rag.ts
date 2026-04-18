/**
 * Voice-line RAG using local sentence embeddings.
 *
 * At startup, loads Xenova/all-MiniLM-L6-v2 (~85MB on disk, ~90MB resident)
 * and pulls every precomputed voiceline vector out of SQLite into memory.
 * Per-request, embeds the user's message and does a brute-force cosine
 * similarity scan across ~10k 384-dim vectors (sub-50ms on a Pi 4).
 *
 * Ingest the voicelines table first:  npm run ingest
 */

import { pipeline } from "@xenova/transformers";
import Database from "better-sqlite3";
import { config } from "../config.js";
import { logger } from "../utils/logger.js";

const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";
const DIM = 384;

interface VoicelineMeta {
  line: string;
  tags: string[];
  category: string;
}

// The transformers.js pipeline returns a union-typed callable; we know the
// feature-extraction task shape so we narrow to a minimal interface.
type FeatureExtractor = (
  texts: string | string[],
  opts: { pooling: "mean"; normalize: boolean }
) => Promise<{ data: Float32Array }>;

// Lazy-initialized — call initRAG() eagerly at boot to avoid a cold-start
// penalty on the first user message.
let extractor: FeatureExtractor | null = null;
const vectors: Float32Array[] = [];
const metas: VoicelineMeta[] = [];

export async function initRAG(): Promise<void> {
  if (extractor) return;

  logger.info(`Loading embedding model: ${MODEL_NAME}`);
  extractor = (await pipeline("feature-extraction", MODEL_NAME)) as unknown as FeatureExtractor;
  logger.info("Embedding model loaded.");

  const db = new Database(config.dbPath, { readonly: true });
  try {
    const rows = db
      .prepare("SELECT line, tags, category, vec FROM voicelines")
      .all() as Array<{ line: string; tags: string; category: string; vec: Buffer }>;

    for (const row of rows) {
      // The Buffer backs a slice of a larger block — copy to a fresh
      // Float32Array so we don't hold refs to the DB's internal memory.
      const view = new Float32Array(row.vec.buffer, row.vec.byteOffset, DIM);
      vectors.push(new Float32Array(view));
      metas.push({
        line: row.line,
        tags: JSON.parse(row.tags),
        category: row.category,
      });
    }
  } finally {
    db.close();
  }

  if (metas.length === 0) {
    logger.warn(
      "RAG voicelines table is empty. Run `npm run ingest` to populate it. " +
        "Squee will reply without voiceline grounding until then."
    );
  } else {
    logger.info(`RAG initialized with ${metas.length} voicelines.`);
  }
}

/**
 * Return the top-k voicelines most semantically similar to `query`.
 * Empty array if RAG is not initialized or has no data.
 */
export async function retrieveVoicelines(
  query: string,
  k = 5
): Promise<string[]> {
  if (!extractor) await initRAG();
  if (metas.length === 0 || !extractor) return [];

  const output = await extractor(query, { pooling: "mean", normalize: true });
  const qvec = output.data;

  // Vectors are pre-normalized, so cosine similarity == dot product.
  // Heap-based top-k would be more efficient but a full sort on 10k items is ~2ms.
  const scored = new Array<{ idx: number; score: number }>(vectors.length);
  for (let i = 0; i < vectors.length; i++) {
    scored[i] = { idx: i, score: dot(qvec, vectors[i]!) };
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k).map((s) => metas[s.idx]!.line);
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}
