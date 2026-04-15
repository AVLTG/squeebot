/**
 * Per-user "goblin notes" — Squee's memory about each user he talks to.
 *
 * Stored as a single-line note per user, written in Squee's own voice
 * (updated by the model on each interaction).
 *
 * Uses a simple SQLite table. For ~20 users this is overkill, but it gives us:
 *   - atomic writes (no JSON race conditions)
 *   - easy inspection (`sqlite3 data/squeebot.db "SELECT * FROM user_memory"`)
 *   - room to grow if we add more fields later (last_seen, interaction_count, etc.)
 */

import Database from "better-sqlite3";
import { existsSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { config } from "../config.js";
import { logger } from "../utils/logger.js";

let db: Database.Database | null = null;

function getDb(): Database.Database {
  if (db) return db;

  // Ensure the data directory exists
  const dir = dirname(config.dbPath);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
    logger.info(`Created data directory: ${dir}`);
  }

  db = new Database(config.dbPath);
  // WAL = better concurrency and crash safety
  db.pragma("journal_mode = WAL");

  // Table: user_memory
  //   user_id    - Discord user snowflake (primary key)
  //   user_name  - display name (for debug; not authoritative)
  //   note       - Squee's current 1-line note about this user
  //   updated_at - ISO timestamp, last time Squee updated his note
  //   created_at - ISO timestamp, first time we saw this user
  db.exec(`
    CREATE TABLE IF NOT EXISTS user_memory (
      user_id    TEXT PRIMARY KEY,
      user_name  TEXT NOT NULL,
      note       TEXT NOT NULL DEFAULT '',
      updated_at TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
  `);

  logger.info(`Memory DB ready at ${config.dbPath}`);
  return db;
}

export interface UserMemory {
  userId: string;
  userName: string;
  note: string;
  updatedAt: string;
  createdAt: string;
}

/**
 * Retrieve Squee's current note on a user. Returns null if we've never seen them.
 */
export function getMemory(userId: string): UserMemory | null {
  const row = getDb()
    .prepare(
      `SELECT user_id as userId, user_name as userName, note,
              updated_at as updatedAt, created_at as createdAt
       FROM user_memory WHERE user_id = ?`
    )
    .get(userId) as UserMemory | undefined;
  return row ?? null;
}

/**
 * Upsert Squee's note for a user. Creates the row if it doesn't exist.
 * No-op if `note` is empty (we only save when the model actually wrote something).
 */
export function setMemory(
  userId: string,
  userName: string,
  note: string
): void {
  const trimmed = note.trim();
  if (!trimmed) return;

  const now = new Date().toISOString();
  getDb()
    .prepare(
      `INSERT INTO user_memory (user_id, user_name, note, updated_at, created_at)
       VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(user_id) DO UPDATE SET
         user_name = excluded.user_name,
         note = excluded.note,
         updated_at = excluded.updated_at`
    )
    .run(userId, userName, trimmed, now, now);

  logger.debug(`Memory updated for ${userName} (${userId}): "${trimmed}"`);
}

/**
 * Close the DB. Call this on graceful shutdown so WAL is checkpointed.
 */
export function closeMemory(): void {
  if (db) {
    db.close();
    db = null;
  }
}
