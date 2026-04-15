import { GoogleGenAI } from "@google/genai";
import { config } from "../config.js";

/**
 * Gemini SDK wrapper.
 *
 * We use @google/genai (the unified Google AI SDK, replaces @google/generative-ai).
 * The client is lazily initialized so tests / dev work without a key set.
 */

let client: GoogleGenAI | null = null;

export function getGeminiClient(): GoogleGenAI {
  if (!client) {
    if (!config.geminiApiKey) {
      throw new Error(
        "GEMINI_API_KEY is not set. Add it to .env to use Gemini."
      );
    }
    client = new GoogleGenAI({ apiKey: config.geminiApiKey });
  }
  return client;
}

// Gemini 2.5 Flash Lite — smaller/faster/cheaper than regular Flash.
// Free tier: ~15 RPM, ~1000 RPD. More forgiving than full Flash during
// free-tier capacity crunches, and plenty smart for character roleplay
// where the system prompt does most of the heavy lifting.
// See https://ai.google.dev/gemini-api/docs/models for the latest list.
export const GEMINI_MODEL = "gemini-2.5-flash-lite";
