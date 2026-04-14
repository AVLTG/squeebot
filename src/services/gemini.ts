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

// Gemini 2.5 Flash — cheapest/fastest current model with 1,500 req/day free tier.
// See https://ai.google.dev/gemini-api/docs/models for the latest list.
export const GEMINI_MODEL = "gemini-2.5-flash";
