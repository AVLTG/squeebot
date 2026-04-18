import { GoogleGenAI, Type } from "@google/genai";
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
// Free tier used to be ~1000 RPD but dropped to ~20 RPD as of 2026-04.
// Kept as a fallback provider; primary is now Groq (see groq.ts).
export const GEMINI_MODEL = "gemini-2.5-flash-lite";

export interface GeminiCallOpts {
  systemPrompt: string;
  userTurn: string;
  temperature: number;
  maxOutputTokens: number;
}

const RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    reply: {
      type: Type.STRING,
      description: "What Squee says to the user (1-3 sentences, max 500 chars).",
    },
    memory: {
      type: Type.STRING,
      description:
        "Squee's updated goblin note about this user, in Squee's voice (1-2 sentences). The user never sees this.",
    },
  },
  required: ["reply", "memory"],
  propertyOrdering: ["reply", "memory"],
} as const;

/**
 * Call Gemini with structured JSON output. Returns the raw text body
 * (a JSON string matching {reply, memory}).
 */
export async function callGemini(opts: GeminiCallOpts): Promise<string | null> {
  const response = await getGeminiClient().models.generateContent({
    model: GEMINI_MODEL,
    contents: [{ role: "user", parts: [{ text: opts.userTurn }] }],
    config: {
      systemInstruction: opts.systemPrompt,
      temperature: opts.temperature,
      maxOutputTokens: opts.maxOutputTokens,
      thinkingConfig: { thinkingBudget: 0 },
      responseMimeType: "application/json",
      responseSchema: RESPONSE_SCHEMA,
    },
  });
  return response.text?.trim() ?? null;
}
