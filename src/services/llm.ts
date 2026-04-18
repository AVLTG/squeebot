/**
 * LLM provider dispatcher.
 *
 * Routes calls to Groq (primary) or Gemini (fallback) based on LLM_PROVIDER
 * env var. Both providers return the raw text body of a JSON-formatted reply;
 * parsing and error-handling live in squeeBrain.ts.
 */

import { config } from "../config.js";
import { callGroq } from "./groq.js";
import { callGemini } from "./gemini.js";

export interface LLMCallOpts {
  systemPrompt: string;
  userTurn: string;
  temperature: number;
  maxOutputTokens: number;
}

export async function callLLM(opts: LLMCallOpts): Promise<string | null> {
  if (config.llmProvider === "groq") return callGroq(opts);
  return callGemini(opts);
}
