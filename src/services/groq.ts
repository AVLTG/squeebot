import Groq from "groq-sdk";
import { config } from "../config.js";

/**
 * Groq SDK wrapper.
 *
 * Groq's API is OpenAI-compatible. Free tier on llama-3.3-70b-versatile is
 * generous (~1k RPD) and inference is blazing fast (~200+ tok/s) — much
 * higher headroom than Gemini Flash Lite free tier (~20 RPD).
 */

let client: Groq | null = null;

export function getGroqClient(): Groq {
  if (!client) {
    if (!config.groqApiKey) {
      throw new Error(
        "GROQ_API_KEY is not set. Add it to .env to use Groq."
      );
    }
    client = new Groq({ apiKey: config.groqApiKey });
  }
  return client;
}

// llama-3.3-70b-versatile — Groq's flagship general-purpose model.
// 128k context, JSON mode supported, strong character roleplay ability.
// Free tier limits listed at https://console.groq.com/settings/limits
export const GROQ_MODEL = "llama-3.3-70b-versatile";

export interface GroqCallOpts {
  systemPrompt: string;
  userTurn: string;
  temperature: number;
  maxOutputTokens: number;
}

/**
 * Call Groq with JSON-mode enabled. Returns the raw text body of the first
 * choice (which should be a JSON string matching {reply, memory}).
 *
 * Groq's JSON mode requires the word "JSON" to appear somewhere in the
 * messages, so we append a short schema reminder to the system prompt here
 * rather than polluting the shared prompt.ts.
 */
export async function callGroq(opts: GroqCallOpts): Promise<string | null> {
  const jsonReminder = `\n\n# Response format
Respond with ONLY a JSON object matching this shape — no prose before or after:
{
  "reply": "<what Squee says to the user, 1-3 sentences, max 500 chars, in Squee's voice>",
  "memory": "<Squee's updated 1-line note about this user, in Squee's voice — the user never sees this>"
}`;

  const completion = await getGroqClient().chat.completions.create({
    model: GROQ_MODEL,
    messages: [
      { role: "system", content: opts.systemPrompt + jsonReminder },
      { role: "user", content: opts.userTurn },
    ],
    temperature: opts.temperature,
    max_tokens: opts.maxOutputTokens,
    response_format: { type: "json_object" },
  });

  return completion.choices[0]?.message?.content ?? null;
}
