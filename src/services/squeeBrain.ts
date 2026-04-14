/**
 * The Squee Brain — generates in-character responses.
 *
 * Phase 2: calls Gemini Flash with the Squee system prompt.
 * Falls back to hardcoded quote if the API call fails (rate limit, network, etc.).
 *
 * Phase 3 will add per-user memory context.
 * Phase 4 will add RAG-retrieved voice lines.
 */

import { getGeminiClient, GEMINI_MODEL } from "./gemini.js";
import { SQUEE_SYSTEM_PROMPT } from "./prompt.js";
import { logger } from "../utils/logger.js";

export interface BrainContext {
  userId: string;
  userName: string;
  userMessage: string;
}

// If Gemini ever fails, we fall back to one of these so the bot doesn't go silent.
const FALLBACK_QUOTES = [
  "Squee's brain got stuck. Try pokin' Squee again?",
  "Wuh? Squee din't hear ya. Say it again, louder!",
  "Squee got distracted by a bug. What wuz you sayin'?",
  "Gerrard always say Squee don't pay attention. He wrong... mostly.",
  "Squee too busy bein' immortal to answer dat one. Try again!",
];

export async function generateReply(ctx: BrainContext): Promise<string> {
  try {
    const client = getGeminiClient();

    // Build the user turn. We prefix with the username so Squee can reference it.
    const userTurn = `${ctx.userName} says to Squee: ${ctx.userMessage}`;

    const response = await client.models.generateContent({
      model: GEMINI_MODEL,
      contents: [
        {
          role: "user",
          parts: [{ text: userTurn }],
        },
      ],
      config: {
        systemInstruction: SQUEE_SYSTEM_PROMPT,
        temperature: 1.0, // Squee should be unpredictable
        maxOutputTokens: 300, // ~200 words, well under Discord's 2000 char limit
        // Gemini 2.5 models burn tokens on internal "thinking" by default, which
        // counts against maxOutputTokens and causes mid-sentence truncation.
        // Squee is a goblin — he does not need to think before speaking.
        thinkingConfig: { thinkingBudget: 0 },
      },
    });

    const text = response.text?.trim();
    if (!text) {
      logger.warn("Gemini returned empty response, using fallback");
      return pickFallback();
    }

    logger.debug(`Gemini reply (${text.length} chars): ${text}`);
    return text;
  } catch (err) {
    logger.error("Gemini call failed:", err);
    return pickFallback();
  }
}

function pickFallback(): string {
  return FALLBACK_QUOTES[Math.floor(Math.random() * FALLBACK_QUOTES.length)]!;
}
