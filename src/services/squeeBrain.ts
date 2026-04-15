/**
 * The Squee Brain — generates in-character responses.
 *
 * Phase 3: calls Gemini Flash with structured JSON output containing both
 * the reply and an updated per-user "goblin note". Reads existing memory
 * before each call, saves the updated memory after.
 *
 * Errors from Gemini are surfaced as in-character fallback quotes that hint
 * at what went wrong (rate limit, server down, bad request, network, etc.)
 * so users know Squee is temporarily broken without seeing stack traces.
 *
 * Phase 4 will add RAG-retrieved voice lines.
 */

import { Type } from "@google/genai";
import { getGeminiClient, GEMINI_MODEL } from "./gemini.js";
import { SQUEE_SYSTEM_PROMPT } from "./prompt.js";
import { getMemory, setMemory } from "./memory.js";
import { logger } from "../utils/logger.js";

export interface BrainContext {
  userId: string;
  userName: string;
  userMessage: string;
}

// JSON schema Gemini is instructed to follow. Both fields required so the
// model doesn't skip one (we can discard empty memory downstream).
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

// In-character fallback quotes, keyed by error category.
// Each category has multiple options so repeat failures don't show the same line.
const FALLBACKS = {
  // 429 — Gemini rate-limit. Squee got yelled at too many times in a row.
  rateLimit: [
    "AIEEEE! Squee's head is full! Too many questions at once! Give Squee a minute ta breathe!",
    "Squee's tongue tired from yappin'! Leave Squee alone for a bit, yeah? Go talk to Karn!",
    "Too many humans pokin' Squee! Squee need a nap! Come back when you ain't all shoutin'!",
    "Whoa whoa whoa — slow down! Squee only got one mouth an' you people gots a gabillion questions!",
  ],
  // 5xx — Gemini/Google server issue. Squee's "talkin' rock" is broken.
  serverDown: [
    "Squee's talkin' rock is busted! Stupid googool magic not workin'! Squee gotta find a new one!",
    "De big magic box dat makes Squee smart is broke! Squee back ta bein' regular dumb goblin! Try again later!",
    "AIEEEE! Sumpthin' went BOOM inside Squee's head! Squee gotta sit down! Ask Squee again in a bit!",
    "De magic wires in Squee's brain come unplugged! Squee tryin' ta find where dey go! Poke Squee again soon!",
  ],
  // 400 — bad request. Rare, usually means our prompt is malformed.
  badRequest: [
    "Wuh? Dat don't make no sense ta Squee! Use normal goblin words!",
    "Squee's brain can't chew on dat! Too many weird symbols! Try sumpthin' simpler!",
    "Dat question made Squee's eyes cross! What language wuz dat even?",
  ],
  // 401 / 403 — auth issue. Shouldn't happen in normal operation (means the API key is bad).
  authBad: [
    "Squee's magic pass-code got rejected! De googool guards won't let Squee in de library! Tell a smart person!",
    "Somebody stole Squee's permission papers! Squee can't get to de thinkin'-place! Get Gerrard!",
  ],
  // Network / unknown. Can't reach the API at all.
  network: [
    "Hello? HELLOOOO?! Squee shoutin' but nuthin' comin' back! De invisible web ting must be busted!",
    "Squee tried to send his thoughts across de big nothin' but dey got lost! Try again!",
    "Stupid talkin' rock not ringin' out! Dis thing broken? Did Gerrard forget ta pay de bill?",
  ],
  // Empty / unparseable response from Gemini.
  emptyResponse: [
    "Squee opened his mouth but nuthin' came out! Dat's weird, right?",
    "Squee had a thought, an' den he forgot it. Too fast! Squee's brain is like a bug — zippy!",
    "Words got stuck in Squee's throat! Squee gotta cough dem up! Try pokin' him again!",
  ],
} as const;

type FallbackKey = keyof typeof FALLBACKS;

export async function generateReply(ctx: BrainContext): Promise<string> {
  try {
    // Load any existing note Squee has scribbled about this user
    const existing = getMemory(ctx.userId);
    const memoryBlock = existing
      ? `Squee's existing goblin note on ${ctx.userName}: "${existing.note}"`
      : `No existing note — this is the first time Squee talks to ${ctx.userName}.`;

    const userTurn = [
      memoryBlock,
      "",
      `${ctx.userName} says to Squee: ${ctx.userMessage}`,
    ].join("\n");

    const client = getGeminiClient();
    const response = await client.models.generateContent({
      model: GEMINI_MODEL,
      contents: [{ role: "user", parts: [{ text: userTurn }] }],
      config: {
        systemInstruction: SQUEE_SYSTEM_PROMPT,
        temperature: 1.0,
        maxOutputTokens: 400,
        thinkingConfig: { thinkingBudget: 0 },
        responseMimeType: "application/json",
        responseSchema: RESPONSE_SCHEMA,
      },
    });

    const raw = response.text?.trim();
    if (!raw) {
      logger.warn("Gemini returned empty response, using fallback");
      return pickFallback("emptyResponse");
    }

    // Try to parse the structured response. If it fails, fall back to
    // using the raw text as the reply and skip the memory update (option B).
    const parsed = tryParseResponse(raw);
    if (!parsed) {
      logger.warn(
        `Could not parse JSON response, using raw text as reply. Raw: ${raw.slice(0, 200)}`
      );
      return raw;
    }

    if (parsed.memory) {
      setMemory(ctx.userId, ctx.userName, parsed.memory);
    }

    logger.debug(
      `Reply (${parsed.reply.length} chars): ${parsed.reply} | memory: "${parsed.memory}"`
    );
    return parsed.reply;
  } catch (err) {
    const category = classifyError(err);
    logger.error(`Gemini call failed (${category}):`, err);
    return pickFallback(category);
  }
}

/**
 * Classify an error into one of our fallback categories so the user sees
 * an in-character hint about what went wrong.
 */
function classifyError(err: unknown): FallbackKey {
  if (typeof err !== "object" || err === null) return "network";

  const anyErr = err as { status?: number; code?: number; message?: string };
  const num = anyErr.status ?? anyErr.code;

  // Try to parse a status code out of the error message as a fallback
  let resolved = typeof num === "number" ? num : NaN;
  if (Number.isNaN(resolved)) {
    const match = (anyErr.message ?? "").match(/\b(4\d\d|5\d\d)\b/);
    if (match) resolved = Number(match[1]);
  }

  if (resolved === 429) return "rateLimit";
  if (resolved === 400) return "badRequest";
  if (resolved === 401 || resolved === 403) return "authBad";
  if (resolved >= 500 && resolved < 600) return "serverDown";
  return "network";
}

interface ParsedResponse {
  reply: string;
  memory: string;
}

function tryParseResponse(raw: string): ParsedResponse | null {
  try {
    const obj = JSON.parse(raw);
    if (
      typeof obj === "object" &&
      obj !== null &&
      typeof obj.reply === "string" &&
      obj.reply.trim().length > 0
    ) {
      return {
        reply: obj.reply.trim(),
        memory: typeof obj.memory === "string" ? obj.memory.trim() : "",
      };
    }
    return null;
  } catch {
    return null;
  }
}

function pickFallback(key: FallbackKey): string {
  const options = FALLBACKS[key];
  return options[Math.floor(Math.random() * options.length)]!;
}
