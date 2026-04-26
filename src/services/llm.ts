/**
 * LLM provider dispatcher.
 *
 * Default behavior (LLM_PROVIDER=cascade): try Groq → Gemini → Modal,
 * advancing to the next only on a 429. Other errors surface immediately
 * (network/auth issues are real bugs, not "try the next provider").
 *
 * Pinning LLM_PROVIDER=groq|gemini|modal disables the cascade for testing.
 *
 * Why this order:
 *   - Groq:   fast, generous free tier, llama-3.3-70b is the strongest model
 *   - Gemini: free fallback (Flash Lite, ~20 RPD)
 *   - Modal:  our 8B fine-tune. Last resort because (a) costs Modal credits
 *             and (b) production prompt format diverges from training format,
 *             so quality is noticeably lower than Groq's 70B baseline.
 */

import { config } from "../config.js";
import { logger } from "../utils/logger.js";
import { callGroq } from "./groq.js";
import { callGemini } from "./gemini.js";
import { callModal } from "./modal.js";

export interface LLMCallOpts {
  systemPrompt: string;
  userTurn: string;
  temperature: number;
  maxOutputTokens: number;
}

const CASCADE = ["groq", "gemini", "modal"] as const;
type Provider = (typeof CASCADE)[number];

function isProviderConfigured(provider: Provider): boolean {
  if (provider === "groq") return Boolean(config.groqApiKey);
  if (provider === "gemini") return Boolean(config.geminiApiKey);
  return Boolean(config.modalVllmUrl && config.modalVllmKey);
}

function callProvider(
  provider: Provider,
  opts: LLMCallOpts
): Promise<string | null> {
  if (provider === "groq") return callGroq(opts);
  if (provider === "gemini") return callGemini(opts);
  return callModal(opts);
}

export async function callLLM(opts: LLMCallOpts): Promise<string | null> {
  if (config.llmProvider !== "cascade") {
    return callProvider(config.llmProvider, opts);
  }

  const configured = CASCADE.filter(isProviderConfigured);
  if (configured.length === 0) {
    throw new Error(
      "No LLM providers configured. Set GROQ_API_KEY, GEMINI_API_KEY, " +
        "or MODAL_VLLM_URL+SQUEE_VLLM_KEY in your .env."
    );
  }

  let lastErr: unknown = null;
  for (let i = 0; i < configured.length; i++) {
    const provider = configured[i]!;
    try {
      return await callProvider(provider, opts);
    } catch (err) {
      lastErr = err;
      const status = (err as { status?: number })?.status;

      // Only fall through on rate-limit. Other errors (auth, network, 5xx)
      // are real problems we want to surface immediately.
      if (status !== 429) throw err;

      const next = configured[i + 1];
      if (next) {
        logger.warn(
          `Provider "${provider}" hit 429. Falling back to "${next}".`
        );
      } else {
        logger.error(
          `All providers exhausted (last 429: "${provider}"). Surfacing error.`
        );
      }
    }
  }

  throw lastErr ?? new Error("LLM cascade exhausted with no error captured.");
}
