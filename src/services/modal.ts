/**
 * Modal vLLM provider — calls our fine-tuned Llama 3.1 8B AWQ model served
 * on Modal (see fine-tune/serve.py).
 *
 * The fine-tune was trained on plain Squee dialogue, NOT JSON output. To
 * preserve the goblin-notes memory feature, we run two calls in parallel:
 *   - vLLM (Modal) for the actual in-character reply
 *   - Groq for the memory note update
 * Total latency stays ~1 sec since both run via Promise.all.
 *
 * Returns the same {reply, memory} JSON shape Groq/Gemini do, so squeeBrain.ts
 * doesn't need to know which provider produced what.
 */

import { config } from "../config.js";
import { callGroq } from "./groq.js";

const VLLM_MODEL_NAME = "squee";

export interface ModalCallOpts {
  systemPrompt: string;
  userTurn: string;
  temperature: number;
  maxOutputTokens: number;
}

export async function callModal(opts: ModalCallOpts): Promise<string | null> {
  if (!config.modalVllmUrl || !config.modalVllmKey) {
    throw new Error(
      "MODAL_VLLM_URL and SQUEE_VLLM_KEY must be set to use the modal provider."
    );
  }

  const [reply, memory] = await Promise.all([
    callVllmReply(opts),
    extractMemoryFromGroq(opts),
  ]);

  if (!reply) return null;

  return JSON.stringify({ reply, memory });
}

async function callVllmReply(opts: ModalCallOpts): Promise<string | null> {
  // Strip the JSON/memory section — the fine-tune wasn't trained on JSON output.
  const cleanedSystem = stripMemoryInstructions(opts.systemPrompt);

  const response = await fetch(`${config.modalVllmUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${config.modalVllmKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: VLLM_MODEL_NAME,
      messages: [
        { role: "system", content: cleanedSystem },
        { role: "user", content: opts.userTurn },
      ],
      temperature: opts.temperature,
      max_tokens: opts.maxOutputTokens,
    }),
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    const err = new Error(
      `Modal vLLM returned ${response.status}: ${body.slice(0, 200)}`
    );
    (err as { status?: number }).status = response.status;
    throw err;
  }

  const data = (await response.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  return data.choices?.[0]?.message?.content?.trim() ?? null;
}

/**
 * Run a parallel Groq call and pull just the memory field out of its JSON
 * response. The reply Groq generates is discarded — vLLM is the source of
 * truth for that. Free at our volume; Groq's median latency is ~300ms which
 * fits inside vLLM's window so total time-to-response doesn't grow.
 *
 * Failure is silent: returns "" and squeeBrain.ts skips the memory update.
 */
async function extractMemoryFromGroq(opts: ModalCallOpts): Promise<string> {
  try {
    const raw = await callGroq(opts);
    if (!raw) return "";
    const parsed = JSON.parse(raw) as { memory?: unknown };
    return typeof parsed.memory === "string" ? parsed.memory.trim() : "";
  } catch {
    return "";
  }
}

function stripMemoryInstructions(systemPrompt: string): string {
  const idx = systemPrompt.indexOf("# Your memory (goblin notes)");
  if (idx === -1) return systemPrompt.trim();
  return systemPrompt.substring(0, idx).trim();
}
