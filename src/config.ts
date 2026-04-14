import "dotenv/config";

function required(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

function optional(name: string, fallback = ""): string {
  return process.env[name] ?? fallback;
}

export const config = {
  discordToken: required("DISCORD_TOKEN"),
  discordClientId: required("DISCORD_CLIENT_ID"),
  geminiApiKey: optional("GEMINI_API_KEY"),
  allowedChannelIds: optional("ALLOWED_CHANNEL_IDS")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean),
  logLevel: optional("LOG_LEVEL", "info") as "debug" | "info",
} as const;

export function isChannelAllowed(channelId: string): boolean {
  if (config.allowedChannelIds.length === 0) return true;
  return config.allowedChannelIds.includes(channelId);
}
