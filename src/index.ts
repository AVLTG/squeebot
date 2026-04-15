import { Client, GatewayIntentBits, Partials } from "discord.js";
import { config } from "./config.js";
import { logger } from "./utils/logger.js";
import { registerMessageCreate } from "./events/messageCreate.js";
import { closeMemory } from "./services/memory.js";

async function main(): Promise<void> {
  const client = new Client({
    intents: [
      GatewayIntentBits.Guilds,
      GatewayIntentBits.GuildMessages,
      GatewayIntentBits.MessageContent, // REQUIRED to read message.content
    ],
    partials: [Partials.Channel],
  });

  // Register event handlers
  registerMessageCreate(client);

  client.once("clientReady", (readyClient) => {
    logger.info(`Squee is alive! Logged in as ${readyClient.user.tag}`);
    logger.info(
      `In ${readyClient.guilds.cache.size} guild(s). Listening for mentions.`
    );
  });

  client.on("error", (err) => {
    logger.error("Discord client error:", err);
  });

  // Graceful shutdown — Discord wants a clean disconnect and SQLite needs
  // to checkpoint its WAL file to avoid data loss.
  const shutdown = async (signal: string) => {
    logger.info(`Received ${signal}, shutting down...`);
    await client.destroy();
    closeMemory();
    process.exit(0);
  };
  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));

  await client.login(config.discordToken);
}

main().catch((err) => {
  logger.error("Fatal error in main:", err);
  process.exit(1);
});
