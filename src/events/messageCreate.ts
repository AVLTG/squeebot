import type { Client, Message } from "discord.js";
import { Events } from "discord.js";
import { config, isChannelAllowed } from "../config.js";
import { logger } from "../utils/logger.js";
import { generateReply } from "../services/squeeBrain.js";

export function registerMessageCreate(client: Client): void {
  client.on(Events.MessageCreate, async (message: Message) => {
    try {
      await handleMessage(client, message);
    } catch (err) {
      logger.error("Error handling message:", err);
    }
  });
}

async function handleMessage(client: Client, message: Message): Promise<void> {
  // Ignore bots (including ourselves) to prevent loops
  if (message.author.bot) return;

  // Ignore messages in channels not on the allowlist (if set)
  if (!isChannelAllowed(message.channelId)) return;

  // Only respond when the bot is mentioned
  const botUser = client.user;
  if (!botUser) return;
  if (!message.mentions.has(botUser.id)) return;

  logger.debug(
    `Mention from ${message.author.tag} in #${
      "name" in message.channel ? message.channel.name : message.channelId
    }: ${message.content}`
  );

  // Strip the bot mention(s) from the message to get the user's actual question
  const userMessage = message.content
    .replace(new RegExp(`<@!?${botUser.id}>`, "g"), "")
    .trim();

  // If they just mentioned with no content, treat it as a greeting
  const promptText = userMessage || "(user mentioned Squee with no message)";

  // Show typing indicator while we think (not supported on all channel types)
  if ("sendTyping" in message.channel) {
    await message.channel.sendTyping().catch(() => {
      // Not fatal — some channel types refuse typing indicators
    });
  }

  const reply = await generateReply({
    userId: message.author.id,
    userName: message.author.displayName ?? message.author.username,
    userMessage: promptText,
  });

  // Reply using Discord's reply feature so it's clear what we're responding to
  await message.reply({
    content: reply,
    allowedMentions: { repliedUser: false }, // don't ping them with the reply
  });
}
