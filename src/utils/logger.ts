import { config } from "../config.js";

function timestamp(): string {
  return new Date().toISOString();
}

export const logger = {
  debug(...args: unknown[]): void {
    if (config.logLevel === "debug") {
      console.log(`[${timestamp()}] [DEBUG]`, ...args);
    }
  },
  info(...args: unknown[]): void {
    console.log(`[${timestamp()}] [INFO]`, ...args);
  },
  warn(...args: unknown[]): void {
    console.warn(`[${timestamp()}] [WARN]`, ...args);
  },
  error(...args: unknown[]): void {
    console.error(`[${timestamp()}] [ERROR]`, ...args);
  },
};
