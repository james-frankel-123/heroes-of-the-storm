const LOG_LEVELS = { debug: 0, info: 1, warn: 2, error: 3 } as const

type LogLevel = keyof typeof LOG_LEVELS

class Logger {
  private level: number

  constructor(level: LogLevel = 'info') {
    this.level = LOG_LEVELS[level]
  }

  private timestamp(): string {
    return new Date().toISOString()
  }

  debug(msg: string) {
    if (this.level <= 0) console.log(`[${this.timestamp()}] [DEBUG] ${msg}`)
  }

  info(msg: string) {
    if (this.level <= 1) console.log(`[${this.timestamp()}] [INFO]  ${msg}`)
  }

  warn(msg: string) {
    if (this.level <= 2) console.warn(`[${this.timestamp()}] [WARN]  ${msg}`)
  }

  error(msg: string, err?: unknown) {
    if (this.level <= 3) {
      console.error(`[${this.timestamp()}] [ERROR] ${msg}`)
      if (err instanceof Error) console.error(err.stack)
      else if (err) console.error(err)
    }
  }
}

const level = (process.env.LOG_LEVEL as LogLevel) || 'info'
export const log = new Logger(level)
