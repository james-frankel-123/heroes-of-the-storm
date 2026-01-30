import OpenAI from 'openai'

// Warn during build if missing, but don't throw (allows build to succeed)
// Will throw at runtime when API is actually called
if (!process.env.OPENAI_API_KEY) {
  console.warn('Warning: OPENAI_API_KEY environment variable is not set. AI commentary features will not work.')
}

export const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || 'sk-dummy-key-for-build',
})

export const OPENAI_CONFIG = {
  model: process.env.OPENAI_MODEL || 'gpt-4o',
  maxTokens: parseInt(process.env.OPENAI_MAX_TOKENS || '800'),
  temperature: 0.7,
  stream: true,
}
