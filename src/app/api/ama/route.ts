import OpenAI from 'openai'
import { AMA_SYSTEM_PROMPT } from '@/lib/ama/system-prompt'

export const runtime = 'nodejs'

export async function POST(req: Request) {
  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) {
    return new Response('OPENAI_API_KEY not configured', { status: 500 })
  }

  let messages: Array<{ role: 'user' | 'assistant'; content: string }>
  let draftContext: string | undefined

  try {
    const body = await req.json()
    messages = body.messages
    draftContext = body.draftContext
  } catch {
    return new Response('Invalid request body', { status: 400 })
  }

  if (!Array.isArray(messages) || messages.length === 0) {
    return new Response('messages array required', { status: 400 })
  }

  const systemContent = draftContext
    ? `${AMA_SYSTEM_PROMPT}\n\n---\nACTIVE DRAFT CONTEXT (live data from the draft tool):\n${draftContext}`
    : AMA_SYSTEM_PROMPT

  const openai = new OpenAI({ apiKey })

  try {
    const stream = await openai.chat.completions.create({
      model: 'o4-mini',
      stream: true,
      messages: [
        { role: 'developer' as 'system', content: systemContent },
        ...messages,
      ],
      max_completion_tokens: 4000,
    })

    const readable = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder()
        try {
          for await (const chunk of stream) {
            const text = chunk.choices[0]?.delta?.content ?? ''
            if (text) {
              controller.enqueue(encoder.encode(text))
            }
          }
        } finally {
          controller.close()
        }
      },
    })

    return new Response(readable, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'X-Content-Type-Options': 'nosniff',
        'Cache-Control': 'no-cache',
      },
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    console.error('OpenAI API error:', message)
    return new Response(`OpenAI error: ${message}`, { status: 502 })
  }
}
