'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { cn } from '@/lib/utils'
import type { DraftContextCard } from '@/lib/ama/context'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface ExpandPopup { x: number; y: number; text: string }

function AssistantMessage({ content, isStreaming, onExpand }: {
  content: string; isStreaming: boolean; onExpand: (text: string) => void
}) {
  const [popup, setPopup] = useState<ExpandPopup | null>(null)
  const ref = useRef<HTMLDivElement>(null)

  const handleMouseUp = () => {
    const sel = window.getSelection()
    if (!sel || sel.isCollapsed || !ref.current) return
    const selected = sel.toString().trim()
    if (selected.length < 5) return
    const range = sel.getRangeAt(0)
    if (!ref.current.contains(range.commonAncestorContainer)) return
    const rect = range.getBoundingClientRect()
    const containerRect = ref.current.getBoundingClientRect()
    setPopup({ x: rect.left - containerRect.left + rect.width / 2, y: rect.top - containerRect.top - 8, text: selected })
  }

  const handleExpand = () => {
    if (!popup) return
    onExpand(popup.text)
    setPopup(null)
    window.getSelection()?.removeAllRanges()
  }

  useEffect(() => {
    const dismiss = () => setPopup(null)
    if (popup) document.addEventListener('mousedown', dismiss)
    return () => document.removeEventListener('mousedown', dismiss)
  }, [popup])

  const lines = content.split('\n')

  return (
    <div ref={ref} className="relative" onMouseUp={handleMouseUp}>
      <div className="leading-relaxed text-sm space-y-1">
        {lines.map((line, i) => {
          if (line.trim().startsWith('(Conjecture')) return null
          const isLast = i === lines.length - 1
          return line.trim() === '' ? null : (
            <p key={i} className="text-[#c8d0e0]">
              {line}
              {isLast && isStreaming && (
                <span className="inline-block w-1.5 h-4 ml-0.5 bg-[#6b8dd4] animate-pulse rounded-sm align-middle" />
              )}
            </p>
          )
        })}
        {isStreaming && lines[lines.length - 1]?.trim() === '' && (
          <span className="inline-block w-1.5 h-4 bg-[#6b8dd4] animate-pulse rounded-sm" />
        )}
      </div>
      {popup && (
        <button
          onMouseDown={(e) => e.stopPropagation()}
          onClick={handleExpand}
          className="absolute z-10 px-2.5 py-1 rounded text-[11px] font-medium bg-[#1a1f3a] border border-[#6b8dd4]/60 text-[#6b8dd4] shadow-lg hover:bg-[#6b8dd4]/20 transition-colors whitespace-nowrap"
          style={{ left: popup.x, top: popup.y, transform: 'translate(-50%, -100%)' }}
        >
          Expand on this ↗
        </button>
      )}
    </div>
  )
}

interface StoredContext {
  textBlock: string
  card: DraftContextCard
}

const STARTER_PROMPTS_GENERIC = [
  'What heroes are generally strong on Cursed Hollow?',
  'How do I build a dive composition in HotS?',
  'What should I prioritize banning at mid tier?',
  'How does MAWP affect hero recommendations?',
]

function starterPrompts(card: DraftContextCard | null): string[] {
  if (!card || !card.map) return STARTER_PROMPTS_GENERIC
  const topRec = card.topRecs[0]?.hero
  return [
    topRec ? `Why is ${topRec} the top recommendation here?` : 'Why is the top hero recommended?',
    'What is our team composition trying to accomplish?',
    'What are the biggest threats in the enemy draft?',
    'Is our current ban strategy sound?',
  ]
}

export function AMAClient() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [storedContext, setStoredContext] = useState<StoredContext | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Load context from sessionStorage on mount
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem('ama-draft-context')
      if (raw) {
        const parsed = JSON.parse(raw) as StoredContext
        setStoredContext(parsed)
      }
    } catch {
      // sessionStorage unavailable or invalid JSON
    }
  }, [])

  const refreshContext = () => {
    try {
      const raw = sessionStorage.getItem('ama-draft-context')
      if (raw) {
        setStoredContext(JSON.parse(raw) as StoredContext)
      }
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, streaming])

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const sendMessage = useCallback(async (text: string) => {
    const userMessage = text.trim()
    if (!userMessage || streaming) return

    const newMessages: Message[] = [...messages, { role: 'user', content: userMessage }]
    setMessages(newMessages)
    setInput('')
    setStreaming(true)
    setMessages(prev => [...prev, { role: 'assistant', content: '' }])

    try {
      const res = await fetch('/api/ama', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: newMessages,
          draftContext: storedContext?.textBlock,
        }),
      })

      if (!res.ok || !res.body) {
        setMessages(prev => {
          const updated = [...prev]
          updated[updated.length - 1] = {
            role: 'assistant',
            content: 'Something went wrong. Please try again.',
          }
          return updated
        })
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        setMessages(prev => {
          const updated = [...prev]
          updated[updated.length - 1] = {
            role: 'assistant',
            content: updated[updated.length - 1].content + chunk,
          }
          return updated
        })
      }
    } catch {
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          role: 'assistant',
          content: 'Network error. Please try again.',
        }
        return updated
      })
    } finally {
      setStreaming(false)
    }
  }, [messages, streaming, storedContext])

  const handleExpand = useCallback((selectedText: string) => {
    sendMessage(`Expand on this: "${selectedText}"`)
  }, [sendMessage])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  const card = storedContext?.card ?? null

  return (
    <div className="max-w-5xl mx-auto">
      {/* Page title */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white">Coach</h1>
        <p className="text-[#8b9bc8] mt-1 text-sm">Ask me anything.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-6 items-start">
        {/* Main chat panel */}
        <div
          className="rounded-lg flex flex-col overflow-hidden border border-[#3a4050]"
          style={{
            background: 'radial-gradient(ellipse at top, #1a1f3a 0%, #0a0d1f 100%)',
            minHeight: '600px',
          }}
        >
          {/* Messages */}
          <div
            ref={scrollRef}
            className="flex-1 overflow-y-auto px-5 py-5 space-y-4"
            style={{ maxHeight: '520px' }}
          >
            {messages.length === 0 ? (
              <div className="space-y-4 pt-2">
                <p className="text-sm text-[#8b9bc8] text-center">
                  {card?.map
                    ? `Draft loaded: ${card.map} · ${card.tier} · Step ${card.step}/${card.totalSteps}`
                    : 'No draft context loaded. Open the Draft tool and click "Ask the Coach" to load your current draft, or ask a general HotS question below.'}
                </p>
                <div className="space-y-2">
                  {starterPrompts(card).map((prompt) => (
                    <button
                      key={prompt}
                      onClick={() => sendMessage(prompt)}
                      className="w-full text-left px-4 py-3 rounded-lg border border-[#3a4050] bg-[#0a0d1f]/40 hover:border-[#6b8dd4]/50 hover:bg-[#6b8dd4]/8 transition-colors text-sm text-[#8b9bc8] hover:text-[#d6dbe0]"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg, i) => (
                <div key={i} className={cn('flex', msg.role === 'user' ? 'justify-end' : 'justify-start')}>
                  <div
                    className={cn(
                      'max-w-[88%] rounded-lg px-4 py-3',
                      msg.role === 'user'
                        ? 'bg-[#6b8dd4]/20 border border-[#6b8dd4]/40 text-[#d6dbe0] text-sm'
                        : 'bg-[#0a0d1f]/60 border border-[#3a4050]'
                    )}
                  >
                    {msg.role === 'assistant' ? (
                      msg.content ? (
                        <AssistantMessage
                          content={msg.content}
                          isStreaming={i === messages.length - 1 && streaming}
                          onExpand={handleExpand}
                        />
                      ) : (
                        <div className="flex items-center gap-1.5 text-[#8b9bc8] py-0.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-[#6b8dd4] animate-bounce" style={{ animationDelay: '0ms' }} />
                          <span className="w-1.5 h-1.5 rounded-full bg-[#6b8dd4] animate-bounce" style={{ animationDelay: '150ms' }} />
                          <span className="w-1.5 h-1.5 rounded-full bg-[#6b8dd4] animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                      )
                    ) : (
                      <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Input area */}
          <div className="px-5 py-4 border-t border-[#3a4050] shrink-0">
            {messages.length > 0 && !streaming && (
              <button
                onClick={() => setMessages([])}
                className="text-[10px] text-[#8b9bc8] hover:text-[#d6dbe0] mb-2.5 transition-colors block"
              >
                ↺ New conversation
              </button>
            )}
            <div className="flex gap-3 items-end">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={card?.map
                  ? `Ask the coach about this ${card.map} draft…`
                  : 'Ask the coach anything about HotS drafting…'
                }
                rows={3}
                disabled={streaming}
                className={cn(
                  'flex-1 resize-none rounded-lg px-4 py-3 text-sm',
                  'bg-[#0a0d1f]/80 border border-[#3a4050] text-[#d6dbe0] placeholder-[#8b9bc8]/60',
                  'focus:outline-none focus:border-[#6b8dd4]/60 transition-colors',
                  'disabled:opacity-50',
                )}
              />
              <button
                onClick={() => sendMessage(input)}
                disabled={!input.trim() || streaming}
                className={cn(
                  'px-4 py-3 rounded-lg text-sm font-medium transition-colors border shrink-0',
                  input.trim() && !streaming
                    ? 'bg-[#6b8dd4]/20 border-[#6b8dd4]/60 text-[#6b8dd4] hover:bg-[#6b8dd4]/30'
                    : 'border-[#3a4050] text-[#8b9bc8]/50 cursor-not-allowed'
                )}
              >
                Send
              </button>
            </div>
            <p className="text-[10px] text-[#8b9bc8]/50 mt-1.5">
              Enter to send · Shift+Enter for newline · Select text to expand
            </p>
          </div>
        </div>

        {/* Right sidebar — draft context card */}
        <div className="space-y-4">
          {card ? (
            <div
              className="rounded-lg border border-[#3a4050] p-4 space-y-3"
              style={{ background: 'rgba(10, 13, 31, 0.8)' }}
            >
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold text-[#d6dbe0] tracking-widest uppercase">Draft Context</h3>
                <button
                  onClick={refreshContext}
                  className="text-[10px] text-[#8b9bc8] hover:text-[#d6dbe0] transition-colors"
                  title="Refresh context from draft tool"
                >
                  ↻ Refresh
                </button>
              </div>

              <div className="space-y-1 text-xs text-[#8b9bc8]">
                <div className="flex justify-between">
                  <span>Map</span>
                  <span className="text-[#d6dbe0]">{card.map ?? '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Tier</span>
                  <span className="text-[#d6dbe0] capitalize">{card.tier}</span>
                </div>
                <div className="flex justify-between">
                  <span>Our team</span>
                  <span className="text-[#d6dbe0]">Team {card.ourTeam}</span>
                </div>
                <div className="flex justify-between">
                  <span>Step</span>
                  <span className="text-[#d6dbe0]">{card.step}/{card.totalSteps}</span>
                </div>
                {card.winPct !== null && (
                  <div className="flex justify-between">
                    <span>Win est.</span>
                    <span className={cn(
                      'font-medium',
                      card.winPct >= 55 ? 'text-[#6fd46f]' :
                      card.winPct >= 48 ? 'text-[#d4b85a]' : 'text-[#d46b6b]'
                    )}>
                      {card.winPct.toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>

              {card.ourPicks.length > 0 && (
                <div>
                  <p className="text-[10px] text-[#8b9bc8] uppercase tracking-widest mb-1.5">Our picks</p>
                  <div className="flex flex-wrap gap-1">
                    {card.ourPicks.map(h => (
                      <span key={h} className="text-xs px-2 py-0.5 rounded bg-[#6b8dd4]/15 border border-[#6b8dd4]/30 text-[#a0b4e8]">{h}</span>
                    ))}
                  </div>
                </div>
              )}

              {card.enemyPicks.length > 0 && (
                <div>
                  <p className="text-[10px] text-[#8b9bc8] uppercase tracking-widest mb-1.5">Enemy picks</p>
                  <div className="flex flex-wrap gap-1">
                    {card.enemyPicks.map(h => (
                      <span key={h} className="text-xs px-2 py-0.5 rounded bg-[#d46b6b]/15 border border-[#d46b6b]/30 text-[#e8a0a0]">{h}</span>
                    ))}
                  </div>
                </div>
              )}

              {card.bans.length > 0 && (
                <div>
                  <p className="text-[10px] text-[#8b9bc8] uppercase tracking-widest mb-1.5">Bans</p>
                  <div className="flex flex-wrap gap-1">
                    {card.bans.map(h => (
                      <span key={h} className="text-xs px-2 py-0.5 rounded bg-[#3a4050]/60 border border-[#3a4050] text-[#8b9bc8] line-through">{h}</span>
                    ))}
                  </div>
                </div>
              )}

              {card.topRecs.length > 0 && (
                <div>
                  <p className="text-[10px] text-[#8b9bc8] uppercase tracking-widest mb-1.5">Top recommendations</p>
                  <div className="space-y-1">
                    {card.topRecs.slice(0, 5).map((rec) => (
                      <button
                        key={rec.hero}
                        onClick={() => sendMessage(`Why is ${rec.hero} recommended here?`)}
                        className="w-full flex items-center justify-between px-2 py-1.5 rounded border border-[#3a4050] bg-[#0a0d1f]/40 hover:border-[#6b8dd4]/50 hover:bg-[#6b8dd4]/8 transition-colors text-xs group"
                      >
                        <span className="text-[#d6dbe0] group-hover:text-white">{rec.rank}. {rec.hero}</span>
                        <span className={cn(
                          'font-medium',
                          rec.netDelta >= 3 ? 'text-[#6fd46f]' :
                          rec.netDelta >= 0 ? 'text-[#d4b85a]' : 'text-[#d46b6b]'
                        )}>
                          {rec.netDelta >= 0 ? '+' : ''}{rec.netDelta.toFixed(1)}%
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div
              className="rounded-lg border border-[#3a4050] p-4 space-y-3"
              style={{ background: 'rgba(10, 13, 31, 0.8)' }}
            >
              <h3 className="text-xs font-semibold text-[#d6dbe0] tracking-widest uppercase">Draft Context</h3>
              <p className="text-xs text-[#8b9bc8] leading-relaxed">
                No draft loaded. Go to the{' '}
                <a href="/draft" className="text-[#6b8dd4] hover:underline">Draft tool</a>
                , set up your picks, and click <strong className="text-[#d6dbe0]">Ask the Coach</strong> to load your current board here.
              </p>
              <p className="text-xs text-[#8b9bc8] leading-relaxed">
                You can still ask general HotS questions without a draft context.
              </p>
              <button
                onClick={refreshContext}
                className="w-full px-3 py-2 rounded border border-[#3a4050] text-xs text-[#8b9bc8] hover:text-[#d6dbe0] hover:border-[#6b8dd4]/50 transition-colors"
              >
                ↻ Check for draft context
              </button>
            </div>
          )}

          <div
            className="rounded-lg border border-[#3a4050] p-4"
            style={{ background: 'rgba(10, 13, 31, 0.8)' }}
          >
            <p className="text-[10px] text-[#8b9bc8]/70 leading-relaxed">
              <strong className="text-[#8b9bc8]">About AMA</strong><br/>
              Responses are conjecture. The draft engine scores heroes numerically and doesn&apos;t explain itself. AMA reconstructs reasoning from data signals + HotS knowledge.
              Always use your own judgment.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
