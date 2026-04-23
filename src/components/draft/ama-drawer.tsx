'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { cn } from '@/lib/utils'
import type { EnrichedDraftContext } from '@/lib/ama/context'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface AMADrawerProps {
  open: boolean
  onClose: () => void
  draftContext: EnrichedDraftContext | null
}

const STARTER_PROMPTS = (topRec: string | undefined) => [
  topRec ? `Why is ${topRec} the top recommendation right now?` : 'Why is the top hero recommended here?',
  'What is our team composition trying to do?',
  'What are the biggest threats in the enemy draft?',
  'Is our ban strategy sound given the enemy team?',
]

interface ExpandPopup {
  x: number
  y: number
  text: string
}

function AssistantMessage({
  content,
  isStreaming,
  onExpand,
}: {
  content: string
  isStreaming: boolean
  onExpand: (text: string) => void
}) {
  const [popup, setPopup] = useState<ExpandPopup | null>(null)
  const ref = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)

  const handleMouseUp = () => {
    const sel = window.getSelection()
    if (!sel || sel.isCollapsed || !ref.current) return
    const selected = sel.toString().trim()
    if (selected.length < 5) return
    const range = sel.getRangeAt(0)
    if (!ref.current.contains(range.commonAncestorContainer)) return
    const rect = range.getBoundingClientRect()
    const containerRect = ref.current.getBoundingClientRect()
    setPopup({
      x: rect.left - containerRect.left + rect.width / 2,
      y: rect.top - containerRect.top - 8,
      text: selected,
    })
  }

  const handleExpand = () => {
    if (!popup) return
    const text = popup.text
    setPopup(null)
    window.getSelection()?.removeAllRanges()
    onExpand(text)
  }

  // Dismiss on outside click — but not if clicking the expand button itself
  useEffect(() => {
    const dismiss = (e: MouseEvent) => {
      if (buttonRef.current?.contains(e.target as Node)) return
      setPopup(null)
    }
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
            <p
              key={i}
              className="text-[#c8d0e0]"
            >
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
          ref={buttonRef}
          onClick={handleExpand}
          className="absolute z-10 px-2.5 py-1 rounded text-[11px] font-medium bg-[#1a1f3a] border border-[#6b8dd4]/60 text-[#6b8dd4] shadow-lg hover:bg-[#6b8dd4]/20 transition-colors whitespace-nowrap"
          style={{
            left: popup.x,
            top: popup.y,
            transform: 'translate(-50%, -100%)',
          }}
        >
          Expand on this ↗
        </button>
      )}
    </div>
  )
}

export function AMADrawer({ open, onClose, draftContext }: AMADrawerProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [contextUpdated, setContextUpdated] = useState(false)
  const [disclaimerVisible, setDisclaimerVisible] = useState(false)
  const [disclaimerHighlighted, setDisclaimerHighlighted] = useState(false)
  const [headerHeight, setHeaderHeight] = useState(56)
  const [isMobile, setIsMobile] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const prevContextRef = useRef<string | null>(null)

  useEffect(() => {
    const el = document.getElementById('app-header')
    if (!el) return
    const observer = new ResizeObserver(() => setHeaderHeight(el.getBoundingClientRect().height))
    observer.observe(el)
    setHeaderHeight(el.getBoundingClientRect().height)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 640)
    check()
    window.addEventListener('resize', check)
    return () => window.removeEventListener('resize', check)
  }, [])

  useEffect(() => {
    if (!open || messages.length > 0) return
    setDisclaimerVisible(true)
    setDisclaimerHighlighted(true)
    const t1 = setTimeout(() => setDisclaimerHighlighted(false), 2000)
    const t2 = setTimeout(() => setDisclaimerVisible(false), 10000)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [open, messages.length])

  useEffect(() => {
    const newBlock = draftContext?.textBlock ?? null
    if (prevContextRef.current !== null && prevContextRef.current !== newBlock && messages.length > 0) {
      setContextUpdated(true)
      const t = setTimeout(() => setContextUpdated(false), 2500)
      return () => clearTimeout(t)
    }
    prevContextRef.current = newBlock
  }, [draftContext, messages.length])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, streaming])

  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 150)
  }, [open])

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
          draftContext: draftContext?.textBlock,
        }),
      })

      if (!res.ok || !res.body) {
        setMessages(prev => {
          const u = [...prev]
          u[u.length - 1] = { role: 'assistant', content: 'Something went wrong. Please try again.' }
          return u
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
          const u = [...prev]
          u[u.length - 1] = { role: 'assistant', content: u[u.length - 1].content + chunk }
          return u
        })
      }
    } catch {
      setMessages(prev => {
        const u = [...prev]
        u[u.length - 1] = { role: 'assistant', content: 'Network error. Please try again.' }
        return u
      })
    } finally {
      setStreaming(false)
    }
  }, [messages, streaming, draftContext])

  const handleExpand = useCallback((selectedText: string) => {
    sendMessage(`Expand on this: "${selectedText}"`)
  }, [sendMessage])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  const card = draftContext?.card

  return (
    /* No backdrop — drawer pushes content via margin on the parent */
    <div
      className={cn(
        'fixed z-40 flex flex-col shadow-2xl',
        'transition-transform duration-300 ease-in-out',
        isMobile
          ? cn(
              'bottom-0 left-0 right-0 rounded-t-2xl border-t border-[#3a4050]',
              open ? 'translate-y-0' : 'translate-y-full',
            )
          : cn(
              'right-0 w-[420px] border-l border-[#3a4050]',
              open ? 'translate-x-0' : 'translate-x-full',
            ),
      )}
      style={isMobile
        ? { background: 'radial-gradient(ellipse at top, #1a1f3a 0%, #0a0d1f 100%)', height: '75vh' }
        : { background: 'radial-gradient(ellipse at top, #1a1f3a 0%, #0a0d1f 100%)', top: headerHeight, height: `calc(100vh - ${headerHeight}px)` }
      }
    >
      {/* Mobile drag handle */}
      {isMobile && (
        <div className="flex justify-center pt-2.5 pb-1 shrink-0">
          <div className="w-10 h-1 rounded-full bg-[#3a4050]" />
        </div>
      )}
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#3a4050] shrink-0">
        <div className="flex items-center gap-3">
          <div>
            <h2 className="text-sm font-semibold text-white tracking-wide">COACH</h2>
            <p className="text-xs text-[#8b9bc8]">
              {card?.map
                ? `${card.map} · ${card.tier} · Step ${card.step}/${card.totalSteps}${card.stepType ? ` · ${card.isOurTurn ? 'Your' : 'Enemy'} ${card.stepType}` : ''}`
                : 'Ask me anything'}
            </p>
            <p
              className="text-[10px] italic mt-0.5 transition-[opacity,color] duration-1000"
              style={{
                opacity: disclaimerVisible ? 1 : 0,
                color: disclaimerHighlighted ? '#facc15' : 'rgba(139,155,200,0.6)',
              }}
            >
              Disclaimer: any explanations of statistics are conjecture.
            </p>
          </div>
          {contextUpdated && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#6b8dd4]/20 text-[#6b8dd4] border border-[#6b8dd4]/40 animate-pulse">
              Context updated
            </span>
          )}
        </div>
        <button onClick={onClose} className="text-[#8b9bc8] hover:text-white transition-colors p-1 rounded">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Top recs quick-ask */}
      {card && card.topRecs.length > 0 && messages.length === 0 && (
        <div className="px-4 py-3 border-b border-[#3a4050] shrink-0">
          <p className="text-[10px] text-[#8b9bc8] uppercase tracking-widest mb-2">Current top picks</p>
          <div className="flex flex-wrap gap-1.5">
            {card.topRecs.slice(0, 4).map((rec) => (
              <button
                key={rec.hero}
                onClick={() => sendMessage(`Why is ${rec.hero} recommended here?`)}
                className="flex items-center gap-1.5 px-2 py-1 rounded border border-[#3a4050] bg-[#0a0d1f]/60 hover:border-[#6b8dd4]/60 hover:bg-[#6b8dd4]/10 transition-colors text-xs"
              >
                <span className="text-[#d6dbe0]">{rec.hero}</span>
                <span className={cn(
                  'font-medium',
                  rec.netDelta >= 3 ? 'text-[#6fd46f]' : rec.netDelta >= 0 ? 'text-[#d4b85a]' : 'text-[#d46b6b]'
                )}>
                  {rec.netDelta >= 0 ? '+' : ''}{rec.netDelta.toFixed(1)}%
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4 min-h-0">
        {messages.length === 0 ? (
          <div className="space-y-3">
            <p className="text-xs text-[#8b9bc8] text-center pt-2">
              Ask me anything about the draft, hero picks, or strategy.
              {!card?.map && ' Load a draft first for context-aware answers.'}
            </p>
            <div className="space-y-2">
              {STARTER_PROMPTS(card?.topRecs[0]?.hero).map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => sendMessage(prompt)}
                  className="w-full text-left px-3 py-2.5 rounded border border-[#3a4050] bg-[#0a0d1f]/40 hover:border-[#6b8dd4]/50 hover:bg-[#6b8dd4]/8 transition-colors text-sm text-[#8b9bc8] hover:text-[#d6dbe0]"
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
                  'max-w-[90%] rounded-lg px-3 py-2.5',
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

      {/* Input */}
      <div className="px-4 py-3 border-t border-[#3a4050] shrink-0">
        {messages.length > 0 && !streaming && (
          <button onClick={() => setMessages([])} className="text-[10px] text-[#8b9bc8] hover:text-[#d6dbe0] mb-2 transition-colors">
            ↺ New conversation
          </button>
        )}
        <div className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask the coach…"
            rows={2}
            disabled={streaming}
            className={cn(
              'flex-1 resize-none rounded-lg px-3 py-2.5 text-sm',
              'bg-[#0a0d1f]/80 border border-[#3a4050] text-[#d6dbe0] placeholder-[#8b9bc8]/60',
              'focus:outline-none focus:border-[#6b8dd4]/60 transition-colors disabled:opacity-50',
            )}
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={!input.trim() || streaming}
            className={cn(
              'px-3 py-2.5 rounded-lg text-sm font-medium transition-colors border shrink-0',
              input.trim() && !streaming
                ? 'bg-[#6b8dd4]/20 border-[#6b8dd4]/60 text-[#6b8dd4] hover:bg-[#6b8dd4]/30'
                : 'border-[#3a4050] text-[#8b9bc8]/50 cursor-not-allowed'
            )}
          >
            Send
          </button>
        </div>
        <p className="text-[10px] text-[#8b9bc8]/50 mt-1.5">Enter to send · Shift+Enter for newline · Select text to expand</p>
      </div>
    </div>
  )
}
