'use client'

import * as React from 'react'
import { PasswordGate } from '@/components/auth/password-gate'

export default function StatsPage() {
  return (
    <PasswordGate requiredPassword="ronpaul2012" storageKey="protected_pages_auth">
      <div className="flex h-screen">
        {/* Main Content - 70% width */}
        <div className="flex-1 overflow-y-auto pr-6 space-y-8">
          <div>
            <h1 className="text-4xl font-bold tracking-tight glow">Statistics</h1>
            <p className="mt-2 text-muted-foreground">
              Advanced statistical analysis and trends
            </p>
            <p className="mt-1 text-sm text-primary-500">
              ✨ Click any statistic to analyze it with AI
            </p>
          </div>

          {/* Content components will go here */}
          <div className="text-center py-12 text-muted-foreground">
            Building comprehensive statistics dashboard...
          </div>
        </div>

        {/* AI Chat Panel - 30% width */}
        <div className="w-96 border-l border-border bg-card/30 backdrop-blur-sm flex flex-col">
          <div className="p-4 border-b border-border">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <span>🤖</span>
                <span>AI Assistant</span>
              </h3>
              <button className="text-xs text-muted-foreground hover:text-foreground transition-colors">
                Clear
              </button>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Click stats to add context, then ask questions
            </p>
          </div>

          {/* Context cards area */}
          <div className="p-4 border-b border-border min-h-[120px]">
            <p className="text-sm text-muted-foreground text-center py-4">
              Click a statistic to start analyzing
            </p>
          </div>

          {/* Chat input */}
          <div className="p-4 border-b border-border">
            <input
              type="text"
              placeholder="Ask about the stats above..."
              className="w-full px-3 py-2 bg-background border border-input rounded-md text-sm"
              disabled
            />
            <button
              className="mt-2 w-full bg-primary-500 hover:bg-primary-600 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50"
              disabled
            >
              Send
            </button>
          </div>

          {/* Suggested questions */}
          <div className="p-4 border-b border-border">
            <p className="text-xs font-medium text-muted-foreground mb-2">Suggested:</p>
            <div className="space-y-1">
              <button className="w-full text-left text-xs px-2 py-1 rounded hover:bg-muted transition-colors text-muted-foreground">
                • Why this trend?
              </button>
              <button className="w-full text-left text-xs px-2 py-1 rounded hover:bg-muted transition-colors text-muted-foreground">
                • Best heroes to play?
              </button>
              <button className="w-full text-left text-xs px-2 py-1 rounded hover:bg-muted transition-colors text-muted-foreground">
                • How to improve?
              </button>
            </div>
          </div>

          {/* Conversation history */}
          <div className="flex-1 overflow-y-auto p-4">
            <p className="text-sm text-muted-foreground text-center">
              Ask a question to start the conversation
            </p>
          </div>
        </div>
      </div>
    </PasswordGate>
  )
}
