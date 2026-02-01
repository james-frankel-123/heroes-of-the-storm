'use client'

import * as React from 'react'
import { useState, useEffect } from 'react'
import { Lock } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

interface PasswordGateProps {
  children: React.ReactNode
  requiredPassword: string
  storageKey: string
}

export function PasswordGate({ children, requiredPassword, storageKey }: PasswordGateProps) {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Check if already authenticated in this session
    const stored = sessionStorage.getItem(storageKey)
    if (stored === requiredPassword) {
      setIsAuthenticated(true)
    }
    setIsLoading(false)
  }, [storageKey, requiredPassword])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    if (password === requiredPassword) {
      sessionStorage.setItem(storageKey, password)
      setIsAuthenticated(true)
      setError('')
    } else {
      setError('Incorrect password')
      setPassword('')
    }
  }

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="flex min-h-screen items-center justify-center p-4">
        <div className="w-full max-w-md space-y-6 rounded-lg border border-border bg-card p-8 shadow-lg">
          <div className="flex flex-col items-center space-y-2">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary-500/10">
              <Lock className="h-6 w-6 text-primary-500" />
            </div>
            <h2 className="text-2xl font-bold">Password Required</h2>
            <p className="text-center text-sm text-muted-foreground">
              This page is password protected. Please enter the password to continue.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <Input
                type="password"
                placeholder="Enter password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className={error ? 'border-red-500' : ''}
                autoFocus
              />
              {error && (
                <p className="mt-2 text-sm text-red-500">{error}</p>
              )}
            </div>

            <Button type="submit" className="w-full">
              Unlock
            </Button>
          </form>
        </div>
      </div>
    )
  }

  return <>{children}</>
}
