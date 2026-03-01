import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { ThemeProvider } from '@/components/providers/theme-provider'
import { NavHeader } from '@/components/layout/nav-header'
import './globals.css'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
})

export const metadata: Metadata = {
  title: 'HotS Fever',
  description: 'Insights and draft assistant for Heroes of the Storm',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          <div className="min-h-screen gradient-gaming">
            <NavHeader />
            <main className="container py-6">{children}</main>
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}
