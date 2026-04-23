import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { ThemeProvider } from '@/components/providers/theme-provider'
import { NavHeader } from '@/components/layout/nav-header'
import { NavProgress } from '@/components/layout/nav-progress'
import { PraiseBanner } from '@/components/layout/praise-banner'
import { SiteFooter } from '@/components/layout/site-footer'
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
            <div id="app-header">
              <PraiseBanner />
              <NavProgress />
              <NavHeader />
            </div>
            <main className="container py-6">{children}</main>
            <SiteFooter />
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}
