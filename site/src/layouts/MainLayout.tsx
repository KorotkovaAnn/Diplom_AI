import type { ReactNode } from 'react'
import { Header } from '../widgets/Header.tsx'
import { Footer } from '../widgets/Footer.tsx'
import { HelpWidget } from '../widgets/HelpWidget.tsx'

interface MainLayoutProps {
  children: ReactNode
}

export function MainLayout({ children }: MainLayoutProps) {
  return (
    <div className="app-root">
      <Header />
      <main className="app-main">{children}</main>
      <Footer />
      <HelpWidget />
    </div>
  )
}

