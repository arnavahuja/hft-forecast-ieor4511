import clsx from 'clsx'
import { NavLink, Outlet } from 'react-router-dom'

const NAV = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/data', label: 'Data' },
  { to: '/runs', label: 'Runs' },
  { to: '/compare', label: 'Compare' },
  { to: '/settings', label: 'Settings' },
] as const

export function Layout() {
  return (
    <div className="flex min-h-screen bg-canvas text-primary">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-elevated focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <aside className="sticky top-0 flex h-screen w-52 shrink-0 flex-col border-r border-border bg-surface">
        <div className="flex items-center gap-2 px-4 py-4">
          <svg viewBox="0 0 32 32" className="h-6 w-6" aria-hidden>
            <rect width="32" height="32" rx="6" className="fill-canvas" />
            <path
              d="M6 22 L12 14 L17 18 L26 8"
              className="stroke-accent"
              strokeWidth="3"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span className="text-sm font-semibold tracking-wide">hft-forecast</span>
        </div>
        <nav aria-label="Main" className="flex flex-1 flex-col gap-0.5 px-2">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={'end' in item ? item.end : false}
              className={({ isActive }) =>
                clsx(
                  'rounded-md px-3 py-2 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent',
                  isActive
                    ? 'bg-elevated text-primary'
                    : 'text-secondary hover:bg-elevated hover:text-primary',
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="px-4 py-3 text-2xs text-muted">localhost only</div>
      </aside>
      <main id="main" className="min-w-0 flex-1 px-6 py-6">
        <Outlet />
      </main>
    </div>
  )
}
