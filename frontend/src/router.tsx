import { createBrowserRouter } from 'react-router-dom'
import { Layout } from '@/components/Layout'
import { ComparePage } from '@/pages/Compare'
import { DashboardPage } from '@/pages/Dashboard'
import { DataPage } from '@/pages/Data'
import { NotFoundPage } from '@/pages/NotFound'
import { RunDetailPage } from '@/pages/RunDetail'
import { RunsPage } from '@/pages/Runs'
import { SettingsPage } from '@/pages/Settings'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: 'data', element: <DataPage /> },
      { path: 'runs', element: <RunsPage /> },
      { path: 'runs/:runId', element: <RunDetailPage /> },
      { path: 'compare', element: <ComparePage /> },
      { path: 'settings', element: <SettingsPage /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
])
