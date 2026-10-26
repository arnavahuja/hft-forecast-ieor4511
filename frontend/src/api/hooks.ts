import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { API_BASE, ApiError, api, unwrap, type Schemas } from './client'

export type Plugin = Schemas['PluginOut']
export type PluginKind = Schemas['PluginKind']
export type RunRecord = Schemas['RunRecord']
export type BacktestConfig = Schemas['BacktestConfig-Input']
export type ChartSpec = Schemas['ChartSpec']
export type ChartInfo = Schemas['ChartInfoOut']
export type MetricsOut = Schemas['MetricsOut']
export type MetricDescriptor = Schemas['MetricDescriptor']
export type MetricValue = Schemas['MetricValue']
export type Job = Schemas['JobOut']
export type DatasetRecord = Schemas['DatasetRecord']
export type CompareTable = Schemas['CompareMetricsOut']
export type ImportMapping = Schemas['ImportMapping']
export type ReportRequest = Schemas['ReportRequest']

export interface Window {
  start?: string | null
  end?: string | null
}

const REFRESH_ACTIVE_MS = 2000

export const keys = {
  plugins: ['plugins'] as const,
  pluginHealth: ['plugins', 'health'] as const,
  runs: (filters: object) => ['runs', filters] as const,
  run: (id: string) => ['run', id] as const,
  metrics: (id: string, w: Window) => ['run', id, 'metrics', w] as const,
  charts: (id: string) => ['run', id, 'charts'] as const,
  chart: (id: string, name: string, w: Window, params: string) =>
    ['run', id, 'chart', name, w, params] as const,
  catalog: ['catalog'] as const,
  sources: ['sources'] as const,
  jobs: ['jobs'] as const,
}

export function usePlugins() {
  return useQuery({
    queryKey: keys.plugins,
    queryFn: async () => unwrap(await api.GET('/api/v1/plugins')),
    staleTime: 30_000,
  })
}

export function usePluginsOfKind(kind: PluginKind) {
  const all = usePlugins()
  return { ...all, data: all.data?.filter((p) => p.kind === kind) }
}

export function usePluginHealth() {
  return useQuery({
    queryKey: keys.pluginHealth,
    queryFn: async () => unwrap(await api.GET('/api/v1/plugins/health')),
  })
}

export interface RunFilters {
  search?: string
  experiment?: string
  status?: string
  tag?: string
  include_archived?: boolean
  limit?: number
}

export function useRuns(filters: RunFilters = {}) {
  return useQuery({
    queryKey: keys.runs(filters),
    queryFn: async () => unwrap(await api.GET('/api/v1/runs', { params: { query: filters } })),
    placeholderData: keepPreviousData,
    refetchInterval: (q) =>
      q.state.data?.some((r) => r.status === 'queued' || r.status === 'running')
        ? REFRESH_ACTIVE_MS
        : false,
  })
}

export function useRun(id: string | undefined) {
  return useQuery({
    queryKey: keys.run(id ?? ''),
    enabled: Boolean(id),
    queryFn: async () =>
      unwrap(await api.GET('/api/v1/runs/{run_id}', { params: { path: { run_id: id ?? '' } } })),
    refetchInterval: (q) =>
      q.state.data && ['queued', 'running'].includes(q.state.data.status)
        ? REFRESH_ACTIVE_MS
        : false,
  })
}

export function useRunMetrics(id: string, window: Window, enabled = true) {
  return useQuery({
    queryKey: keys.metrics(id, window),
    enabled,
    placeholderData: keepPreviousData,
    queryFn: async () =>
      unwrap(
        await api.GET('/api/v1/runs/{run_id}/metrics', {
          params: {
            path: { run_id: id },
            query: { start: window.start ?? null, end: window.end ?? null },
          },
        }),
      ),
  })
}

export function useRunCharts(id: string, enabled = true) {
  return useQuery({
    queryKey: keys.charts(id),
    enabled,
    queryFn: async () =>
      unwrap(await api.GET('/api/v1/runs/{run_id}/charts', { params: { path: { run_id: id } } })),
  })
}

export function useRunChart(
  id: string,
  chart: string,
  window: Window = {},
  params: Record<string, unknown> | null = null,
  enabled = true,
) {
  const p = params ? JSON.stringify(params) : ''
  return useQuery({
    queryKey: keys.chart(id, chart, window, p),
    enabled,
    queryFn: async () =>
      unwrap(
        await api.GET('/api/v1/runs/{run_id}/charts/{chart}', {
          params: {
            path: { run_id: id, chart },
            query: { start: window.start ?? null, end: window.end ?? null, params: p || null },
          },
        }),
      ),
  })
}

export function useRunTrades(id: string, offset = 0, limit = 500) {
  return useQuery({
    queryKey: ['run', id, 'trades', offset, limit],
    queryFn: async () =>
      unwrap(
        await api.GET('/api/v1/runs/{run_id}/trades', {
          params: { path: { run_id: id }, query: { offset, limit } },
        }),
      ),
    placeholderData: keepPreviousData,
  })
}

export function useRunPositions(id: string) {
  return useQuery({
    queryKey: ['run', id, 'positions'],
    queryFn: async () =>
      unwrap(
        await api.GET('/api/v1/runs/{run_id}/positions', { params: { path: { run_id: id } } }),
      ),
  })
}

export function useRunOverlays(id: string) {
  return useQuery({
    queryKey: ['run', id, 'overlays'],
    queryFn: async () =>
      unwrap(await api.GET('/api/v1/runs/{run_id}/overlays', { params: { path: { run_id: id } } })),
  })
}

export function useRunLogs(id: string, refetch: boolean) {
  return useQuery({
    queryKey: ['run', id, 'logs'],
    queryFn: async () => {
      const res = await fetch(`/api/v1/runs/${encodeURIComponent(id)}/logs`)
      return res.text()
    },
    refetchInterval: refetch ? REFRESH_ACTIVE_MS : false,
  })
}

export function useExperiments() {
  return useQuery({
    queryKey: ['experiments'],
    queryFn: async () => unwrap(await api.GET('/api/v1/runs/experiments')),
  })
}

export function useCatalog() {
  return useQuery({
    queryKey: keys.catalog,
    queryFn: async () => unwrap(await api.GET('/api/v1/data/catalog')),
  })
}

export function useSources() {
  return useQuery({
    queryKey: keys.sources,
    queryFn: async () => unwrap(await api.GET('/api/v1/data/sources')),
    staleTime: 60_000,
  })
}

export function useDatasetPreview(id: string | null) {
  return useQuery({
    queryKey: ['dataset', id, 'preview'],
    enabled: Boolean(id),
    queryFn: async () =>
      unwrap(
        await api.GET('/api/v1/data/datasets/{dataset_id}/preview', {
          params: { path: { dataset_id: id ?? '' }, query: { limit: 100 } },
        }),
      ),
  })
}

export function useDatasetQuality(id: string | null) {
  return useQuery({
    queryKey: ['dataset', id, 'quality'],
    enabled: Boolean(id),
    queryFn: async () =>
      unwrap(
        await api.GET('/api/v1/data/datasets/{dataset_id}/quality', {
          params: { path: { dataset_id: id ?? '' } },
        }),
      ),
  })
}

export function useJobs() {
  return useQuery({
    queryKey: keys.jobs,
    queryFn: async () => unwrap(await api.GET('/api/v1/jobs')),
    refetchInterval: (q) =>
      q.state.data?.some((j) => j.status === 'queued' || j.status === 'running')
        ? REFRESH_ACTIVE_MS
        : 10_000,
  })
}

export function useSettings() {
  return useQuery({
    queryKey: ['settings'],
    queryFn: async () => unwrap(await api.GET('/api/v1/settings')),
  })
}

export function useCompareCharts() {
  return useQuery({
    queryKey: ['compare', 'charts'],
    queryFn: async () => unwrap(await api.GET('/api/v1/compare/charts')),
    staleTime: 60_000,
  })
}

export function useCompareMetrics(runIds: string[], align: boolean) {
  return useQuery({
    queryKey: ['compare', 'metrics', runIds, align],
    enabled: runIds.length > 0,
    placeholderData: keepPreviousData,
    queryFn: async () =>
      unwrap(await api.POST('/api/v1/compare/metrics', { body: { run_ids: runIds, align } })),
  })
}

export function useCompareChart(
  chart: string,
  runIds: string[],
  align: boolean,
  params: Record<string, unknown> = {},
) {
  return useQuery({
    queryKey: ['compare', 'chart', chart, runIds, align, params],
    enabled: runIds.length > 0,
    queryFn: async () =>
      unwrap(
        await api.POST('/api/v1/compare/charts/{chart}', {
          params: { path: { chart } },
          body: { run_ids: runIds, align, params },
        }),
      ),
  })
}

// ------------------------------------------------------------------ mutations

/** Build the comparison report (zip of CSVs and PNG charts) and save it. */
export function useDownloadReport() {
  return useMutation({
    mutationFn: async (body: ReportRequest) => {
      const response = await fetch(`${API_BASE}/compare/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!response.ok) {
        const err = (await response.json().catch(() => null)) as {
          code?: string
          message?: string
        } | null
        throw new ApiError(
          {
            code: err?.code ?? `http_${response.status}`,
            message: err?.message ?? response.statusText,
          },
          response.status,
        )
      }
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'comparison-report.zip'
      link.click()
      URL.revokeObjectURL(url)
      return blob.size
    },
  })
}

export function useLaunchRun() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (config: BacktestConfig) =>
      unwrap(await api.POST('/api/v1/runs', { body: { config } })),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['runs'] })
      void qc.invalidateQueries({ queryKey: keys.jobs })
    },
  })
}

export function useUpdateRun(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (body: Schemas['LabelsUpdate']) =>
      unwrap(await api.PATCH('/api/v1/runs/{run_id}', { params: { path: { run_id: id } }, body })),
    onSuccess: (run) => {
      qc.setQueryData(keys.run(id), run)
      void qc.invalidateQueries({ queryKey: ['runs'] })
    },
  })
}

export function useDeleteRuns() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (ids: string[]) => {
      for (const id of ids) {
        const res = await api.DELETE('/api/v1/runs/{run_id}', { params: { path: { run_id: id } } })
        unwrap({ ...res, data: null })
      }
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['runs'] }),
  })
}

export function useCancelRun() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (id: string) =>
      unwrap(await api.POST('/api/v1/runs/{run_id}/cancel', { params: { path: { run_id: id } } })),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['runs'] }),
  })
}

export function useCloneRun() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (args: { id: string; changes: Record<string, unknown>; run: boolean }) =>
      unwrap(
        await api.POST('/api/v1/runs/{run_id}/clone', {
          params: { path: { run_id: args.id } },
          body: { changes: args.changes, run: args.run },
        }),
      ),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['runs'] }),
  })
}

export function usePullData() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (body: Schemas['PullRequest']) =>
      unwrap(await api.POST('/api/v1/data/pull', { body })),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.jobs }),
  })
}

export function useImportData() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (body: Schemas['ImportRequest']) =>
      unwrap(await api.POST('/api/v1/data/import', { body })),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.catalog }),
  })
}

export function useRefreshDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (id: string) =>
      unwrap(
        await api.POST('/api/v1/data/datasets/{dataset_id}/refresh', {
          params: { path: { dataset_id: id } },
        }),
      ),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.catalog }),
  })
}

export function useDeleteDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (id: string) => {
      const res = await api.DELETE('/api/v1/data/datasets/{dataset_id}', {
        params: { path: { dataset_id: id } },
      })
      unwrap({ ...res, data: null })
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.catalog }),
  })
}

export function useCombine() {
  return useMutation({
    mutationFn: async (body: Schemas['CombineRequest']) =>
      unwrap(await api.POST('/api/v1/compare/combine', { body })),
  })
}

export function useWrdsTest() {
  return useMutation({
    mutationFn: async () => unwrap(await api.POST('/api/v1/settings/wrds-test')),
  })
}

// ------------------------------------------------------------------ research
