import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { JsonObject } from '@/components/SchemaForm'

export interface PluginRefDraft {
  /** Stable key for list rendering and drag-and-drop. */
  key: string
  name: string
  params: JsonObject
}

export interface LabDraft {
  name: string
  source: string
  dataset: string
  instruments: string
  universe: PluginRefDraft | null
  start: string
  end: string
  frequency: string
  adjustment: string
  strategy: PluginRefDraft | null
  portfolio: PluginRefDraft | null
  overlays: PluginRefDraft[]
  costs: PluginRefDraft[]
  slippage: PluginRefDraft | null
  fillModel: PluginRefDraft | null
  engine: string
  lagBars: number
  price: string
  rebalance: string
  rebalanceEveryN: number
  cashRate: number
  maxParticipation: number
  latencyBars: number
  benchmark: string
  riskFree: string
  initialCapital: number
  seed: number
  experiment: string
  tags: string
  notes: string
  trainEnd: string
  validationEnd: string
  testUnlocked: boolean
}

export const defaultDraft: LabDraft = {
  name: '',
  source: 'yahoo',
  dataset: 'daily',
  instruments: 'SPY, TLT, GLD',
  universe: null,
  start: '2010-01-01',
  end: '2024-12-31',
  frequency: '1d',
  adjustment: 'split',
  strategy: { key: 'strategy', name: 'sma_crossover', params: {} },
  portfolio: null,
  overlays: [],
  costs: [{ key: 'cost-0', name: 'bps_notional', params: {} }],
  slippage: { key: 'slippage', name: 'half_spread', params: {} },
  fillModel: null,
  engine: 'vectorized',
  lagBars: 1,
  price: 'close',
  rebalance: 'on_change',
  rebalanceEveryN: 1,
  cashRate: 0,
  maxParticipation: 1,
  latencyBars: 0,
  benchmark: 'SPY',
  riskFree: '',
  initialCapital: 1_000_000,
  seed: 42,
  experiment: '',
  tags: '',
  notes: '',
  trainEnd: '',
  validationEnd: '',
  testUnlocked: false,
}

interface LabState {
  draft: LabDraft
  set: (patch: Partial<LabDraft>) => void
  replace: (draft: LabDraft) => void
  reset: () => void
}

export const useLabStore = create<LabState>()(
  persist(
    (set) => ({
      draft: defaultDraft,
      set: (patch) => set((s) => ({ draft: { ...s.draft, ...patch } })),
      replace: (draft) => set({ draft }),
      reset: () => set({ draft: defaultDraft }),
    }),
    {
      name: 'backbone-lab-draft',
      version: 3,
      // fill fields added in later versions with their defaults
      migrate: (persisted) => {
        const old = (persisted as { draft?: Partial<LabDraft> } | null)?.draft ?? {}
        return { draft: { ...defaultDraft, ...old } } as LabState
      },
    },
  ),
)

let counter = 0
export function newKey(prefix: string): string {
  counter += 1
  return `${prefix}-${Date.now().toString(36)}-${counter}`
}
