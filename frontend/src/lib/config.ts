import type { BacktestConfig } from '@/api/hooks'
import { newKey, type LabDraft, type PluginRefDraft } from '@/store/lab'

type Ref = BacktestConfig['strategy']

function ref(d: PluginRefDraft): Ref {
  return { name: d.name, params: d.params, version: null }
}

function splitList(text: string): string[] {
  return text
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter(Boolean)
}

/** Build the API config from the Strategy Lab draft. */
export function draftToConfig(d: LabDraft): BacktestConfig {
  const hasSplit = Boolean(d.trainEnd || d.validationEnd)
  return {
    name: d.name,
    data: {
      source: d.source,
      dataset: d.dataset,
      instruments: splitList(d.instruments),
      universe: d.universe ? ref(d.universe) : null,
      start: d.start,
      end: d.end,
      frequency: d.frequency as BacktestConfig['data']['frequency'],
      adjustment: d.adjustment as BacktestConfig['data']['adjustment'],
      fields: [],
      options: {},
    },
    strategy: d.strategy ? ref(d.strategy) : { name: '', params: {}, version: null },
    constructor: d.portfolio ? ref(d.portfolio) : null,
    overlays: d.overlays.map(ref),
    costs: d.costs.map(ref),
    slippage: d.slippage ? ref(d.slippage) : null,
    fill_model: d.fillModel ? ref(d.fillModel) : null,
    execution: {
      engine: d.engine,
      lag_bars: d.lagBars,
      price: d.price as NonNullable<BacktestConfig['execution']>['price'],
      rebalance: d.rebalance as NonNullable<BacktestConfig['execution']>['rebalance'],
      rebalance_every_n: d.rebalanceEveryN,
      cash_rate: d.cashRate,
      max_participation: d.maxParticipation,
      latency_bars: d.latencyBars,
      calendar: 'XNYS',
    },
    benchmark: d.benchmark.trim() || null,
    risk_free: d.riskFree.trim() || null,
    initial_capital: d.initialCapital,
    seed: d.seed,
    split: hasSplit
      ? {
          train_end: d.trainEnd || null,
          validation_end: d.validationEnd || null,
          test_unlocked: d.testUnlocked,
        }
      : null,
    tags: splitList(d.tags),
    notes: d.notes,
    experiment: d.experiment.trim() || null,
  }
}

function draftRef(prefix: string, r: Ref | null | undefined): PluginRefDraft | null {
  return r ? { key: newKey(prefix), name: r.name, params: r.params ?? {} } : null
}

/** Load a config (e.g. from a preset or a run) back into the Strategy Lab. */
export function configToDraft(c: BacktestConfig): LabDraft {
  return {
    name: c.name ?? '',
    source: c.data.source ?? 'yahoo',
    dataset: c.data.dataset ?? 'daily',
    instruments: (c.data.instruments ?? []).join(', '),
    universe: draftRef('universe', c.data.universe),
    start: c.data.start,
    end: c.data.end,
    frequency: c.data.frequency ?? '1d',
    adjustment: c.data.adjustment ?? 'split',
    strategy: draftRef('strategy', c.strategy),
    portfolio: draftRef('constructor', c.constructor),
    overlays: (c.overlays ?? []).map((o) => ({
      key: newKey('overlay'),
      name: o.name,
      params: o.params ?? {},
    })),
    costs: (c.costs ?? []).map((o) => ({
      key: newKey('cost'),
      name: o.name,
      params: o.params ?? {},
    })),
    slippage: draftRef('slippage', c.slippage),
    fillModel: draftRef('fill', c.fill_model),
    engine: c.execution?.engine ?? 'vectorized',
    lagBars: c.execution?.lag_bars ?? 1,
    price: c.execution?.price ?? 'close',
    rebalance: c.execution?.rebalance ?? 'on_change',
    rebalanceEveryN: c.execution?.rebalance_every_n ?? 1,
    cashRate: c.execution?.cash_rate ?? 0,
    maxParticipation: c.execution?.max_participation ?? 1,
    latencyBars: c.execution?.latency_bars ?? 0,
    benchmark: c.benchmark ?? '',
    riskFree: c.risk_free ?? '',
    initialCapital: c.initial_capital ?? 1_000_000,
    seed: c.seed ?? 42,
    experiment: c.experiment ?? '',
    tags: (c.tags ?? []).join(', '),
    notes: c.notes ?? '',
    trainEnd: c.split?.train_end ?? '',
    validationEnd: c.split?.validation_end ?? '',
    testUnlocked: c.split?.test_unlocked ?? false,
  }
}
