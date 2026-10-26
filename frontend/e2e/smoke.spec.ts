import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

const API = '/api/v1'

/** The economic test and `hftf run` both post this shape; the UI only ever reads runs. */
function config(name: string, costBps: number) {
  return {
    name,
    data: {
      source: 'synthetic',
      dataset: 'gbm',
      instruments: ['AAA', 'BBB', 'CCC'],
      start: '2016-01-01',
      end: '2020-12-31',
    },
    strategy: { name: 'buy_and_hold' },
    costs: [{ name: 'bps_notional', params: { bps: costBps } }],
    benchmark: 'MKT',
    experiment: 'e2e',
  }
}

/** Launch a run through the API and wait for it to finish; returns the run id. */
async function launch(request: APIRequestContext, name: string, costBps: number): Promise<string> {
  const response = await request.post(`${API}/runs`, { data: { config: config(name, costBps) } })
  expect(response.ok()).toBeTruthy()
  const { run } = (await response.json()) as { run: { id: string } }
  await expect
    .poll(
      async () => {
        const got = await request.get(`${API}/runs/${run.id}`)
        return ((await got.json()) as { status: string }).status
      },
      { timeout: 60_000 },
    )
    .toBe('completed')
  return run.id
}

test('see a run, its charts and a comparison of two runs', async ({ page, request }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))

  const first = await launch(request, 'e2e hold', 2)
  await launch(request, 'e2e hold cheap', 0)

  await page.goto(`/runs/${first}`)
  await expect(page.getByText('Equity curve', { exact: true })).toBeVisible()
  await expect(page.getByText('Sharpe', { exact: true }).first()).toBeVisible()
  await expect(page.locator('canvas').first()).toBeVisible()
  await page.getByRole('tab', { name: 'Trades' }).click()
  await expect(page.getByText(/^Trades \(\d+\)$/)).toBeVisible()

  await page.goto('/runs')
  await expect(page.getByRole('link', { name: /e2e hold cheap/ })).toBeVisible()
  await page.getByLabel('Select all').check()
  await page.getByRole('button', { name: 'Compare' }).click()
  await expect(page).toHaveURL(/\/compare$/)
  await expect(page.getByText('Metrics (best in green, worst in red per row)')).toBeVisible()
  await expect(page.getByText('Equity curves', { exact: true })).toBeVisible()
  await expect(page.getByText('Return correlation', { exact: true })).toBeVisible()

  expect(errors).toEqual([])
})

test('the data page lists the catalog and offers an import', async ({ page }) => {
  await page.goto('/data')
  await expect(page.getByRole('heading', { name: 'Data' })).toBeVisible()
  await expect(page.getByText('Import a local file (CSV, Parquet, Excel, Feather)')).toBeVisible()
  await expect(page.getByLabel('Book levels to import')).toHaveCount(0) // only after a preview
})

test('theme contains no purple', async ({ page }) => {
  await page.goto('/')
  const colors = await page.evaluate(() => {
    const seen = new Set<string>()
    for (const el of Array.from(document.querySelectorAll('*'))) {
      const style = getComputedStyle(el)
      for (const prop of ['color', 'backgroundColor', 'borderColor', 'outlineColor'] as const) {
        seen.add(style[prop])
      }
    }
    return Array.from(seen)
  })
  const purple = colors.filter((c) => {
    const m = /rgba?\((\d+), (\d+), (\d+)/.exec(c)
    if (!m) return false
    const [r, g, b] = [Number(m[1]), Number(m[2]), Number(m[3])]
    // hue in the purple/violet/magenta range with meaningful saturation
    const max = Math.max(r, g, b)
    const min = Math.min(r, g, b)
    if (max - min < 40) return false
    let h = 0
    if (max === r) h = ((g - b) / (max - min)) % 6
    else if (max === g) h = (b - r) / (max - min) + 2
    else h = (r - g) / (max - min) + 4
    const hue = (h * 60 + 360) % 360
    return hue >= 250 && hue <= 330
  })
  expect(purple).toEqual([])
})
