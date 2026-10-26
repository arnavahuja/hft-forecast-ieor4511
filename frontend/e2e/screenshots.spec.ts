import { expect, test } from '@playwright/test'

/** Visual review helper: `SCREENSHOTS=1 pnpm e2e screenshots` saves page screenshots. */
test.skip(!process.env.SCREENSHOTS, 'set SCREENSHOTS=1 to capture screenshots')

const OUT = process.env.SCREENSHOT_DIR ?? 'test-results/screenshots'
const API = '/api/v1'

test('capture pages', async ({ page, request }) => {
  for (const [name, bps] of [
    ['screenshot hold', 2],
    ['screenshot hold cheap', 0],
  ] as const) {
    await request.post(`${API}/runs`, {
      data: {
        config: {
          name,
          data: {
            source: 'synthetic',
            dataset: 'gbm',
            instruments: ['AAA', 'BBB', 'CCC', 'DDD'],
            start: '2012-01-01',
            end: '2022-12-31',
          },
          strategy: { name: 'buy_and_hold' },
          costs: [{ name: 'bps_notional', params: { bps } }],
          benchmark: 'MKT',
        },
      },
    })
  }
  await page.goto('/runs')
  const row = page.getByRole('link', { name: /screenshot hold cheap/ })
  await expect(row).toBeVisible({ timeout: 60_000 })
  await page.screenshot({ path: `${OUT}/runs.png`, fullPage: true })

  await row.click()
  await expect(page.locator('canvas').first()).toBeVisible()
  await page.waitForTimeout(800)
  await page.screenshot({ path: `${OUT}/run-overview.png`, fullPage: true })
  for (const tab of ['Performance', 'Risk', 'Positions', 'Trades']) {
    await page.getByRole('tab', { name: tab }).click()
    await page.waitForTimeout(1200)
    await page.screenshot({ path: `${OUT}/run-${tab.toLowerCase()}.png`, fullPage: true })
  }

  await page.goto('/runs')
  await page.getByLabel('Select all').check()
  await page.getByRole('button', { name: 'Compare' }).click()
  await page.waitForTimeout(2500)
  await page.screenshot({ path: `${OUT}/compare.png`, fullPage: true })
  await page.goto('/data')
  await page.waitForTimeout(800)
  await page.screenshot({ path: `${OUT}/data.png`, fullPage: true })
  await page.goto('/')
  await page.waitForTimeout(800)
  await page.screenshot({ path: `${OUT}/dashboard.png`, fullPage: true })
})
