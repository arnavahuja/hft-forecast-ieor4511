import { defineConfig, devices } from '@playwright/test'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const PYTHON = process.env.PYTHON ?? 'python'
const API_PORT = 8765
const WEB_PORT = 5174
const dataDir = mkdtempSync(join(tmpdir(), 'backbone-e2e-'))

export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    trace: 'retain-on-failure',
    colorScheme: 'dark',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1600, height: 1000 } } }],
  webServer: [
    {
      command: `${PYTHON} -m backbone.cli serve --port ${API_PORT}`,
      cwd: '..',
      url: `http://127.0.0.1:${API_PORT}/api/v1/health`,
      reuseExistingServer: false,
      env: {
        HFTF_DATA_DIR: dataDir,
        HFTF_USER_PLUGINS_DIR: join(dataDir, 'user_plugins'),
        HFTF_FRONTEND_ORIGIN: `http://127.0.0.1:${WEB_PORT}`,
        HFTF_LOG_LEVEL: 'WARNING',
        HFTF_MAX_WORKERS: '1',
      },
      timeout: 60_000,
    },
    {
      command: `pnpm exec vite --host 127.0.0.1 --port ${WEB_PORT} --strictPort`,
      url: `http://127.0.0.1:${WEB_PORT}`,
      reuseExistingServer: false,
      env: { HFTF_API: `http://127.0.0.1:${API_PORT}` },
      timeout: 60_000,
    },
  ],
})
