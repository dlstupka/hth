# HTH Work Summary

`HTH report` → `hth-work-summary` produces a project-wide daily, monthly, annual and
lifetime work report. The human-readable report is
`reports/hth-work-summary/summary.md` in the results repository; the adjacent
`summary.json`, `days/YYYY-MM-DD.json`, `months/YYYY-MM.json` and
`years/YYYY.json` files are the
durable machine-readable record.

The report writer is a **consumer**, not an execution telemetry producer. It
reads the pipeline's full Git history, GitHub Actions run and release metadata,
the current CBE resource-lifecycle count, and an explicit estimate ledger at
`config/work-summary-estimates.json`. It never infers actual CPU utilization,
token consumption or billed GPT spend from Git activity.

## Interpretation

- Git commits, authors, release tags/dates and Actions run counts are observed
  facts. Actions wall hours are the interval from `run_started_at` to
  `updated_at` for completed runs, a proxy rather than a CPU measurement.
- June–September 2026 human and compute ranges are the historical Project
  Juana reconstruction, not time sheets. The ChatGPT and Codex ranges are
  deliberately broad assumptions. In months with Git history, the historical
  monthly estimate is allocated over Git-active days by commit count. June has
  no Git history and is only estimated at monthly/lifetime levels.
- On later days without explicit time records, commit count produces a bounded
  human-hours proxy and assistant-hours proxy. Completed Actions wall time
  produces a deliberately broad compute-core-hours proxy. A day without such
  evidence is **unknown**, not zero.
- GPT dollars are an **illustrative API-equivalent scenario** from token
  intensity and rate ranges in the ledger. They are neither actual API charges
  nor the user's ChatGPT/Codex subscription/credit spend. To show actual spend,
  add a daily entry backed by account billing/usage evidence.
- Human and assistant hours overlap. Do not add them into one labor total.

The ledger supports explicit `days` overrides, for example:

```json
"2026-10-04": {
  "human_hours": [4, 5],
  "chatgpt_hours": [0.5, 1],
  "codex_hours": [2, 3],
  "compute_core_hours": [12, 20],
  "gpt_cost_usd": [1, 3]
}
```

These remain estimates unless separately documented as observations. A future
canonical telemetry producer can populate the same daily input without moving
the measurement logic into Report Writer. For a June–September correction,
also update that month's `historical_months` range; those monthly ranges are
authoritative over the allocated daily values.

## Build avoidance and persistence

The workflow checks out only `reports/hth-work-summary/` and the compact CBE
lifecycle ledger from the results repository. `pending-since` identifies the
oldest day without a persisted snapshot. GitHub run collection is bounded to
that date onward; the report reads Git history only from missing days onward.
Closed days, months and years are reused unchanged. The current day, month and
year are recomputed. This avoids repeatedly reconstructing historical estimates or
re-fetching historical Actions runs. Report publication uses the existing
bounded, concurrent-writer-safe results transaction, regenerating against the
fresh results checkout on a retry.

Cached records deliberately preserve the methodology and evidence available at
their first publication. To correct a historical record, use the CLI's explicit
`--refresh-history` option with complete historical GitHub evidence and review
the replacement before publishing; the normal workflow never refreshes closed
history automatically.

## Local use

The report is also available through `python -m hth.report_generator
hth-work-summary --repository-root . --results-root RESULTS --output-dir OUT
--ledger config/work-summary-estimates.json`. Supplying
`--github-snapshot SNAPSHOT` adds observed Actions and release facts. The
snapshot can be collected with `python -m hth.work_summary collect-github
--repository dlstupka/hth --release-repository dlstupka/hth --since YYYY-MM-DD
--through YYYY-MM-DD --output SNAPSHOT` using `GITHUB_TOKEN`. Without a
snapshot, Actions/release counts in newly generated days are incomplete and
should not be interpreted as zero lifetime activity.
