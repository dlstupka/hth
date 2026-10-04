"""Persistent, evidence-labelled HTH project work summary.

Execution systems own telemetry. This report only consumes Git/GitHub facts and
explicit estimates; it never presents a modeled hour or dollar as billed usage.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA = "hth-work-summary-v1"


class _CentralFallback(tzinfo):
    """US Central time since 2007 when Windows has no IANA tzdata package."""

    @staticmethod
    def _second_sunday_march(year):
        first = date(year, 3, 1)
        return 1 + (6 - first.weekday()) % 7 + 7

    @staticmethod
    def _first_sunday_november(year):
        first = date(year, 11, 1)
        return 1 + (6 - first.weekday()) % 7

    def dst(self, dt):
        if dt is None:
            return timedelta(0)
        start = datetime(dt.year, 3, self._second_sunday_march(dt.year), 2)
        end = datetime(dt.year, 11, self._first_sunday_november(dt.year), 2)
        local = dt.replace(tzinfo=None)
        if local.date() == end.date() and local.hour == 1 and dt.fold:
            return timedelta(0)
        return timedelta(hours=1) if start <= local < end else timedelta(0)

    def utcoffset(self, dt):
        return timedelta(hours=-6) + self.dst(dt)

    def tzname(self, dt):
        return "CDT" if self.dst(dt) else "CST"

    def fromutc(self, dt):
        utc = dt.replace(tzinfo=timezone.utc)
        start = datetime(dt.year, 3, self._second_sunday_march(dt.year), 8, tzinfo=timezone.utc)
        end = datetime(dt.year, 11, self._first_sunday_november(dt.year), 7, tzinfo=timezone.utc)
        offset = timedelta(hours=-5 if start <= utc < end else -6)
        fold = 1 if end <= utc < end + timedelta(hours=1) else 0
        return (utc + offset).replace(tzinfo=self, fold=fold)

    def __str__(self):
        return "America/Chicago"


try:
    ZONE = ZoneInfo("America/Chicago")
except ZoneInfoNotFoundError:
    ZONE = _CentralFallback()
START = date(2026, 6, 1)
FIELDS = ("human_hours", "chatgpt_hours", "codex_hours", "compute_core_hours",
          "chatgpt_cost_usd", "codex_cost_usd", "gpt_cost_usd")
EFFORT_FIELDS = ("human_hours", "chatgpt_hours", "codex_hours", "compute_core_hours")


def _json(path: Path, fallback=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else fallback


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _month_range(month: str) -> tuple[date, date]:
    year, number = map(int, month.split("-"))
    return date(year, number, 1), date(year, number, calendar.monthrange(year, number)[1])


def _range(value):
    if not isinstance(value, list) or len(value) != 2:
        return None
    low, high = (float(item) for item in value)
    if low < 0 or high < low:
        raise ValueError(f"Invalid work estimate range: {value!r}")
    return [round(low, 3), round(high, 3)]


def _sum_ranges(values):
    known = [item for item in values if item is not None]
    return [round(sum(item[0] for item in known), 3), round(sum(item[1] for item in known), 3)] if known else None


def _scaled(value, share):
    return [round(value[0] * share, 3), round(value[1] * share, 3)] if value is not None else None


def _git_commits(repo: Path, since: date, through: date):
    # A full checkout is required. Use commit timestamps (not author timestamps)
    # and the project's reporting timezone, so rebases do not silently shift days.
    start = datetime.combine(since, datetime.min.time(), ZONE)
    finish = datetime.combine(through + timedelta(days=1), datetime.min.time(), ZONE)
    repository = repo.resolve()
    command = ["git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(repository), "log", "--all", "--format=%H%x1f%cI%x1f%aN%x1f%s%x1e",
               f"--since={start.isoformat()}", f"--until={finish.isoformat()}"]
    raw = subprocess.check_output(command, text=True, encoding="utf-8", errors="replace")
    commits = defaultdict(list)
    for record in raw.split("\x1e"):
        parts = record.strip().split("\x1f")
        if len(parts) != 4:
            continue
        sha, stamp, author, subject = parts
        local_day = datetime.fromisoformat(stamp).astimezone(ZONE).date()
        if since <= local_day <= through:
            commits[local_day.isoformat()].append({"sha": sha, "author": author, "subject": subject})
    return commits


def _github_get(url: str, token: str):
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "hth-work-summary",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            error = exc
        except (TimeoutError, ConnectionError, urllib.error.URLError) as exc:
            error = exc
        if attempt == 3:
            raise RuntimeError(f"GitHub API request failed after 4 attempts: {url}") from error
        delay = min(30, 2 ** (attempt + 1))
        if isinstance(error, urllib.error.HTTPError):
            retry_after = error.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = min(30, max(delay, int(retry_after)))
        print(f"GitHub API transient failure ({type(error).__name__}); retrying {url} in {delay}s", flush=True)
        time.sleep(delay)


def collect_github(repository: str, release_repositories: list[str], since: date,
                   through: date, output: Path, token: str = "") -> Path:
    """Collect GitHub's observed run/release facts, not CPU or GPT estimates."""
    if not repository or "/" not in repository:
        raise ValueError("A GitHub owner/repository is required")
    runs = []
    query = urllib.parse.urlencode({"per_page": 100, "created": f"{since}..{through}"})
    for page in range(1, 101):
        payload = _github_get(f"https://api.github.com/repos/{repository}/actions/runs?{query}&page={page}", token)
        batch = payload.get("workflow_runs", [])
        runs.extend({"id": item["id"], "name": item.get("name", ""), "status": item.get("status", ""),
                     "conclusion": item.get("conclusion"), "created_at": item.get("created_at"),
                     "run_started_at": item.get("run_started_at"), "updated_at": item.get("updated_at"),
                     "html_url": item.get("html_url")}
                    for item in batch)
        if len(batch) < 100:
            break
    else:
        raise RuntimeError("GitHub run pagination exceeded 10,000 records; narrow the date range")
    releases = []
    for release_repo in dict.fromkeys(release_repositories):
        if "/" not in release_repo:
            raise ValueError(f"Invalid release repository: {release_repo}")
        for page in range(1, 101):
            batch = _github_get(f"https://api.github.com/repos/{release_repo}/releases?per_page=100&page={page}", token)
            releases.extend({"repository": release_repo, "tag": item.get("tag_name", ""),
                             "name": item.get("name", ""), "published_at": item.get("published_at"),
                             "html_url": item.get("html_url")}
                            for item in batch if (published := _local_day(item.get("published_at")))
                            and since.isoformat() <= published <= through.isoformat())
            # The releases endpoint is newest-first. Stop paging as soon as the
            # returned page reaches history already sealed in day snapshots.
            published_days = [_local_day(item.get("published_at")) for item in batch if item.get("published_at")]
            if len(batch) < 100 or published_days and min(published_days) < since.isoformat():
                break
        else:
            raise RuntimeError(f"GitHub release pagination exceeded 10,000 records: {release_repo}")
    _write(output, {"schema": SCHEMA, "repository": repository, "since": since.isoformat(),
                    "through": through.isoformat(), "runs": runs, "releases": releases})
    return output


def pending_since(results_root: Path, through: date, start: date = START) -> date:
    for day in _days(start, through):
        cached = _json(results_root / "reports" / "hth-work-summary" / "days" / f"{day}.json", {})
        if day == through or cached.get("closed") is not True:
            return day
    return through


def _local_day(stamp: str | None):
    if not stamp:
        return None
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(ZONE).date().isoformat()


def _headline(subject: str) -> bool:
    text = subject.lower()
    return not any(word in text for word in ("merge ", "wip", "typo", "minor", "test fix", "formatting"))


def _estimate_cost(chatgpt, codex, model):
    # Dollar figures are an API-equivalent scenario, never actual subscription
    # charges. Rates and token intensity are explicit report assumptions.
    hours = _sum_ranges([chatgpt, codex])
    if hours is None or not model.get("enabled", False):
        return None
    input_tokens = _range(model["input_tokens_per_assistant_hour"])
    output_tokens = _range(model["output_tokens_per_assistant_hour"])
    input_rate = _range(model["input_usd_per_million_tokens"])
    output_rate = _range(model["output_usd_per_million_tokens"])
    return [round(hours[0] * (input_tokens[0] * input_rate[0] + output_tokens[0] * output_rate[0]) / 1e6, 3),
            round(hours[1] * (input_tokens[1] * input_rate[1] + output_tokens[1] * output_rate[1]) / 1e6, 3)]


def _run_wall_hours(run):
    if run.get("status") != "completed":
        return None
    try:
        start = datetime.fromisoformat((run.get("run_started_at") or run["created_at"]).replace("Z", "+00:00"))
        finish = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError):
        return None
    duration = (finish - start).total_seconds() / 3600
    return round(duration, 4) if 0 <= duration <= 48 else None


def _day_record(day: date, as_of: date, commits, github, ledger, shares, cost_model):
    key = day.isoformat()
    month = key[:7]
    manual = ledger.get("days", {}).get(key, {})
    monthly = ledger.get("historical_months", {}).get(month, {})
    day_commits = commits.get(key, [])
    day_runs = [run for run in github.get("runs", []) if _local_day(run.get("created_at")) == key]
    day_releases = [release for release in github.get("releases", []) if _local_day(release.get("published_at")) == key]
    wall_hours = round(sum(_run_wall_hours(run) or 0 for run in day_runs), 4)
    estimates = {field: _range(manual[field]) if field in manual else None for field in FIELDS}
    for field in ("human_hours", "chatgpt_hours", "codex_hours", "compute_core_hours"):
        if estimates[field] is None and monthly.get(field) is not None and shares.get(month, {}).get(key) is not None:
            estimates[field] = _scaled(_range(monthly[field]), shares[month][key])
    if not monthly:
        model = ledger.get("daily_commit_estimate", {})
        if day_commits and estimates["human_hours"] is None:
            per_commit = _range(model["human_hours_per_commit"])
            cap = float(model["human_hours_daily_cap"])
            estimates["human_hours"] = [min(cap, per_commit[0] * len(day_commits)),
                                         min(cap, per_commit[1] * len(day_commits))]
        for assistant, ratio_name in (("chatgpt_hours", "chatgpt_to_human_hours"),
                                      ("codex_hours", "codex_to_human_hours")):
            if estimates[assistant] is None and estimates["human_hours"] is not None:
                ratio = _range(model[ratio_name])
                estimates[assistant] = [round(estimates["human_hours"][0] * ratio[0], 3),
                                        round(estimates["human_hours"][1] * ratio[1], 3)]
        if estimates["compute_core_hours"] is None and wall_hours:
            cores = _range(model["runner_cores_per_workflow_wall_hour"])
            estimates["compute_core_hours"] = [round(wall_hours * cores[0], 3),
                                               round(wall_hours * cores[1], 3)]
    if estimates["chatgpt_cost_usd"] is None:
        estimates["chatgpt_cost_usd"] = _estimate_cost(estimates["chatgpt_hours"], None, cost_model)
    if estimates["codex_cost_usd"] is None:
        estimates["codex_cost_usd"] = _estimate_cost(None, estimates["codex_hours"], cost_model)
    if estimates["gpt_cost_usd"] is None:
        estimates["gpt_cost_usd"] = _sum_ranges([estimates["chatgpt_cost_usd"], estimates["codex_cost_usd"]])
    return {"schema": SCHEMA, "date": key, "closed": day < as_of, "timezone": str(ZONE), "estimate_ranges": estimates,
            "estimate_basis": "explicit daily ledger" if manual else "monthly historical range allocated by commit count" if monthly else "daily Git/Actions proxy; unknown where evidence absent",
            "commits": day_commits, "workflow_runs": day_runs, "releases": day_releases,
            "workflow_wall_hours_proxy": wall_hours,
            "sources": {"git": bool(day_commits), "github_snapshot": bool(github), "monthly_reconstruction": bool(monthly)}}


def _month_record(month: str, days: list[dict], ledger: dict, cost_model: dict, closed: bool):
    historical = ledger.get("historical_months", {}).get(month, {})
    estimates = {}
    for field in FIELDS:
        estimates[field] = _range(historical[field]) if field in historical else _sum_ranges(
            day["estimate_ranges"][field] for day in days)
    if historical:
        if "chatgpt_cost_usd" not in historical:
            estimates["chatgpt_cost_usd"] = _estimate_cost(estimates["chatgpt_hours"], None, cost_model)
        if "codex_cost_usd" not in historical:
            estimates["codex_cost_usd"] = _estimate_cost(None, estimates["codex_hours"], cost_model)
        if "gpt_cost_usd" not in historical:
            estimates["gpt_cost_usd"] = _sum_ranges([estimates["chatgpt_cost_usd"], estimates["codex_cost_usd"]])
    authors = Counter(commit["author"] for day in days for commit in day["commits"])
    highlights = [{"subject": commit["subject"], "sha": commit["sha"]}
                  for day in reversed(days) for commit in day["commits"] if _headline(commit["subject"])][:8]
    return {"schema": SCHEMA, "month": month, "closed": closed, "estimate_ranges": estimates,
            "estimate_basis": historical.get("source", "sum of daily recorded/modelled ranges"),
            "commits": sum(len(day["commits"]) for day in days), "contributors": dict(authors),
            "workflow_runs": sum(len(day["workflow_runs"]) for day in days),
            "workflow_wall_hours_proxy": round(sum(day.get("workflow_wall_hours_proxy", 0) for day in days), 3),
            "completed_workflow_runs": sum(run["status"] == "completed" for day in days for run in day["workflow_runs"]),
            "successful_workflow_runs": sum(run.get("conclusion") == "success" for day in days for run in day["workflow_runs"]),
            "failed_workflow_runs": sum(run.get("conclusion") == "failure" for day in days for run in day["workflow_runs"]),
            "workflow_names": dict(Counter(run.get("name") or "unnamed" for day in days for run in day["workflow_runs"])),
            "releases": [release for day in days for release in day["releases"]],
            "highlights": highlights, "days": [day["date"] for day in days]}


def _year_record(year: str, months: list[dict], closed: bool):
    contributors = Counter()
    workflow_names = Counter()
    for month in months:
        contributors.update(month["contributors"])
        workflow_names.update(month["workflow_names"])
    return {"schema": SCHEMA, "year": year, "closed": closed,
            "estimate_ranges": {field: _sum_ranges(month["estimate_ranges"][field] for month in months)
                                for field in FIELDS},
            "commits": sum(month["commits"] for month in months),
            "contributors": dict(contributors),
            "workflow_runs": sum(month["workflow_runs"] for month in months),
            "completed_workflow_runs": sum(month["completed_workflow_runs"] for month in months),
            "successful_workflow_runs": sum(month["successful_workflow_runs"] for month in months),
            "failed_workflow_runs": sum(month["failed_workflow_runs"] for month in months),
            "workflow_wall_hours_proxy": round(sum(month["workflow_wall_hours_proxy"] for month in months), 3),
            "workflow_names": dict(workflow_names),
            "releases": sum(len(month["releases"]) for month in months),
            "months": [month["month"] for month in months]}


def _fmt(value, unit="h"):
    return "unknown" if value is None else f"{value[0]:,.1f}–{value[1]:,.1f} {unit}"


def _effort_range(estimates):
    parts = [estimates[field] for field in EFFORT_FIELDS]
    return _sum_ranges(parts) if all(part is not None for part in parts) else None


def _effort_point(value, unit="activity-h"):
    return "-" if value is None else f"{(value[0] + value[1]) / 2:,.1f} {unit}"


def _render(summary: dict, years: list[dict], months: list[dict]) -> str:
    totals = summary["estimate_ranges"]
    lines = ["# HTH Work Summary", "", f"As of **{summary['as_of']}** (America/Chicago).",
             "", "## Lifetime", "", "| Measure | Estimate / observed count | Effort |", "|---|---:|---:|",
             f"| Dan Stupka / human effort | {_fmt(totals['human_hours'])} | {_effort_point(totals['human_hours'], 'h')} |",
             f"| ChatGPT active time | {_fmt(totals['chatgpt_hours'])} | {_effort_point(totals['chatgpt_hours'], 'h')} |",
             f"| Codex active time | {_fmt(totals['codex_hours'])} | {_effort_point(totals['codex_hours'], 'h')} |",
             f"| Compute | {_fmt(totals['compute_core_hours'], 'core-h')} | {_effort_point(totals['compute_core_hours'], 'core-h')} |",
             f"| Combined activity | {_fmt(_effort_range(totals), 'activity-h')} | {_effort_point(_effort_range(totals))} |",
             f"| ChatGPT API-equivalent cost scenario | {_fmt(totals['chatgpt_cost_usd'], 'USD')} | — |",
             f"| Codex API-equivalent cost scenario | {_fmt(totals['codex_cost_usd'], 'USD')} | — |",
             f"| GPT API-equivalent cost scenario | {_fmt(totals['gpt_cost_usd'], 'USD')} | — |",
             f"| Git commits | {summary['commits']:,} | — |",
             f"| GitHub workflow runs captured | {summary['workflow_runs']:,} | — |",
             f"| Succeeded / failed workflow runs | {summary['successful_workflow_runs']:,} / {summary['failed_workflow_runs']:,} | — |",
             f"| Workflow wall-time proxy | {summary['workflow_wall_hours_proxy']:,.1f} h | — |",
             f"| GitHub releases captured | {summary['releases']:,} | — |",
             f"| CBE build records (current lifecycle ledger) | {summary['cbe_build_records'] if summary['cbe_build_records'] is not None else 'unavailable'} | — |",
             f"| CBE cache elements / release elements | {summary['cbe_cache_elements'] if summary['cbe_cache_elements'] is not None else 'unavailable'} / {summary['cbe_release_elements'] if summary['cbe_release_elements'] is not None else 'unavailable'} | — |",
             "", "## Annual", "",
             "| Year | Human h | ChatGPT h | Codex h | Compute core-h | GPT cost scenario | Effort | Commits | Runs | Releases |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for year in years:
        e = year["estimate_ranges"]
        label = year["year"] if year["closed"] else f"{year['year']} YTD"
        lines.append("| " + " | ".join([label, _fmt(e["human_hours"]), _fmt(e["chatgpt_hours"]),
            _fmt(e["codex_hours"]), _fmt(e["compute_core_hours"], "core-h"),
            _fmt(e["gpt_cost_usd"], "USD"), _effort_point(_effort_range(e)),
            str(year["commits"]), str(year["workflow_runs"]),
            str(year["releases"])]) + " |")
    lines.extend(["", "## Monthly", "",
             "| Month | Human h | ChatGPT h | Codex h | Compute core-h | GPT cost scenario | Effort | Commits | Runs | Releases |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for month in months:
        e = month["estimate_ranges"]
        lines.append("| " + " | ".join([month["month"], _fmt(e["human_hours"]), _fmt(e["chatgpt_hours"]),
            _fmt(e["codex_hours"]), _fmt(e["compute_core_hours"], "core-h"),
            _fmt(e["gpt_cost_usd"], "USD"), _effort_point(_effort_range(e)),
            str(month["commits"]), str(month["workflow_runs"]),
            str(len(month["releases"]))]) + " |")
    lines.extend(["", "## Daily", "", "The durable JSON files in `reports/hth-work-summary/days/` contain each day's hours, source links and activity. Closed days are reused; only an explicit refresh recalculates them.", "",
                  "| Day | Human h | ChatGPT h | Codex h | Compute core-h | GPT cost scenario | Effort | Commits | Runs |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for day in summary["recent_days"]:
        e = day["estimate_ranges"]
        lines.append("| " + " | ".join([day["date"], _fmt(e["human_hours"]), _fmt(e["chatgpt_hours"]),
            _fmt(e["codex_hours"]), _fmt(e["compute_core_hours"], "core-h"), _fmt(e["gpt_cost_usd"], "USD"),
            _effort_point(_effort_range(e)),
            str(len(day["commits"])), str(len(day["workflow_runs"]))]) + " |")
    if summary["workflow_names"]:
        lines.extend(["", "## Build activity by workflow", "", "| Workflow | Runs |", "|---|---:|"])
        for name, count in sorted(summary["workflow_names"].items(), key=lambda item: (-item[1], item[0])):
            safe_name = name.replace("|", "&#124;")
            lines.append(f"| {safe_name} | {count:,} |")
    lines.extend(["", "## Contributors and feature highlights", "",
                  "Confirmed human contributors (project ledger):", ""])
    for person in summary["people"]:
        lines.append(f"- {person['name']} — {person['role']}")
    lines.extend(["", "Git authorship counts (including any automation accounts) are separate from confirmed people:", ""])
    for author, count in summary["contributors"].items():
        lines.append(f"- {author}: {count:,} commits")
    for month in reversed(months):
        if month["highlights"] or month["releases"]:
            lines.append(f"\n### {month['month']}")
            for release in month["releases"][:5]:
                lines.append(f"- Release: [{release['tag']}]({release['html_url']}) ({release['repository']})")
            for commit in month["highlights"][:5]:
                lines.append(f"- {commit['subject']} ([{commit['sha'][:8]}](https://github.com/{summary['repository']}/commit/{commit['sha']}))")
    lines.extend(["", "## Evidence and caveats", "",
                  "- Effort is the midpoint of the summed low/high ranges for human, ChatGPT, Codex, and compute hours. It is a mixed activity-hour indicator, **not** person-hours, elapsed time, billable labor, or a cost. A period's Effort shows `-` if any component is unknown; the component ranges remain the primary evidence.",
                  f"- Historical June–September 2026 human and compute ranges come from [Project Juana CRT/DRT-9]({summary['historical_source']}); they are reconstructions, not time sheets or CPU counters.",
                  "- Historical monthly ranges are allocated across Git-active days in proportion to commit count solely to create daily estimates. A missing day is not proof of no work. June predates this repository's Git history and remains unallocated by day.",
                  "- ChatGPT/Codex active-time ranges are planning assumptions in the versioned estimate ledger, not observed session durations. Overlapping human and assistant hours must not be added into a single labor total.",
                  "- GPT cost is an illustrative API-equivalent token/rate scenario from the ledger, not billed subscription, credit, or API spend. [OpenAI distinguishes included usage, purchased credits, and API billing](https://help.openai.com/en/articles/12642688-using-credits-for-flexible-usage-in-chatgpt-personal-plans); actual cost requires account usage/billing records.",
                  f"- GPT scenario assumptions: {summary['gpt_cost_scenario']}. These are user-editable bounds, not a model-specific published price quote.",
                  "- Workflow-run counts are GitHub Actions records, not necessarily successful builds. Compute core-hours are historical estimates unless explicit daily evidence is supplied. CBE build records are the current lifecycle count, not a lifetime execution count.",
                  "- New-day Git commit counts and Actions wall time feed broad proxy ranges until actual human/assistant/CPU telemetry is supplied in the daily ledger. Workflow wall time is not CPU time; runner core allocation and utilization are unknown.",
                  f"- Cache: {summary['cache']['days_reused']} closed days reused; {summary['cache']['days_built']} days built; {summary['cache']['months_reused']} closed months reused; {summary['cache']['months_built']} months built; {summary['cache']['years_reused']} closed years reused; {summary['cache']['years_built']} years built.", ""])
    return "\n".join(lines)


def generate(repo: Path, results_root: Path, output_dir: Path, as_of: date,
             ledger_path: Path, github_path: Path | None = None, refresh_history: bool = False) -> dict:
    ledger = _json(ledger_path, {})
    github = _json(github_path, {}) if github_path else {}
    if github and github.get("schema") != SCHEMA:
        raise ValueError("Unsupported GitHub work-summary snapshot")
    if as_of < START:
        raise ValueError("Report date precedes project start")
    base = results_root / "reports" / "hth-work-summary"
    day_paths = {day.isoformat(): base / "days" / f"{day}.json" for day in _days(START, as_of)}
    existing = {key: cached for key, path in day_paths.items()
                if not refresh_history and key != as_of.isoformat()
                and (cached := _json(path, {})).get("closed") is True}
    for key, value in existing.items():
        if value.get("schema") != SCHEMA or value.get("date") != key:
            raise ValueError(f"Invalid cached work-summary day: {key}")
    missing = [date.fromisoformat(key) for key in day_paths if key not in existing]
    commits = _git_commits(repo, min(missing), as_of) if missing else {}
    # Monthly reconstructions are distributed only over observed Git-active days.
    weights = defaultdict(dict)
    for key, records in commits.items():
        weights[key[:7]][key] = len(records)
    for key, record in existing.items():
        weights[key[:7]][key] = len(record["commits"])
    shares = {month: {key: count / sum(days.values()) for key, count in days.items()} for month, days in weights.items() if sum(days.values())}
    cost_model = ledger.get("gpt_cost_scenario", {})
    day_records = []
    for key, path in day_paths.items():
        day = existing.get(key)
        if day is None:
            day = _day_record(date.fromisoformat(key), as_of, commits, github, ledger, shares, cost_model)
            _write(output_dir / "days" / f"{key}.json", day)
        day_records.append(day)
    by_month = defaultdict(list)
    for day in day_records:
        by_month[day["date"][:7]].append(day)
    month_records = []
    months_reused = 0
    months_built = 0
    for month, days in sorted(by_month.items()):
        cached = base / "months" / f"{month}.json"
        closed = _month_range(month)[1] < as_of
        record = _json(cached) if closed and not refresh_history else None
        if record is not None and record.get("closed") is not True:
            record = None
        if record is not None:
            if record.get("schema") != SCHEMA or record.get("month") != month:
                raise ValueError(f"Invalid cached work-summary month: {month}")
            months_reused += 1
        else:
            record = _month_record(month, days, ledger, cost_model, closed)
            _write(output_dir / "months" / f"{month}.json", record)
            months_built += 1
        month_records.append(record)
    by_year = defaultdict(list)
    for month in month_records:
        by_year[month["month"][:4]].append(month)
    year_records = []
    years_reused = 0
    years_built = 0
    for year, months in sorted(by_year.items()):
        cached = base / "years" / f"{year}.json"
        closed = date(int(year), 12, 31) < as_of
        record = _json(cached) if closed and not refresh_history else None
        if record is not None and record.get("closed") is not True:
            record = None
        if record is not None:
            if record.get("schema") != SCHEMA or record.get("year") != year:
                raise ValueError(f"Invalid cached work-summary year: {year}")
            years_reused += 1
        else:
            record = _year_record(year, months, closed)
            _write(output_dir / "years" / f"{year}.json", record)
            years_built += 1
        year_records.append(record)
    lifecycle = _json(results_root / "metadata" / "resource-lifecycle.json", {})
    cbe_count = lifecycle.get("summary", {}).get("build_records")
    workflow_names = Counter()
    for month in month_records:
        workflow_names.update(month.get("workflow_names", {}))
    summary = {"schema": SCHEMA, "as_of": as_of.isoformat(), "repository": ledger.get("repository", "dlstupka/hth"),
               "historical_source": ledger.get("historical_source", ""),
               "people": ledger.get("people", []), "gpt_cost_scenario": cost_model,
               "estimate_ranges": {field: _sum_ranges(year["estimate_ranges"][field] for year in year_records) for field in FIELDS},
               "commits": sum(month["commits"] for month in month_records),
               "workflow_runs": sum(month["workflow_runs"] for month in month_records),
               "successful_workflow_runs": sum(month.get("successful_workflow_runs", 0) for month in month_records),
               "failed_workflow_runs": sum(month.get("failed_workflow_runs", 0) for month in month_records),
               "workflow_names": dict(workflow_names),
               "workflow_wall_hours_proxy": round(sum(month.get("workflow_wall_hours_proxy", 0) for month in month_records), 3),
               "releases": sum(len(month["releases"]) for month in month_records),
               "cbe_build_records": cbe_count,
               "cbe_cache_elements": lifecycle.get("summary", {}).get("cache_elements"),
               "cbe_release_elements": lifecycle.get("summary", {}).get("release_elements"),
               "contributors": dict(Counter({name: sum(month["contributors"].get(name, 0) for month in month_records)
                                               for month in month_records for name in month["contributors"]})),
               "recent_days": list(reversed(day_records[-31:])),
               "cache": {"days_reused": len(existing), "days_built": len(missing),
                         "months_reused": months_reused, "months_built": months_built,
                         "years_reused": years_reused, "years_built": years_built}}
    _write(output_dir / "summary.json", summary)
    report = output_dir / "summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_render(summary, year_records, month_records), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pending = commands.add_parser("pending-since")
    pending.add_argument("--results-root", type=Path, required=True)
    pending.add_argument("--as-of", type=date.fromisoformat, default=datetime.now(ZONE).date())
    collect = commands.add_parser("collect-github")
    collect.add_argument("--repository", required=True)
    collect.add_argument("--release-repository", action="append", default=[])
    collect.add_argument("--since", type=date.fromisoformat, required=True)
    collect.add_argument("--through", type=date.fromisoformat, required=True)
    collect.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("generate")
    build.add_argument("--repository-root", type=Path, required=True)
    build.add_argument("--results-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--ledger", type=Path, required=True)
    build.add_argument("--github-snapshot", type=Path)
    build.add_argument("--as-of", type=date.fromisoformat, default=datetime.now(ZONE).date())
    build.add_argument("--refresh-history", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "pending-since":
        print(pending_since(args.results_root, args.as_of))
    elif args.command == "collect-github":
        collect_github(args.repository, args.release_repository or [args.repository], args.since,
                       args.through, args.output, os.getenv("GITHUB_TOKEN", ""))
        print(args.output)
    else:
        summary = generate(args.repository_root, args.results_root, args.output_dir, args.as_of,
                           args.ledger, args.github_snapshot, args.refresh_history)
        print(json.dumps(summary["cache"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
