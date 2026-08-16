#!/usr/bin/env python3
"""Capture and persist aggregate multi-detector execution telemetry."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hth.domain.multidetector_schedule import workload_class

INDEX_SCHEMA_VERSION = 1
OBSERVATION_SCHEMA_VERSION = 1
MAX_OBSERVATIONS = 200


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite timestamp: {text}")
    return value


def _read_worker(path: Path) -> dict[str, Any]:
    events: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        kind, timestamp = line.split("\t", 1)
        events[kind] = _float(timestamp)
    start, end = events.get("start"), events.get("end")
    if start is None or end is None:
        raise ValueError(f"Incomplete worker telemetry: {path}")
    return {"pipeline": int(path.stem), "started_epoch": start, "final_idle_epoch": end, "span_seconds": max(0.0, end-start)}


def _read_task(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"task_index": int(path.stem)}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if parts[0] == "claim":
            _, ts, pipeline, detector, shard_index, shard_count, threads = parts
            row.update({"claimed_epoch": _float(ts), "pipeline": int(pipeline), "detector": detector, "shard_index": int(shard_index), "shard_count": int(shard_count), "allocated_threads": int(threads)})
        elif parts[0] == "start":
            row["started_epoch"] = _float(parts[1])
        elif parts[0] == "finish":
            row["finished_epoch"] = _float(parts[1]); row["status"] = parts[2]
    required = {"claimed_epoch", "started_epoch", "finished_epoch"}
    if not required.issubset(row):
        raise ValueError(f"Incomplete task telemetry: {path}")
    row["claim_wait_seconds"] = max(0.0, row["started_epoch"]-row["claimed_epoch"])
    row["busy_seconds"] = max(0.0, row["finished_epoch"]-row["started_epoch"])
    return row


def _active_timeline(tasks: list[dict[str, Any]]) -> tuple[dict[str, float], float, dict[str, float]]:
    events=[]
    for task in tasks:
        events += [(float(task["started_epoch"]), +1), (float(task["finished_epoch"]), -1)]
    events.sort(key=lambda x:(x[0], -x[1]))
    if not events:
        return {}, 0.0, {}
    active=0; prior=events[0][0]; segments=[]; by_active={}
    for when, delta in events:
        if when>prior:
            dur=when-prior; by_active[str(active)] = by_active.get(str(active),0.0)+dur; segments.append((prior,when,active))
        active += delta; prior=when
    peak=max((s[2] for s in segments), default=0); tail={}; total=0.0
    for start,end,count in reversed(segments):
        if peak>0 and count>=peak:
            break
        if count>0:
            dur=end-start; tail[str(count)] = tail.get(str(count),0.0)+dur; total += dur
    return by_active,total,tail


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    telemetry=args.telemetry_root
    batch=dict(line.split("\t",1) for line in (telemetry/"batch.tsv").read_text(encoding="utf-8").splitlines())
    batch_start, batch_end = _float(batch["start"]), _float(batch["end"]); makespan=max(0.0,batch_end-batch_start)
    workers=[_read_worker(p) for p in sorted((telemetry/"workers").glob("*.tsv"))]
    tasks=[_read_task(p) for p in sorted((telemetry/"tasks").glob("*.tsv"))]
    busy_by_worker={w["pipeline"]:0.0 for w in workers}
    for task in tasks: busy_by_worker[task["pipeline"]]=busy_by_worker.get(task["pipeline"],0.0)+task["busy_seconds"]
    for worker in workers:
        busy=busy_by_worker.get(worker["pipeline"],0.0); worker["busy_seconds"]=busy; worker["idle_seconds"]=max(0.0,worker["span_seconds"]-busy); worker["utilization"]=0.0 if worker["span_seconds"]<=0 else busy/worker["span_seconds"]
    total_busy=sum(float(t["busy_seconds"]) for t in tasks); worker_count=max(1,len(workers)); util=0.0 if makespan<=0 else total_busy/(worker_count*makespan)
    active_seconds, final_tail, tail_by_active = _active_timeline(tasks)
    observation={
        "schema_version":OBSERVATION_SCHEMA_VERSION, "observation_id":args.observation_id,
        "observed_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "github_run_id":args.github_run_id, "github_run_number":args.github_run_number,
        "mode":args.mode, "strategy":args.strategy, "parameter_set_limit":args.limit or None,
        "workload_class":workload_class(args.mode,args.strategy,args.limit),
        "detector_count":args.detector_count, "task_count":len(tasks), "golden_set_sha256":args.golden_set_sha256,
        "runner_label":args.runner_label, "runner_name":args.runner_name, "runner_thread_budget":args.runner_thread_budget,
        "worker_count":worker_count, "threads_per_worker":args.threads_per_worker, "allocated_threads":args.allocated_threads,
        "loading_strategy":args.loading_strategy, "scheduler_source":args.scheduler_source,
        "batch_started_epoch":batch_start, "batch_finished_epoch":batch_end, "makespan_seconds":makespan,
        "total_worker_busy_seconds":total_busy, "total_worker_idle_seconds":max(0.0,worker_count*makespan-total_busy),
        "worker_utilization":util, "active_worker_seconds":active_seconds, "final_tail_seconds":final_tail,
        "final_tail_seconds_by_active_workers":tail_by_active, "workers":workers, "tasks":tasks,
    }
    _write_json(args.output,observation); return observation


def publish(metadata: Path, results_root: Path) -> dict[str, Any]:
    observation=_read_json(metadata); path=results_root/"multidetector-index.json"
    index=_read_json(path) if path.is_file() else {"schema_version":INDEX_SCHEMA_VERSION,"observations":[]}
    by_id={str(r.get("observation_id")):r for r in index.get("observations",[]) if isinstance(r,dict) and r.get("observation_id")}
    by_id[str(observation["observation_id"])]=observation
    rows=sorted(by_id.values(),key=lambda r:str(r.get("observed_at_utc") or ""),reverse=True)[:MAX_OBSERVATIONS]
    index.update({"schema_version":INDEX_SCHEMA_VERSION,"updated_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"observations":rows})
    _write_json(path,index); return index


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    f=sub.add_parser("finalize"); f.add_argument("--telemetry-root",type=Path,required=True); f.add_argument("--output",type=Path,required=True); f.add_argument("--observation-id",required=True); f.add_argument("--github-run-id",default=""); f.add_argument("--github-run-number",default=""); f.add_argument("--mode",required=True); f.add_argument("--strategy",required=True); f.add_argument("--limit",default=""); f.add_argument("--detector-count",type=int,required=True); f.add_argument("--golden-set-sha256",required=True); f.add_argument("--runner-label",required=True); f.add_argument("--runner-name",required=True); f.add_argument("--runner-thread-budget",type=int,required=True); f.add_argument("--threads-per-worker",type=int,required=True); f.add_argument("--allocated-threads",type=int,required=True); f.add_argument("--loading-strategy",required=True); f.add_argument("--scheduler-source",required=True)
    q=sub.add_parser("publish"); q.add_argument("--metadata",type=Path,required=True); q.add_argument("--results-root",type=Path,required=True); return p


def main(argv: list[str] | None=None) -> int:
    args=parser().parse_args(argv)
    if args.command=="finalize":
        obs=finalize(args); print(json.dumps({k:obs[k] for k in ("makespan_seconds","worker_utilization","final_tail_seconds")},sort_keys=True)); return 0
    publish(args.metadata,args.results_root); return 0

if __name__=="__main__": raise SystemExit(main())
