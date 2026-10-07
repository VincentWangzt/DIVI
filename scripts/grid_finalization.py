from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


CONFIG_HASH_IGNORED_KEYS = {
    "config_path",
    "cuda_visible_devices",
    "device",
    "output",
}


@dataclass(frozen=True)
class FinalizationJob:
    repo_root: str
    finalize_events_path: str
    debug_events_path: str | None
    run_id: str
    run_status: str
    attempt: int
    seed: int | None
    method: str
    method_slug: str
    target: str
    target_slug: str
    runner_type: str
    gpu: int | None
    config_path: str
    config_hash: str
    config_hash_version: str
    config_hash_basis: str
    command: list[str]
    started_at: str | None
    finished_at: str | None
    start_wall_time: float | None
    end_wall_time: float | None
    duration_sec: float | None
    exit_code: int | None
    result_path: str | None
    tb_dir: str
    console_log: str | None
    training_failure_reason: str | None = None


@dataclass(frozen=True)
class FinalizeResult:
    status: str
    run_id: str
    run_status: str
    attempt: int
    finalize_started_at: str
    finalize_finished_at: str
    finalize_duration_sec: float
    failure_reason: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def repo_path(repo_root: Path, path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def relpath(repo_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def append_jsonl(path: Path | str, event: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def append_debug_event(debug_path: str | None, event_type: str, payload: dict[str, Any]) -> None:
    if debug_path is None:
        return
    payload = dict(payload)
    payload_status = payload.pop("status", None)
    event = {
        "status": "debug",
        "event_type": event_type,
        "timestamp": utc_now(),
    }
    event.update(payload)
    if payload_status is not None:
        event["payload_status"] = payload_status
    append_jsonl(debug_path, event)


def latest_finalize_statuses(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for event in events:
        run_id = event.get("run_id")
        status = event.get("status")
        if run_id and status in {"finalize_completed", "finalize_failed"}:
            statuses[str(run_id)] = event
    return statuses


def finalize_attempt_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    attempts: dict[str, int] = {}
    for event in events:
        run_id = event.get("run_id")
        if run_id and event.get("status") in {"finalize_started", "finalize_failed", "finalize_completed"}:
            attempts[str(run_id)] = max(attempts.get(str(run_id), 0), int(event.get("attempt") or 0))
    return attempts


def _normalize_for_config_hash(payload: Any) -> Any:
    if isinstance(payload, dict):
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            key_str = str(key)
            if key_str in CONFIG_HASH_IGNORED_KEYS or key_str.endswith("_config_path"):
                continue
            normalized[key_str] = _normalize_for_config_hash(value)
        return normalized
    if isinstance(payload, list):
        return [_normalize_for_config_hash(value) for value in payload]
    if isinstance(payload, tuple):
        return [_normalize_for_config_hash(value) for value in payload]
    return payload


def artifact_config_hash(repo_root: Path, full_config_path: Path | str, config_hash_version: str) -> str:
    path = repo_path(repo_root, full_config_path)
    if path is None:
        raise ValueError("full_config_path is required")
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    encoded = json.dumps(
        {
            "config_hash_version": config_hash_version,
            "payload": _normalize_for_config_hash(payload),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def tail_file(path: Path | None, num_lines: int = 30) -> list[str]:
    if path is None or not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-num_lines:]


def infer_tb_path(repo_root: Path, result_path: Path, job: FinalizationJob) -> Path:
    tb_root = repo_path(repo_root, job.tb_dir)
    if tb_root is None:
        raise RuntimeError("missing TensorBoard root")
    return tb_root / job.runner_type / job.target / result_path.name


def run_extractor(repo_root: Path, tb_path: Path, job: FinalizationJob) -> subprocess.CompletedProcess[str]:
    started = time.perf_counter()
    append_debug_event(
        job.debug_events_path,
        "extractor_started",
        {
            "run_id": job.run_id,
            "gpu": job.gpu,
            "tb_path": relpath(repo_root, tb_path),
            "attempt": job.attempt,
        },
    )
    result = subprocess.run(
        [
            sys.executable,
            "utils/extract_tensorboard_run.py",
            str(tb_path),
            "--out-dir",
            str(tb_path / "extracted"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    append_debug_event(
        job.debug_events_path,
        "extractor_finished",
        {
            "run_id": job.run_id,
            "gpu": job.gpu,
            "tb_path": relpath(repo_root, tb_path),
            "attempt": job.attempt,
            "exit_code": result.returncode,
            "duration_sec": time.perf_counter() - started,
            "stdout_tail": result.stdout.splitlines()[-10:],
            "stderr_tail": result.stderr.splitlines()[-10:],
        },
    )
    return result


def build_finalization_job(
    entry: dict[str, Any],
    terminal_event: dict[str, Any],
    repo_root: Path,
    finalize_events_path: Path,
    debug_events_path: Path | None,
    attempt: int,
) -> FinalizationJob:
    return FinalizationJob(
        repo_root=str(repo_root),
        finalize_events_path=str(finalize_events_path),
        debug_events_path=None if debug_events_path is None else str(debug_events_path),
        run_id=str(entry["run_id"]),
        run_status=str(terminal_event.get("run_status") or terminal_event.get("status") or ""),
        attempt=attempt,
        seed=entry.get("seed"),
        method=str(entry.get("method", "")),
        method_slug=str(entry.get("method_slug", "")),
        target=str(entry.get("target", "")),
        target_slug=str(entry.get("target_slug", "")),
        runner_type=str(entry.get("runner_type", "")),
        gpu=terminal_event.get("gpu"),
        config_path=str(entry.get("config_path", "")),
        config_hash=str(entry.get("config_hash", "")),
        config_hash_version=str(entry.get("config_hash_version", "")),
        config_hash_basis=str(entry.get("config_hash_basis", "")),
        command=list(terminal_event.get("command") or entry.get("command_template") or []),
        started_at=terminal_event.get("started_at"),
        finished_at=terminal_event.get("finished_at"),
        start_wall_time=terminal_event.get("start_wall_time"),
        end_wall_time=terminal_event.get("end_wall_time"),
        duration_sec=terminal_event.get("duration_sec"),
        exit_code=terminal_event.get("exit_code"),
        result_path=terminal_event.get("result_path"),
        tb_dir=str(entry.get("tb_dir", "")),
        console_log=terminal_event.get("console_log"),
        training_failure_reason=terminal_event.get("failure_reason"),
    )


def pending_finalization_jobs(
    entries: list[dict[str, Any]],
    terminal_statuses: dict[str, dict[str, Any]],
    finalize_events: list[dict[str, Any]],
    repo_root: Path,
    finalize_events_path: Path,
    debug_events_path: Path | None,
    max_retries: int,
) -> list[FinalizationJob]:
    finalize_statuses = latest_finalize_statuses(finalize_events)
    attempts = finalize_attempt_counts(finalize_events)
    jobs: list[FinalizationJob] = []
    for entry in entries:
        run_id = str(entry["run_id"])
        terminal = terminal_statuses.get(run_id)
        if terminal is None:
            continue
        latest = finalize_statuses.get(run_id)
        if latest is not None and latest.get("status") == "finalize_completed":
            continue
        attempt = attempts.get(run_id, 0) + 1
        if attempt > max_retries:
            continue
        jobs.append(
            build_finalization_job(
                entry,
                terminal,
                repo_root,
                finalize_events_path,
                debug_events_path,
                attempt,
            )
        )
    return jobs


def finalize_job(job: FinalizationJob) -> FinalizeResult:
    repo_root = Path(job.repo_root)
    started_at = utc_now()
    started = time.perf_counter()
    append_jsonl(
        job.finalize_events_path,
        {
            "status": "finalize_started",
            "run_id": job.run_id,
            "run_status": job.run_status,
            "attempt": job.attempt,
            "gpu": job.gpu,
            "started_at": started_at,
        },
    )
    append_debug_event(
        job.debug_events_path,
        "finalize_started",
        {
            "run_id": job.run_id,
            "gpu": job.gpu,
            "attempt": job.attempt,
            "run_status": job.run_status,
            "result_path": job.result_path,
            "console_log": job.console_log,
        },
    )

    tb_path: Path | None = None
    run_log_path: Path | None = None
    extractor_stdout = ""
    extractor_stderr = ""
    artifact_hash = ""
    finalize_failure_reason: str | None = None
    finalize_traceback = ""

    try:
        result_path = repo_path(repo_root, job.result_path)
        if job.run_status == "completed":
            if result_path is None:
                raise RuntimeError("missing result path")
            tb_path = infer_tb_path(repo_root, result_path, job)
            run_log_path = result_path / "run.log"
            if not result_path.exists():
                raise RuntimeError(f"result path not found: {result_path}")
            if not tb_path.exists():
                raise RuntimeError(f"tb path not found: {tb_path}")
            if not run_log_path.exists():
                raise RuntimeError(f"run log not found: {run_log_path}")
            extractor = run_extractor(repo_root, tb_path, job)
            extractor_stdout = extractor.stdout
            extractor_stderr = extractor.stderr
            if extractor.returncode != 0:
                raise RuntimeError(f"extractor failed with exit code {extractor.returncode}")
            full_config_path = result_path / "full_config.yaml"
            if full_config_path.exists():
                artifact_hash = artifact_config_hash(repo_root, full_config_path, job.config_hash_version)
        else:
            result_path = repo_path(repo_root, job.result_path)
            if result_path is not None:
                run_log_path = result_path / "run.log"
                inferred_tb = infer_tb_path(repo_root, result_path, job)
                if inferred_tb.exists():
                    tb_path = inferred_tb
        status = "finalize_completed"
    except Exception as exc:
        status = "finalize_failed"
        finalize_failure_reason = repr(exc)
        finalize_traceback = traceback.format_exc()
        result_path = repo_path(repo_root, job.result_path)
        if result_path is not None and run_log_path is None:
            run_log_path = result_path / "run.log"

    finished_at = utc_now()
    duration_sec = time.perf_counter() - started
    event: dict[str, Any] = {
        "status": status,
        "run_id": job.run_id,
        "run_status": job.run_status,
        "attempt": job.attempt,
        "seed": job.seed,
        "method": job.method,
        "method_slug": job.method_slug,
        "target": job.target,
        "target_slug": job.target_slug,
        "runner_type": job.runner_type,
        "gpu": job.gpu,
        "config_path": job.config_path,
        "config_hash": job.config_hash,
        "config_hash_version": job.config_hash_version,
        "config_hash_basis": job.config_hash_basis,
        "artifact_config_hash": artifact_hash,
        "command": job.command,
        "training_started_at": job.started_at,
        "training_finished_at": job.finished_at,
        "start_wall_time": job.start_wall_time,
        "end_wall_time": job.end_wall_time,
        "duration_sec": job.duration_sec,
        "exit_code": job.exit_code,
        "result_path": relpath(repo_root, repo_path(repo_root, job.result_path)),
        "tb_path": relpath(repo_root, tb_path),
        "console_log": relpath(repo_root, repo_path(repo_root, job.console_log)),
        "run_log_tail": tail_file(run_log_path),
        "console_log_tail": tail_file(repo_path(repo_root, job.console_log)),
        "extractor_stdout": extractor_stdout,
        "extractor_stderr": extractor_stderr,
        "training_failure_reason": job.training_failure_reason,
        "finalize_failure_reason": finalize_failure_reason,
        "traceback": finalize_traceback,
        "finalize_started_at": started_at,
        "finalize_finished_at": finished_at,
        "finalize_duration_sec": duration_sec,
    }
    append_jsonl(job.finalize_events_path, event)
    append_debug_event(
        job.debug_events_path,
        "finalize_finished",
        {
            "run_id": job.run_id,
            "gpu": job.gpu,
            "attempt": job.attempt,
            "finalize_status": status,
            "run_status": job.run_status,
            "finalize_failure_reason": finalize_failure_reason,
            "duration_sec": duration_sec,
        },
    )
    return FinalizeResult(
        status=status,
        run_id=job.run_id,
        run_status=job.run_status,
        attempt=job.attempt,
        finalize_started_at=started_at,
        finalize_finished_at=finished_at,
        finalize_duration_sec=duration_sec,
        failure_reason=finalize_failure_reason,
    )
