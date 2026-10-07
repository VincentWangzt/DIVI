from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]

METHODS = ["sivi", "uivi", "aisivi", "divi", "ksivi"]
METHOD_LABELS = {
    "sivi": "SIVI",
    "uivi": "UIVI",
    "aisivi": "AISIVI",
    "divi": "DIVI",
    "ksivi": "KSIVI",
}

CORE_METHODS = ["sivi", "uivi", "aisivi", "divi"]
KERNEL_METHODS = ["ksivi"]
METHOD_GROUPS = {
    "all": METHODS,
    "core": CORE_METHODS,
    "kernel": KERNEL_METHODS,
}
METHOD_TO_GROUP = {
    **{method: "core" for method in CORE_METHODS},
    **{method: "kernel" for method in KERNEL_METHODS},
}

TOY_TARGETS = [
    "banana",
    "multimodal",
    "x_shaped",
    "student_uc",
    "8_gaussians",
]
LANGEVIN_TARGETS = ["Langevin_post"]
LR_TARGETS = ["LRwaveform"]
BNN_TARGETS = [
    "Bnn_boston",
    "Bnn_concrete",
    "Bnn_power",
    "Bnn_protein",
    "Bnn_winered",
    "Bnn_yacht",
]
TARGETS = TOY_TARGETS + LANGEVIN_TARGETS + LR_TARGETS + BNN_TARGETS
TARGET_GROUPS = {
    "all": TARGETS,
    "toy": TOY_TARGETS,
    "Langevin": LANGEVIN_TARGETS,
    "LRwaveform": LR_TARGETS,
    "BNN": BNN_TARGETS,
}
TARGET_TO_GROUP = {
    **{target: "toy" for target in TOY_TARGETS},
    **{target: "Langevin" for target in LANGEVIN_TARGETS},
    **{target: "LRwaveform" for target in LR_TARGETS},
    **{target: "BNN" for target in BNN_TARGETS},
}

REVERSE_RUNNERS = {"AISIVI", "DIVI"}

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ConfigError(Exception):
    pass


def _relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _query_value(query: dict[str, list[str]], name: str, default: str | None = None) -> str | None:
    values = query.get(name)
    if not values:
        return default
    return values[0]


def _query_list(query: dict[str, list[str]], name: str, default: list[str]) -> list[str]:
    raw = _query_value(query, name)
    if raw is None or not raw.strip():
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _validate_method(method: str) -> str:
    if method not in METHODS:
        raise ConfigError(f"Unknown method {method!r}.")
    return method


def _validate_target(target: str) -> str:
    if target not in TARGETS:
        raise ConfigError(f"Unknown target {target!r}.")
    return target


def _base_path(method: str, target: str) -> Path:
    return REPO_ROOT / "configs" / f"{method}_{target}.yaml"


def _read_yaml_config(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        return data, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _merge_nested_config(
    config: dict[str, Any],
    path_key: str,
    type_key: str,
    section_key: str,
    default_dir: str,
) -> tuple[dict[str, Any], str | None]:
    if path_key not in config:
        model_type = config.get(type_key)
        if not model_type:
            return config, None
        config[path_key] = f"configs/{default_dir}/{model_type}.yaml"

    nested_path = REPO_ROOT / str(config[path_key])
    nested_data, error = _read_yaml_config(nested_path)
    if error is not None:
        return config, f"{path_key}: {error}"
    if nested_data is None:
        return config, f"{path_key}: missing file {config[path_key]}"

    merged = OmegaConf.merge(
        {section_key: OmegaConf.create(nested_data)},
        OmegaConf.create(config),
    )
    return OmegaConf.to_container(merged, resolve=True), None


def _resolve_nested_configs(config: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    resolved = deepcopy(config)
    if "device" not in resolved:
        resolved["device"] = "cuda" if resolved.get("use_cuda", False) else "cpu"

    target_type = resolved.get("target_type")
    if target_type and "target_config_path" not in resolved:
        resolved["target_config_path"] = f"configs/targets/{target_type}.yaml"

    for args in [
        ("target_config_path", "target_type", "target", "targets"),
        ("vi_model_config_path", "vi_model_type", "vi_model", "vi_models"),
    ]:
        resolved, error = _merge_nested_config(resolved, *args)
        if error is not None:
            return resolved, error

    # UIVI uses HMC as its reverse model, merged into "hmc" key
    runner_type = str(resolved.get("runner_type", ""))
    if runner_type == "UIVI":
        reverse_path_key = "reverse_model_config_path"
        if reverse_path_key not in resolved:
            resolved[reverse_path_key] = "configs/reverse_models/HMC.yaml"
        nested_path = REPO_ROOT / str(resolved[reverse_path_key])
        nested_data, error = _read_yaml_config(nested_path)
        if error is not None:
            return resolved, f"{reverse_path_key}: {error}"
        if nested_data is None:
            return resolved, f"{reverse_path_key}: missing file {resolved[reverse_path_key]}"
        merged = OmegaConf.merge(
            {"hmc": OmegaConf.create(nested_data)},
            OmegaConf.create(resolved),
        )
        resolved = OmegaConf.to_container(merged, resolve=True)
    elif runner_type in REVERSE_RUNNERS:
        resolved, error = _merge_nested_config(
            resolved,
            "reverse_model_config_path",
            "reverse_model_type",
            "reverse_model",
            "reverse_models",
        )
        if error is not None:
            return resolved, error

    # Sync dimensions from vi_model into reverse_model
    reverse_model = resolved.get("reverse_model")
    vi_model = resolved.get("vi_model")
    if isinstance(reverse_model, dict) and isinstance(vi_model, dict):
        if "z_dim" in vi_model:
            reverse_model["z_dim"] = vi_model["z_dim"]
        if "epsilon_dim" in vi_model:
            reverse_model["epsilon_dim"] = vi_model["epsilon_dim"]

    return resolved, None


def _read_effective_config(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    config, error = _read_yaml_config(path)
    if error is not None or config is None:
        return config, error
    return _resolve_nested_configs(config)


def _raw_file(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    try:
        return path.read_text(encoding="utf-8"), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _yaml_text(data: dict[str, Any] | None) -> str | None:
    if data is None:
        return None
    return OmegaConf.to_yaml(OmegaConf.create(data), resolve=True)


def _config_payload(
    method: str,
    target: str,
    path: Path,
    data: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "id": f"{method}:{target}",
        "method": method,
        "method_label": METHOD_LABELS[method],
        "target": target,
        "target_group": TARGET_TO_GROUP[target],
        "path": _relpath(path),
        "display_path": _relpath(path),
        "exists": data is not None or path.exists(),
        "error": error,
        "data": data,
    }


def _load_config(method: str, target: str) -> dict[str, Any]:
    path = _base_path(method, target)
    data, error = _read_effective_config(path)
    return _config_payload(method, target, path, data, error)


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "data"}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    rows: dict[str, Any] = {}
    if isinstance(value, dict):
        if not value:
            rows[prefix or "<root>"] = {}
        for key in sorted(value):
            key_text = str(key)
            path = key_text if not prefix else f"{prefix}.{key_text}"
            rows.update(_flatten(value[key], path))
        return rows
    rows[prefix or "<root>"] = value
    return rows


def _value_signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _display_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(", ", ": "), default=str)


def _comparison_summary(configs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "config_count": len(configs),
        "row_count": len(rows),
        "changed_count": sum(1 for row in rows if row["changed"]),
        "missing_file_count": sum(1 for config in configs if config["data"] is None and config["error"] is None),
        "error_count": sum(1 for config in configs if config["error"] is not None),
    }


def _build_comparison(configs: list[dict[str, Any]]) -> dict[str, Any]:
    flattened_by_config: list[dict[str, Any]] = []
    all_paths: set[str] = set()

    for config in configs:
        flat = {} if config["data"] is None else _flatten(config["data"])
        flattened_by_config.append(flat)
        all_paths.update(flat.keys())

    rows = []
    for path in sorted(all_paths):
        cells = []
        signatures = []
        for config, flat in zip(configs, flattened_by_config):
            if config["data"] is None:
                status = "missing_file" if config["error"] is None else "error"
                value = None
                signature = f"__{status}__"
            elif path not in flat:
                status = "missing_key"
                value = None
                signature = "__missing_key__"
            else:
                status = "present"
                value = flat[path]
                signature = _value_signature(value)
            signatures.append(signature)
            cells.append(
                {
                    "config_id": config["id"],
                    "status": status,
                    "value": value,
                    "display": "" if status != "present" else _display_value(value),
                }
            )
        rows.append(
            {
                "path": path,
                "depth": 0 if path == "<root>" else path.count("."),
                "key": path.rsplit(".", 1)[-1],
                "changed": len(set(signatures)) > 1,
                "cells": cells,
            }
        )

    return {
        "configs": [_public_config(config) for config in configs],
        "rows": rows,
        "summary": _comparison_summary(configs, rows),
    }


def _metadata() -> dict[str, Any]:
    methods = [
        {"name": method, "label": METHOD_LABELS[method], "group": METHOD_TO_GROUP[method]}
        for method in METHODS
    ]
    targets = [
        {"name": target, "label": target, "group": TARGET_TO_GROUP[target]}
        for target in TARGETS
    ]

    availability: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        availability[method] = {}
        for target in TARGETS:
            base_path = _base_path(method, target)
            availability[method][target] = {
                "exists": base_path.exists(),
                "path": _relpath(base_path),
            }

    return {
        "methods": methods,
        "targets": targets,
        "method_groups": METHOD_GROUPS,
        "target_groups": TARGET_GROUPS,
        "defaults": {
            "method": "sivi",
            "target": "banana",
            "method_group": "all",
            "target_group": "all",
        },
        "availability": availability,
    }


def _compare(query: dict[str, list[str]]) -> dict[str, Any]:
    mode = _query_value(query, "mode", "methods_for_target")

    if mode == "methods_for_target":
        target = _validate_target(_query_value(query, "target", "banana") or "banana")
        methods = [_validate_method(method) for method in _query_list(query, "methods", METHODS)]
        configs = [_load_config(method, target) for method in methods]
        payload = _build_comparison(configs)
        payload["mode"] = mode
        payload["title"] = f"Methods on {target}"
        return payload

    if mode == "targets_for_method":
        method = _validate_method(_query_value(query, "method", "sivi") or "sivi")
        targets = [_validate_target(target) for target in _query_list(query, "targets", TARGETS)]
        configs = [_load_config(method, target) for target in targets]
        payload = _build_comparison(configs)
        payload["mode"] = mode
        payload["title"] = f"Targets for {METHOD_LABELS[method]}"
        return payload

    raise ConfigError(f"Unknown compare mode {mode!r}.")


def _raw(query: dict[str, list[str]]) -> dict[str, Any]:
    method = _validate_method(_query_value(query, "method", "sivi") or "sivi")
    target = _validate_target(_query_value(query, "target", "banana") or "banana")

    path = _base_path(method, target)
    data, error = _read_effective_config(path)
    text = _yaml_text(data)

    return {
        "method": method,
        "target": target,
        "path": _relpath(path),
        "display_path": f"resolved from {_relpath(path)}",
        "exists": data is not None,
        "error": error,
        "text": text,
    }


def _snapshot() -> dict[str, Any]:
    roots = [
        REPO_ROOT / "configs",
        STATIC_DIR,
        Path(__file__),
    ]
    files: dict[str, Any] = {}
    for root in roots:
        if root.is_file():
            candidates = [root]
        elif root.exists():
            candidates = sorted(path for path in root.rglob("*") if path.is_file())
        else:
            candidates = []
        for path in candidates:
            if path.suffix not in {".yaml", ".yml", ".py", ".html", ".css", ".js"}:
                continue
            stat = path.stat()
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
            files[_relpath(path)] = {
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
                "sha1": digest,
            }
    snapshot_hash = hashlib.sha1(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"hash": snapshot_hash, "files": files}


class ConfigReviewHandler(BaseHTTPRequestHandler):
    server_version = "ConfigReviewHTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._send_file(STATIC_DIR / "index.html")
            elif parsed.path.startswith("/static/"):
                self._send_file(STATIC_DIR / parsed.path.removeprefix("/static/"))
            elif parsed.path == "/api/metadata":
                self._send_json(_metadata())
            elif parsed.path == "/api/compare":
                self._send_json(_compare(query))
            elif parsed.path == "/api/raw":
                self._send_json(_raw(query))
            elif parsed.path == "/api/snapshot":
                self._send_json(_snapshot())
            else:
                self._send_error(HTTPStatus.NOT_FOUND, f"Unknown path {parsed.path}")
        except ConfigError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # noqa: BLE001
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        message = "%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args)
        sys.stderr.write(message)

    def _send_json(self, payload: Any) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            resolved.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_error(HTTPStatus.FORBIDDEN, "Static path is outside the reviewer directory.")
            return
        if not resolved.exists() or not resolved.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, f"Static file not found: {path.name}")
            return
        body = resolved.read_bytes()
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        if resolved.suffix in {".html", ".css", ".js"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the read-only config reviewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), ConfigReviewHandler)
    except OSError as exc:
        parser.error(f"Could not bind {args.host}:{args.port}: {exc}")
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving config reviewer at {url}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping config reviewer.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
