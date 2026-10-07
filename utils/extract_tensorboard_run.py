import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util


CONFIG_TAG = "config/full_config/text_summary"


def _decode_text_tensor(tensor_event) -> str:
    arr = tensor_util.make_ndarray(tensor_event.tensor_proto)
    value = arr.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _strip_markdown_code_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).rstrip() + "\n"
    return text


def extract_run(run_dir: Path) -> tuple[str | None, list[dict], dict]:
    accumulator = event_accumulator.EventAccumulator(str(run_dir))
    accumulator.Reload()
    tags = accumulator.Tags()

    config_text = None
    if CONFIG_TAG in tags.get("tensors", []):
        tensor_events = accumulator.Tensors(CONFIG_TAG)
        if tensor_events:
            config_text = _strip_markdown_code_fence(
                _decode_text_tensor(tensor_events[-1])
            )

    scalar_rows: list[dict] = []
    scalar_summary: dict[str, dict] = {}
    for tag in sorted(tags.get("scalars", [])):
        events = accumulator.Scalars(tag)
        if not events:
            continue
        scalar_summary[tag] = {
            "count": len(events),
            "first_step": int(events[0].step),
            "last_step": int(events[-1].step),
            "first_value": float(events[0].value),
            "last_value": float(events[-1].value),
        }
        for event in events:
            scalar_rows.append(
                {
                    "tag": tag,
                    "step": int(event.step),
                    "wall_time": float(event.wall_time),
                    "value": float(event.value),
                }
            )

    metadata = {
        "run_dir": str(run_dir.resolve()),
        "num_scalar_tags": len(scalar_summary),
        "scalar_tags": sorted(scalar_summary),
        "has_config_text": config_text is not None,
        "config_tag": CONFIG_TAG if config_text is not None else None,
        "scalar_summary": scalar_summary,
    }
    return config_text, scalar_rows, metadata


def write_outputs(
    out_dir: Path, config_text: str | None, scalar_rows: list[dict], metadata: dict
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if config_text is not None:
        (out_dir / "full_config.yaml").write_text(config_text, encoding="utf-8")

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["tag", "step", "wall_time", "value"])
        writer.writeheader()
        writer.writerows(scalar_rows)

    (out_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract scalar metrics and the logged full config from a TensorBoard run."
        )
    )
    parser.add_argument("run_dir", type=Path, help="TensorBoard run directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_dir>/extracted",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    out_dir = args.out_dir.resolve() if args.out_dir else run_dir / "extracted"
    config_text, scalar_rows, metadata = extract_run(run_dir)
    write_outputs(out_dir, config_text, scalar_rows, metadata)

    print(f"Run: {run_dir}")
    print(f"Output: {out_dir}")
    print(f"Scalar tags: {metadata['num_scalar_tags']}")
    print(f"Scalar points: {len(scalar_rows)}")
    if config_text is None:
        print("Config: not found in TensorBoard event file")
    else:
        print("Config: extracted to full_config.yaml")


if __name__ == "__main__":
    main()
