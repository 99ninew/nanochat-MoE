"""Hyperparameter sweep runner for Chat SFT.

This script launches multiple `torchrun -m scripts.chat_sft` runs with different
hyperparameters (passed via nanochat/configurator.py-style `--key=value` args),
captures logs, and writes a JSONL summary of results.

Example:
  python -m scripts.sweep_chat_sft --nproc-per-node=2 --model-tag=d6 --step=830 --dry-run

  python -m scripts.sweep_chat_sft --nproc-per-node=2 --model-tag=d6 --step=830 \
    --grid "learning_rate=2e-4,3e-4,5e-4" \
    --grid "target_examples_per_step=64,128" \
    --grid "device_batch_size=8,16" \
    --grid "num_epochs=2"

Notes:
- By default we keep `run=dummy` (no wandb). Set `--use-wandb` to enable.
- Logs + results are written under `$NANOCHAT_BASE_DIR/sweeps/chat_sft/<timestamp>/`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NANOCHAT_BASE_DIR = "/thullms/jnminniewang/nanochat-MoE/.cache/nanochat-moe/"

def _default_base_dir() -> Path:
    base = NANOCHAT_BASE_DIR
    return Path(base)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_scalar(value: str) -> Any:
    """Parse scalar values compatible with configurator.py.

    We keep this intentionally conservative:
    - ints
    - floats (incl scientific)
    - booleans
    - otherwise string
    """
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    # int
    if re.fullmatch(r"[+-]?\d+", v):
        try:
            return int(v)
        except Exception:
            pass
    # float (simple / scientific)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?", v):
        try:
            return float(v)
        except Exception:
            pass
    return v


def _parse_grid_kv(expr: str) -> tuple[str, list[Any]]:
    """Parse `key=v1,v2,v3` into (key, [values])."""
    if "=" not in expr:
        raise ValueError(f"Bad --grid entry (expected key=v1,v2,...): {expr}")
    key, rhs = expr.split("=", 1)
    key = key.strip()
    values = [
        _parse_scalar(x)
        for x in rhs.split(",")
        if x.strip() != ""
    ]
    if not key or not values:
        raise ValueError(f"Bad --grid entry: {expr}")
    return key, values


@dataclass(frozen=True)
class RunConfig:
    params: dict[str, Any]

    def short_name(self) -> str:
        parts = []
        for k in sorted(self.params.keys()):
            v = self.params[k]
            if isinstance(v, float):
                sv = f"{v:g}"
            else:
                sv = str(v)
            sv = sv.replace("/", "-")
            parts.append(f"{k}={sv}")
        # keep filesystem-friendly and not too long
        name = "__".join(parts)
        return name[:220]


_METRIC_PATTERNS = {
    "val_loss": re.compile(r"Validation loss:\s*([0-9]+\.[0-9]+)", re.IGNORECASE),
    "train_loss": re.compile(r"Training loss:\s*([0-9]+\.[0-9]+)", re.IGNORECASE),
    "mmlu_acc": re.compile(r"mmlu_acc:\s*([0-9]+\.[0-9]+)", re.IGNORECASE),
    "arc_easy_acc": re.compile(r"arc_easy_acc:\s*([0-9]+\.[0-9]+)", re.IGNORECASE),
}


def _extract_last_metrics(log_text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, pat in _METRIC_PATTERNS.items():
        matches = list(pat.finditer(log_text))
        if matches:
            out[key] = float(matches[-1].group(1))
    return out


def _build_torchrun_cmd(
    *,
    nproc_per_node: int,
    master_port: int | None,
    chat_sft_args: dict[str, Any],
) -> list[str]:
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}"]
    if master_port is not None:
        cmd.append(f"--master_port={master_port}")
    cmd += ["-m", "scripts.chat_sft", "--"]

    # configurator expects --key=value
    for k, v in chat_sft_args.items():
        if isinstance(v, bool):
            vv = "True" if v else "False"
        else:
            vv = str(v)
        cmd.append(f"--{k}={vv}")

    return cmd


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Sweep hyperparameters for scripts.chat_sft")
    p.add_argument("--nproc-per-node", type=int, default=2)
    p.add_argument("--master-port", type=int, default=None)
    p.add_argument("--source", type=str, default="mid")
    p.add_argument("--model-tag", type=str, default=None)
    p.add_argument("--step", type=int, default=None)

    p.add_argument(
        "--grid",
        action="append",
        default=[],
        help='Grid dimension, e.g. "learning_rate=2e-4,3e-4" (repeatable).',
    )

    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip runs with existing log file")

    p.add_argument("--use-wandb", action="store_true", help="Set run to unique names (not dummy)")
    p.add_argument("--run-prefix", type=str, default=f"sweep_chat_sft_{_timestamp()}")

    p.add_argument(
        "--log-metrics-max-problems",
        type=int,
        default=256,
        help="Override eval_metrics_max_problems to speed sweeps.",
    )

    args = p.parse_args(argv)

    base_dir = _default_base_dir()
    sweep_dir = base_dir / "sweeps" / "chat_sft" / _timestamp()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_path = sweep_dir / "results.jsonl"

    # Default grid: tuned for 2x80GB class GPUs.
    grid_entries = [
        ("learning_rate", [2e-4, 3e-4, 5e-4]),
        ("target_examples_per_step", [64, 128]),
        ("device_batch_size", [8, 16]),
        ("num_epochs", [2]),
        ("init_lr_frac", [0.02, 1.0]),
    ]
    for g in args.grid:
        k, vs = _parse_grid_kv(g)
        # replace any default with user-provided
        grid_entries = [e for e in grid_entries if e[0] != k]
        grid_entries.append((k, vs))

    grid_entries.sort(key=lambda x: x[0])

    keys = [k for k, _ in grid_entries]
    values = [v for _, v in grid_entries]
    combos = list(itertools.product(*values))

    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    print(f"Sweep dir: {sweep_dir}")
    print(f"Total runs: {len(combos)}")

    for idx, combo in enumerate(combos):
        config = {k: v for k, v in zip(keys, combo)}

        # Always ensure divisibility (chat_sft.py asserts this).
        # examples_per_step = device_batch_size * ddp_world_size
        examples_per_step = int(config["device_batch_size"]) * int(args.nproc_per_node)
        target = int(config["target_examples_per_step"])
        if target % examples_per_step != 0:
            print(f"[skip] incompatible batch: target={target} not divisible by examples_per_step={examples_per_step}")
            continue

        rc = RunConfig(params=config)
        run_name = f"{args.run_prefix}__{idx:03d}__{rc.short_name()}"
        log_path = sweep_dir / f"{run_name}.log"

        if args.resume and log_path.exists():
            print(f"[resume] skip existing log: {log_path.name}")
            continue

        chat_sft_args: dict[str, Any] = {
            "source": args.source,
            "model_tag": args.model_tag,
            "step": args.step,
            "run": run_name if args.use_wandb else "dummy",
            "eval_metrics_max_problems": int(args.log_metrics_max_problems),
            **config,
        }
        # Remove None values to avoid configurator unknown / type issues
        chat_sft_args = {k: v for k, v in chat_sft_args.items() if v is not None}

        cmd = _build_torchrun_cmd(
            nproc_per_node=args.nproc_per_node,
            master_port=args.master_port,
            chat_sft_args=chat_sft_args,
        )

        print("=" * 80)
        print(f"Run {idx+1}/{len(combos)}: {run_name}")
        print("Command:")
        print("  " + " ".join(shlex.quote(c) for c in cmd))

        if args.dry_run:
            continue

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("CMD: " + " ".join(shlex.quote(c) for c in cmd) + "\n")
            f.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=os.environ.copy(),
            )
            rc_code = proc.wait()

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        metrics = _extract_last_metrics(log_text)

        record = {
            "run_name": run_name,
            "returncode": rc_code,
            "log_path": str(log_path),
            "params": config,
            "metrics": metrics,
        }
        with open(results_path, "a", encoding="utf-8") as rf:
            rf.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Return code: {rc_code}")
        print(f"Metrics: {metrics}")

    print("Done.")
    print(f"Results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
