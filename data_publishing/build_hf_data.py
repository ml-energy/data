"""Build a self-contained HF-ready data package from benchmark results.

This CLI produces a directory structure that can be uploaded to HF Hub
via ``hf upload-large-folder``. The output includes:

- ``runs/llm.parquet`` and ``runs/diffusion.parquet``: pre-computed run
  summary tables browseable via HF Data Studio.
- ``configs/``: embedded model_info.json and monolithic.config.yaml files.
- ``llm/`` and ``diffusion/``: raw results.json and prometheus.json files
  (skipped with ``--skip-raw``).
- ``README.md``: YAML config for HF Data Studio + schema documentation.

Usage::

    python data_publishing/build_hf_data.py \\
      --results-dir /path/to/llm/h100/current/run \\
      --results-dir /path/to/llm/b200/current/run \\
      --results-dir /path/to/diffusion/h100/current/run \\
      --results-dir /path/to/diffusion/b200/current/run \\
      --llm-config-dir /path/to/configs/vllm \\
      --diffusion-config-dir /path/to/configs/xdit \\
      --out-dir /path/to/output
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
import pandas as pd

from mlenergy_data.records.runs import DiffusionRuns, LLMRuns

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent


def _build_readme() -> str:
    """Assemble the HF repo README from the YAML config and dataset card."""
    yaml_config = (_SCRIPT_DIR / "hf_dataset_config.yaml").read_text().strip()
    dataset_card = (_SCRIPT_DIR / "DATASET_CARD.md").read_text().strip()
    return f"---\n{yaml_config}\n---\n\n{dataset_card}\n"


def _rewrite_paths_relative(df: pd.DataFrame, roots: list[Path]) -> pd.DataFrame:
    """Rewrite absolute results_path / prometheus_path to relative paths.

    For each path, find which root it falls under and make it relative
    to that root's grandparent (the benchmark root, e.g. ``3.0/``).
    """
    df = df.copy()
    path_cols = [c for c in ("results_path", "prometheus_path") if c in df.columns]
    for col in path_cols:
        new_vals: list[str] = []
        for val in df[col]:
            p = Path(str(val))
            made_relative = False
            for root in roots:
                try:
                    bench_root = _infer_benchmark_root_from_results_dir(root)
                    rel = p.relative_to(bench_root)
                    new_vals.append(rel.as_posix())
                    made_relative = True
                    break
                except ValueError:
                    continue
            if not made_relative:
                new_vals.append(str(val))
        df[col] = new_vals
    return df


def _infer_benchmark_root_from_results_dir(results_dir: Path) -> Path:
    """Infer the benchmark root from a results directory path.

    The expected structure is: ``<root>/<domain>/<gpu>/<snapshot>/run``
    """
    if results_dir.name == "run" and len(results_dir.parents) >= 4:
        return results_dir.parents[3]
    for candidate in [results_dir, *results_dir.parents]:
        if (candidate / "llm").exists() or (candidate / "diffusion").exists():
            return candidate
    raise ValueError(f"Could not infer benchmark root from: {results_dir}")



def _copy_configs(
    *,
    llm_config_dir: Path | None,
    diffusion_config_dir: Path | None,
    out_dir: Path,
) -> None:
    """Copy model_info.json and monolithic.config.yaml into output configs/."""
    if llm_config_dir is not None and llm_config_dir.is_dir():
        dst_base = out_dir / "configs" / "vllm"
        for src in llm_config_dir.rglob("model_info.json"):
            rel = src.relative_to(llm_config_dir)
            dst = dst_base / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        for src in llm_config_dir.rglob("monolithic.config.yaml"):
            rel = src.relative_to(llm_config_dir)
            dst = dst_base / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        logger.info("Copied LLM configs to %s", dst_base)

    if diffusion_config_dir is not None and diffusion_config_dir.is_dir():
        dst_base = out_dir / "configs" / "xdit"
        for src in diffusion_config_dir.rglob("model_info.json"):
            rel = src.relative_to(diffusion_config_dir)
            dst = dst_base / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        logger.info("Copied diffusion configs to %s", dst_base)


def _copy_raw_results(
    results_dirs: list[Path],
    out_dir: Path,
) -> None:
    """Copy raw results.json and prometheus.json into the output directory.

    Excludes diffusion media files (images, videos).
    """
    media_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".webm", ".avi"}
    copied = 0
    skipped = 0

    for results_dir in results_dirs:
        try:
            bench_root = _infer_benchmark_root_from_results_dir(results_dir)
        except ValueError:
            logger.warning("Could not infer benchmark root from %s, skipping", results_dir)
            continue

        for src in results_dir.rglob("*"):
            if not src.is_file():
                continue
            if src.suffix.lower() in media_suffixes:
                skipped += 1
                continue
            if src.name not in ("results.json", "prometheus.json"):
                skipped += 1
                continue
            try:
                rel = src.relative_to(bench_root)
            except ValueError:
                continue
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    logger.info("Copied %d raw files (%d skipped)", copied, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build self-contained HF-ready data package from benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        action="append",
        dest="results_dirs",
        required=True,
        help="Results directory (can be specified multiple times)",
    )
    parser.add_argument(
        "--llm-config-dir",
        type=str,
        default=None,
        help="LLM config directory (model_info.json, monolithic.config.yaml)",
    )
    parser.add_argument(
        "--diffusion-config-dir",
        type=str,
        default=None,
        help="Diffusion config directory (model_info.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/turbo/jwnchung/benchmark_hf/3.0",
        help="Output directory for the HF-ready package",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip copying raw results files (parquet + configs only)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    results_dirs = [Path(d) for d in args.results_dirs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_config_dir = Path(args.llm_config_dir) if args.llm_config_dir else None
    diffusion_config_dir = Path(args.diffusion_config_dir) if args.diffusion_config_dir else None

    logger.info("Loading LLM runs from %d results directories", len(results_dirs))
    llm = LLMRuns.from_raw_results(
        *sorted(results_dirs),
        config_dir=llm_config_dir,
        stable_only=False,
    )
    logger.info("Loaded %d LLM runs", len(llm))

    logger.info("Loading diffusion runs from %d results directories", len(results_dirs))
    diff = DiffusionRuns.from_raw_results(
        *sorted(results_dirs),
        config_dir=diffusion_config_dir,
    )
    logger.info("Loaded %d diffusion runs", len(diff))

    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    llm_df = llm.to_dataframe()
    diff_df = diff.to_dataframe()

    if not llm_df.empty:
        llm_df = _rewrite_paths_relative(llm_df, results_dirs)
    if not diff_df.empty:
        diff_df = _rewrite_paths_relative(diff_df, results_dirs)

    llm_df.to_parquet(runs_dir / "llm.parquet", index=False)
    logger.info("Wrote runs/llm.parquet (%d rows)", len(llm_df))

    diff_df.to_parquet(runs_dir / "diffusion.parquet", index=False)
    logger.info("Wrote runs/diffusion.parquet (%d rows)", len(diff_df))

    _copy_configs(
        llm_config_dir=llm_config_dir,
        diffusion_config_dir=diffusion_config_dir,
        out_dir=out_dir,
    )

    if not args.skip_raw:
        _copy_raw_results(results_dirs, out_dir)
    else:
        logger.info("Skipping raw results copy (--skip-raw)")

    readme_path = out_dir / "README.md"
    readme_path.write_text(_build_readme())
    logger.info("Wrote %s", readme_path)

    license_path = out_dir / "LICENSE"
    license_path.write_text((_SCRIPT_DIR / "LICENSE").read_text())
    logger.info("Wrote %s", license_path)

    logger.info("Done. Output directory: %s", out_dir)
    logger.info("  runs/llm.parquet: %d rows", len(llm_df))
    logger.info("  runs/diffusion.parquet: %d rows", len(diff_df))


if __name__ == "__main__":
    main()
