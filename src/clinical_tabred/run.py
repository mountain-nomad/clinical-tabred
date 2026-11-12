"""Command-line entry point.

Example:
    python -m clinical_tabred.run --config configs/experiment.yaml --stage all
"""

import argparse
import logging
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from clinical_tabred.benchmark import (
    package_versions,
    quick_config,
    run_all_protocols,
    run_cv_all,
    run_uncertainty,
    save_protocol_results,
    save_selected_models,
    select_best_model,
    summarize_random_seeds,
    write_run_metadata,
)
from clinical_tabred.build_notebook import PACKAGES, build_notebook
from clinical_tabred.diagrams import make_all_diagrams
from clinical_tabred.download import download_all
from clinical_tabred.drift import run_drift
from clinical_tabred.experiment import base_settings, run_ablation
from clinical_tabred.plotting import make_all_figures
from clinical_tabred.prepare import (
    build_dataset,
    load_processed,
    missingness_table,
    save_processed,
    subsample_per_cycle,
)
from clinical_tabred.report_tables import (
    write_captions,
    write_extra_tables,
    write_literature_tables,
    write_model_comparison_md,
    write_number_macros,
    write_report_tables,
)

logger = logging.getLogger("clinical_tabred")
STAGES = ["download", "prepare", "train", "drift", "figures", "report", "presentation",
          "notebook"]


def load_config(path: Path, quick: bool) -> dict:
    """Read the YAML config; in quick mode use cheap settings and separate output folders."""
    config = yaml.safe_load(Path(path).read_text())
    if quick:
        config = quick_config(config)
        config["paths"]["results_dir"] += "_quick"
        config["paths"]["figures_dir"] += "_quick"
    return config


def stage_download(config: dict) -> None:
    """Download and validate the 9 NHANES files (cached after the first run)."""
    manifest = download_all(config, Path(config["paths"]["raw_dir"]))
    logger.info("Manifest:\n%s", manifest[["file", "bytes", "validation"]].to_string())


def stage_prepare(config: dict) -> None:
    """Build the processed dataset and the data-flow / missingness tables."""
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset, flow, missing = build_dataset(config, Path(config["paths"]["raw_dir"]))
    save_processed(dataset, config, Path(config["paths"]["processed_dir"]))
    flow.to_csv(results_dir / "data_flow.csv", index=False)
    missing.to_csv(results_dir / "missing_components.csv", index=False)
    features = config["data"]["numeric_features"] + config["data"]["categorical_features"]
    missingness_table(dataset, features).to_csv(results_dir / "missingness.csv", index=False)
    logger.info("Processed dataset: %d rows, %d features", len(dataset), len(features))


def load_dataset(config: dict, quick: bool):
    """Load the processed dataset (a small per-cycle subsample in quick mode)."""
    dataset = load_processed(Path(config["paths"]["processed_dir"]), config)
    if quick:
        dataset = subsample_per_cycle(dataset, config["quick"]["rows_per_cycle"], config["seed"])
    return dataset


def stage_train(config: dict, quick: bool) -> None:
    """Train and evaluate all models under every protocol; CV, ablation and bootstrap."""
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(config, quick)
    runs = run_all_protocols(dataset, config)
    tables = save_protocol_results(runs, results_dir)
    summarize_random_seeds(tables["metrics"]).to_csv(results_dir / "random_seeds_summary.csv",
                                                     index=False)
    best = select_best_model(tables["metrics"])
    logger.info("Best protocol-A model on validation: %s", best)
    temporal = runs["outputs"][("temporal", config["seed"])]
    ablation = run_ablation(best, temporal["params"][best], temporal["splits"], config,
                            base_settings(config))
    ablation.to_csv(results_dir / "ablation.csv", index=False)
    run_cv_all(runs, config, results_dir)
    run_uncertainty(tables["predictions"], best, config, results_dir)
    save_selected_models(runs, best, config, results_dir)


def stage_drift(config: dict, quick: bool) -> None:
    """Compare feature distributions and prevalence between cycles."""
    run_drift(load_dataset(config, quick), config, config["paths"]["results_dir"])


def stage_figures(config: dict) -> None:
    """Export all figures (PDF + 300-DPI PNG) from the saved result CSVs."""
    paths = config["paths"]
    saved = make_all_figures(Path(paths["results_dir"]), Path(paths["figures_dir"]), "paper",
                             ["pdf", "png"])
    logger.info("Saved %d figure files to %s", len(saved), paths["figures_dir"])


def stage_report(config: dict) -> None:
    """Generate report tables, number macros, literature tables, diagrams; compile the PDF."""
    results, figures = Path(config["paths"]["results_dir"]), Path(config["paths"]["figures_dir"])
    tables = Path("report/tables")
    tables.mkdir(parents=True, exist_ok=True)
    write_literature_tables(results, Path("report/literature"))
    write_report_tables(results, tables)
    write_extra_tables(results, Path("report/timeline.csv"), tables)
    write_number_macros(results, tables / "numbers.tex")
    write_model_comparison_md(results)
    write_captions(figures)
    make_all_diagrams(figures, ["pdf", "png"])
    compile_report(Path("report"))


def compile_report(report_dir: Path) -> None:
    """Compile report/main.tex with latexmk or tectonic; log clearly if neither exists."""
    if shutil.which("latexmk"):
        command = ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"]
    elif shutil.which("tectonic"):
        command = ["tectonic", "--keep-logs", "main.tex"]
    else:
        logger.warning("No LaTeX engine (latexmk/tectonic) found: report/main.tex not compiled.")
        return
    result = subprocess.run(command, cwd=report_dir, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"LaTeX compilation failed:\n{result.stdout[-3000:]}\n"
                           f"{result.stderr[-3000:]}")
    logger.info("Compiled %s/main.pdf with %s", report_dir, command[0])


def stage_notebook(config: dict) -> None:
    """Rebuild the standalone notebook from the current source code and templates."""
    versions = package_versions(list(PACKAGES.values()))
    path = Path("notebooks") / config["project"]["notebook_name"]
    build_notebook(config, Path("notebooks/templates"), path, versions)
    logger.info("Notebook written to %s (execute it with jupyter nbconvert)", path)


def run_stage(stage: str, config: dict, quick: bool) -> None:
    """Dispatch one stage by name."""
    if stage == "download":
        stage_download(config)
    elif stage == "prepare":
        stage_prepare(config)
    elif stage == "train":
        stage_train(config, quick)
    elif stage == "drift":
        stage_drift(config, quick)
    elif stage == "figures":
        stage_figures(config)
    elif stage == "report":
        stage_report(config)
    elif stage == "notebook":
        stage_notebook(config)
    else:
        raise ValueError(f"Unknown stage '{stage}'. Choose from {STAGES + ['all']}.")


def main() -> None:
    """Parse arguments, run the requested stage(s) and record their runtimes."""
    parser = argparse.ArgumentParser(description="Clinical TabReD Lite pipeline")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--stage", default="all", choices=STAGES + ["all"])
    parser.add_argument("--quick", action="store_true", help="small subset smoke test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(Path(args.config), args.quick)
    stages = STAGES if args.stage == "all" else [args.stage]
    runtimes = {}
    for stage in stages:
        start = time.perf_counter()
        logger.info("=== Stage: %s ===", stage)
        run_stage(stage, config, args.quick)
        runtimes[stage] = round(time.perf_counter() - start, 1)
    write_run_metadata(config, Path(config["paths"]["results_dir"]),
                       {"quick": args.quick, "stage_runtimes_s": runtimes})


if __name__ == "__main__":
    main()
