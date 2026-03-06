"""
Electrofacies CLI — command-line interface for the DMG electrofacies pipeline.

Usage:
    electrofacies train    --config configs/default.yaml
    electrofacies predict  --well path/to/well.las --model artifacts/latest/
    electrofacies batch    --inbox data/wells/inbox/ --output outputs/
    electrofacies evaluate --model artifacts/latest/ --test-data data/training/test.csv
    electrofacies info     --model artifacts/latest/
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import click
import yaml

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _setup_logging(verbose: bool, log_dir: str | None = None):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_path / f"electrofacies_{ts}.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        handlers.append(fh)
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_project_root(config_path: str) -> Path:
    """Resolve the project root from the config file location."""
    return Path(config_path).resolve().parent.parent


@click.group()
@click.version_option(version="1.0.0", prog_name="electrofacies")
def cli():
    """DMG Electrofacies Prediction System.

    Production-quality electrofacies classification for the Delaware Mountain Group.
    Supports training, single-well prediction, and batch processing.
    """
    pass


@cli.command()
@click.option("--config", default="configs/default.yaml", help="Path to configuration YAML")
@click.option("--experiment-name", default=None, help="Name for this training run")
@click.option("--tiers", default="all", help="Comma-separated tier names to train (e.g. tier_1,tier_2) or 'all'")
@click.option("--algorithms", default="all", help="Comma-separated algorithms (random_forest,xgboost) or 'all'")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def train(config, experiment_name, tiers, algorithms, verbose):
    """Train electrofacies models from labeled training data."""
    _setup_logging(verbose)
    logger = logging.getLogger("electrofacies.train")

    cfg = _load_config(config)
    project_root = _resolve_project_root(config)
    _setup_logging(verbose, str(project_root / cfg["paths"]["logs"]))

    logger.info("=" * 60)
    logger.info("ELECTROFACIES TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info("Config: %s", config)
    logger.info("Experiment: %s", experiment_name or "auto")

    import pandas as pd

    from electrofacies.preprocessing.features import compute_training_stats, engineer_features
    from electrofacies.preprocessing.standardize import canonicalize_facies_labels, load_mnemonic_map
    from electrofacies.preprocessing.transform import FaciesTransformer
    from electrofacies.qc.ood import OODDetector
    from electrofacies.training.artifacts import save_model_bundle
    from electrofacies.training.evaluate import evaluate_model
    from electrofacies.training.split import create_depth_blocked_split, make_depth_groups
    from electrofacies.training.train import train_random_forest, train_xgboost

    # Load training data
    training_path = project_root / cfg["paths"]["training_data"]
    logger.info("Loading training data: %s", training_path)
    if training_path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(training_path)
    else:
        df = pd.read_csv(training_path)

    # Filter to target formation
    train_cfg = cfg["training"]
    if train_cfg.get("formation_filter") and train_cfg.get("formation_col"):
        df = df[df[train_cfg["formation_col"]] == train_cfg["formation_filter"]].copy()
        logger.info("Filtered to %s: %d rows", train_cfg["formation_filter"], len(df))

    # Canonicalize facies labels
    facies_config_path = project_root / cfg["configs"]["facies_schema"]
    with open(facies_config_path) as f:
        facies_config = yaml.safe_load(f)

    df[train_cfg["target_col"]] = canonicalize_facies_labels(
        df[train_cfg["target_col"]], facies_config
    )

    # Exclude missing_strata
    excluded = facies_config.get("excluded_labels", ["missing_strata"])
    df = df[~df[train_cfg["target_col"]].isin(excluded)].copy()
    logger.info("After excluding %s: %d rows", excluded, len(df))

    # Load tier config
    tiers_config_path = project_root / cfg["configs"]["model_tiers"]
    with open(tiers_config_path) as f:
        tiers_config = yaml.safe_load(f)

    # Determine which tiers and algorithms to train
    tier_list = list(tiers_config["tiers"].keys()) if tiers == "all" else [t.strip() for t in tiers.split(",")]
    algo_list = tiers_config.get("algorithms", ["random_forest", "xgboost"])
    if algorithms != "all":
        algo_list = [a.strip() for a in algorithms.split(",")]

    # Map raw feature columns to canonical
    raw_to_canon = train_cfg["raw_to_canonical"]
    canon_to_raw = {v: k for k, v in raw_to_canon.items()}

    # Encode labels
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    class_names = sorted(df[train_cfg["target_col"]].unique())
    le.fit(class_names)

    # Timestamp for this run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = experiment_name or f"run_{ts}"
    artifacts_dir = project_root / cfg["paths"]["artifacts"] / run_name

    logger.info("Training tiers: %s", tier_list)
    logger.info("Algorithms: %s", algo_list)
    logger.info("Classes: %s", class_names)
    logger.info("Artifacts dir: %s", artifacts_dir)

    fe_config = cfg.get("feature_engineering", tiers_config.get("feature_engineering", {}))

    for tier_name in tier_list:
        tier_def = tiers_config["tiers"].get(tier_name)
        if not tier_def:
            logger.warning("Tier %s not found in config, skipping", tier_name)
            continue

        required_logs = tier_def["required_logs"]
        raw_cols = [canon_to_raw.get(log) for log in required_logs]
        missing_cols = [c for c in raw_cols if c is None or c not in df.columns]

        if missing_cols:
            logger.warning("Tier %s: missing columns %s, skipping", tier_name, missing_cols)
            continue

        # Filter to rows with all required logs
        tier_df = df.dropna(subset=raw_cols + [train_cfg["target_col"]]).copy()
        logger.info("Tier %s: %d rows with complete data for %s", tier_name, len(tier_df), required_logs)

        if len(tier_df) < 50:
            logger.warning("Tier %s: too few rows (%d), skipping", tier_name, len(tier_df))
            continue

        # Compute training stats and engineer features
        training_stats = compute_training_stats(tier_df, raw_cols)
        tier_df, all_feature_cols = engineer_features(
            tier_df, raw_cols, config=fe_config, training_stats=training_stats,
        )

        # Build transformer
        transformer = FaciesTransformer()
        transformer.fit(tier_df, raw_cols)

        # Encode target
        y = le.transform(tier_df[train_cfg["target_col"]].values)
        X = tier_df[all_feature_cols].fillna(0).values

        # Depth groups for CV
        depth_col = train_cfg.get("depth_col")
        groups = None
        if depth_col and depth_col in tier_df.columns:
            groups = make_depth_groups(
                tier_df[depth_col].values,
                group_size=train_cfg.get("depth_group_size", 50),
            )

        # Train/test split
        if depth_col and depth_col in tier_df.columns:
            train_idx, test_idx = create_depth_blocked_split(
                X, y, tier_df[depth_col].values,
                test_fraction=train_cfg.get("test_size", 0.25),
            )
        else:
            from electrofacies.training.split import create_stratified_split
            train_idx, test_idx = create_stratified_split(
                X, y, test_size=train_cfg.get("test_size", 0.25),
            )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit OOD detector on training data
        ood = OODDetector(method="mahalanobis", percentile_threshold=cfg["qc"]["ood_percentile"])
        ood.fit(X_train)

        # Groups for training CV (only on train split)
        if groups is not None:
            train_groups = groups[train_idx]
        else:
            train_groups = None

        for algo in algo_list:
            logger.info("Training %s / %s ...", tier_name, algo)
            try:
                if algo == "random_forest":
                    model, best_params = train_random_forest(
                        X_train, y_train, train_groups, cfg["models"]["random_forest"],
                        use_smote=train_cfg.get("use_smote", True),
                    )
                elif algo == "xgboost":
                    model, best_params = train_xgboost(
                        X_train, y_train, train_groups, cfg["models"]["xgboost"],
                        use_smote=train_cfg.get("use_smote", True),
                    )
                else:
                    logger.warning("Unknown algorithm: %s", algo)
                    continue

                # Evaluate
                metrics = evaluate_model(model, X_test, y_test, list(le.classes_))
                logger.info(
                    "%s/%s — balanced_acc: %.3f, kappa: %.3f",
                    tier_name, algo,
                    metrics["balanced_accuracy"],
                    metrics["cohen_kappa"],
                )

                # Save bundle
                bundle_path = save_model_bundle(
                    model=model,
                    transformer=transformer,
                    ood_detector=ood,
                    config=cfg,
                    metrics=metrics,
                    output_dir=str(artifacts_dir),
                    tier_name=tier_name,
                    algorithm=algo,
                )
                logger.info("Saved: %s", bundle_path)

            except Exception:
                logger.exception("Failed to train %s/%s", tier_name, algo)

    # Create a 'latest' symlink
    latest_link = project_root / cfg["paths"]["artifacts"] / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    try:
        latest_link.symlink_to(artifacts_dir.resolve())
        logger.info("Symlinked latest -> %s", artifacts_dir.name)
    except OSError:
        logger.warning("Could not create 'latest' symlink")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


@cli.command()
@click.option("--well", required=True, type=click.Path(exists=True), help="Path to LAS or CSV well file")
@click.option("--model", required=True, type=click.Path(exists=True), help="Path to model artifacts directory")
@click.option("--output", default="outputs/", help="Output directory")
@click.option("--config", default="configs/default.yaml", help="Path to configuration YAML")
@click.option("--smooth/--no-smooth", default=True, help="Apply modal smoothing filter")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def predict(well, model, output, config, smooth, verbose):
    """Predict electrofacies for a single well."""
    _setup_logging(verbose)
    logger = logging.getLogger("electrofacies.predict")

    cfg = _load_config(config)
    project_root = _resolve_project_root(config)

    logger.info("Predicting: %s", well)
    logger.info("Model: %s", model)

    from electrofacies.inference.batch import BatchRunner

    runner = BatchRunner(config_path=config, artifacts_dir=model)
    result = runner.process_well(well)

    if result["status"] == "success":
        # Write outputs
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)

        from electrofacies.io.writers import write_predictions_csv, write_well_report

        well_name = result.get("well_name", Path(well).stem)
        csv_path = out_dir / f"{well_name}_predictions.csv"
        write_predictions_csv(result["predictions"], str(csv_path))
        write_well_report(result, str(out_dir))

        if cfg["output"].get("save_plots", True):
            try:
                from electrofacies.visualization.log_display import plot_well_predictions

                plot_path = out_dir / f"{well_name}_log_display.png"
                plot_well_predictions(
                    result.get("well_data"), result["predictions"],
                    str(plot_path), title=well_name, dpi=cfg["output"].get("plot_dpi", 300),
                )
            except Exception:
                logger.exception("Failed to generate plot")

        logger.info("Prediction complete: %s", csv_path)
        click.echo(f"Predictions saved: {csv_path}")
        click.echo(f"QC Grade: {result.get('qc_summary', {}).get('qc_grade', 'N/A')}")
        click.echo(f"Tier: {result.get('tier_used', 'N/A')}")
        click.echo(f"Algorithm: {result.get('algorithm', 'N/A')}")
    else:
        logger.error("Prediction failed: %s", result.get("error", "unknown"))
        click.echo(f"FAILED: {result.get('error', 'unknown')}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--inbox", required=True, type=click.Path(exists=True), help="Folder with LAS/CSV well files")
@click.option("--model", required=True, type=click.Path(exists=True), help="Path to model artifacts directory")
@click.option("--output", default="outputs/", help="Output directory for results")
@click.option("--config", default="configs/default.yaml", help="Path to configuration YAML")
@click.option("--move/--no-move", default=False, help="Move processed files to processed/failed folders")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def batch(inbox, model, output, config, move, verbose):
    """Batch-predict electrofacies for all wells in a folder."""
    _setup_logging(verbose)
    logger = logging.getLogger("electrofacies.batch")

    cfg = _load_config(config)
    project_root = _resolve_project_root(config)
    _setup_logging(verbose, str(project_root / cfg["paths"]["logs"]))

    logger.info("=" * 60)
    logger.info("ELECTROFACIES BATCH PREDICTION")
    logger.info("=" * 60)
    logger.info("Inbox: %s", inbox)
    logger.info("Model: %s", model)

    from electrofacies.inference.batch import BatchRunner

    runner = BatchRunner(config_path=config, artifacts_dir=model)
    results = runner.run_batch(inbox_dir=inbox, output_dir=output)

    # Summary
    n_total = len(results)
    n_success = sum(1 for r in results if r["status"] == "success")
    n_failed = n_total - n_success

    click.echo("")
    click.echo("=" * 50)
    click.echo("BATCH COMPLETE")
    click.echo(f"  Total wells:  {n_total}")
    click.echo(f"  Successful:   {n_success}")
    click.echo(f"  Failed:       {n_failed}")

    if n_success > 0:
        tier_counts = {}
        qc_counts = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
        for r in results:
            if r["status"] == "success":
                t = r.get("tier_used", "unknown")
                tier_counts[t] = tier_counts.get(t, 0) + 1
                g = r.get("qc_summary", {}).get("qc_grade", "N/A")
                if g in qc_counts:
                    qc_counts[g] += 1

        click.echo(f"  Tiers used:   {tier_counts}")
        click.echo(f"  QC grades:    {qc_counts}")

    click.echo(f"  Output:       {output}")
    click.echo("=" * 50)


@cli.command()
@click.option("--model", required=True, type=click.Path(exists=True), help="Path to model artifacts directory")
def info(model):
    """Display information about trained model artifacts."""
    import json

    model_dir = Path(model)

    click.echo(f"Model directory: {model_dir.resolve()}")
    click.echo("")

    # Find all tier/algorithm bundles
    for bundle_dir in sorted(model_dir.iterdir()):
        if not bundle_dir.is_dir():
            continue

        meta_path = bundle_dir / "metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        metrics_path = bundle_dir / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        click.echo(f"  {meta.get('tier', '?')} / {meta.get('algorithm', '?')}")
        click.echo(f"    Required logs: {meta.get('required_logs', [])}")
        click.echo(f"    Features:      {len(meta.get('feature_columns', []))} columns")
        click.echo(f"    Classes:       {meta.get('class_names', [])}")
        click.echo(f"    Balanced Acc:  {metrics.get('balanced_accuracy', 'N/A'):.3f}" if isinstance(metrics.get('balanced_accuracy'), (int, float)) else f"    Balanced Acc:  N/A")
        click.echo(f"    Cohen Kappa:   {metrics.get('cohen_kappa', 'N/A'):.3f}" if isinstance(metrics.get('cohen_kappa'), (int, float)) else f"    Cohen Kappa:   N/A")
        click.echo(f"    Trained:       {meta.get('timestamp', 'N/A')}")
        click.echo("")


if __name__ == "__main__":
    cli()
