"""Batch orchestrator for the electrofacies inference pipeline.

Provides the :class:`BatchRunner`, which processes an entire inbox of LAS/CSV
well files end-to-end: read, standardise, validate, route to the best model
tier, predict with all available algorithms, post-process, compute QC
summaries, and write outputs.

Designed for production use: individual well failures are caught and logged
so they never crash the entire batch.
"""

from __future__ import annotations

import json
import logging
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

from electrofacies.io.readers import read_las, read_csv_well, scan_wells_folder
from electrofacies.io.schemas import (
    CANONICAL_LOGS,
    DEPTH,
    PredictionResult,
    WellData,
)
from electrofacies.io.writers import (
    write_batch_summary,
    write_predictions_csv,
    write_predictions_las,
    write_well_report,
)
from electrofacies.inference.tier_router import (
    determine_available_logs,
    load_tier_config,
    load_tier_models,
    select_best_tier,
)
from electrofacies.inference.predict import (
    predict_all_algorithms,
    select_best_prediction,
)
from electrofacies.inference.postprocess import (
    assign_confidence_flags,
    assign_ood_flags,
    assign_qc_status,
    compute_well_summary,
    modal_filter,
)
from electrofacies.preprocessing.standardize import (
    load_mnemonic_map,
    standardize_columns,
)

logger = logging.getLogger(__name__)


class BatchRunner:
    """Orchestrates batch inference over a folder of well files.

    Parameters
    ----------
    config_path : str or Path
        Path to the master ``default.yaml`` configuration file.
    artifacts_dir : str or Path
        Root directory containing trained model bundles organised by tier
        and algorithm.
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        artifacts_dir: Union[str, Path],
    ) -> None:
        self._config_path = Path(config_path).resolve()
        self._artifacts_dir = Path(artifacts_dir).resolve()

        # ----- Load master config -------------------------------------------
        if not self._config_path.is_file():
            raise FileNotFoundError(
                f"Master config not found: {self._config_path}"
            )
        with open(self._config_path, "r") as fh:
            self.config: Dict[str, Any] = yaml.safe_load(fh)
        logger.info("Loaded master config from %s.", self._config_path)

        # ----- Resolve config directory -------------------------------------
        self._config_dir = self._config_path.parent

        # ----- Load mnemonic map --------------------------------------------
        mnemonic_cfg_rel = self.config.get("configs", {}).get(
            "mnemonic_aliases", "configs/mnemonic_aliases.yaml"
        )
        mnemonic_cfg_path = self._resolve_path(mnemonic_cfg_rel)
        self.mnemonic_map = load_mnemonic_map(mnemonic_cfg_path)

        # ----- Load tier config ---------------------------------------------
        tiers_cfg_rel = self.config.get("configs", {}).get(
            "model_tiers", "configs/model_tiers.yaml"
        )
        tiers_cfg_path = self._resolve_path(tiers_cfg_rel)
        self.tiers_config = load_tier_config(tiers_cfg_path)

        # ----- Load physical ranges config for validation --------------------
        ranges_cfg_rel = self.config.get("configs", {}).get(
            "physical_ranges", "configs/physical_ranges.yaml"
        )
        ranges_cfg_path = self._resolve_path(ranges_cfg_rel)
        try:
            with open(ranges_cfg_path, "r") as fh:
                self._ranges_config: Dict[str, Any] = yaml.safe_load(fh)
        except FileNotFoundError:
            logger.warning(
                "Physical ranges config not found at %s; validation will "
                "use empty ranges.",
                ranges_cfg_path,
            )
            self._ranges_config = {"ranges": {}, "quality": {}, "washout": {}}

        # ----- QC thresholds ------------------------------------------------
        qc_cfg = self.config.get("qc", {})
        self._confidence_threshold: float = qc_cfg.get(
            "confidence_threshold", 0.50
        )
        self._min_log_coverage: float = qc_cfg.get("min_log_coverage", 0.70)
        self._smoothing_window: int = 3

        # ----- Optional components (loaded lazily) --------------------------
        self._validator = None
        self._transformer_cache: Dict[str, Any] = {}
        self._ood_detector: Optional[Any] = None

        # ----- Discover available model bundles -----------------------------
        if not self._artifacts_dir.is_dir():
            logger.warning(
                "Artifacts directory not found: %s. "
                "Model loading will fail at prediction time.",
                self._artifacts_dir,
            )

        logger.info(
            "BatchRunner initialised: config=%s, artifacts=%s.",
            self._config_path,
            self._artifacts_dir,
        )

    # ------------------------------------------------------------------
    # Path resolution helper
    # ------------------------------------------------------------------

    def _resolve_path(self, relative_or_absolute: str) -> Path:
        """Resolve a path that may be relative to the project root.

        Project root is assumed to be the parent of the ``configs/``
        directory (i.e. two levels up from a config file inside
        ``configs/``).
        """
        p = Path(relative_or_absolute)
        if p.is_absolute():
            return p

        # Try relative to config directory first, then its parent (project root).
        candidate = self._config_dir / p
        if candidate.exists():
            return candidate.resolve()

        candidate = self._config_dir.parent / p
        if candidate.exists():
            return candidate.resolve()

        # Fall back to returning relative to config dir even if not found yet.
        return (self._config_dir / p).resolve()

    # ------------------------------------------------------------------
    # Lazy loaders for optional pipeline components
    # ------------------------------------------------------------------

    def _get_validator(self):
        """Lazily import and return the validate_well function."""
        if self._validator is None:
            try:
                from electrofacies.preprocessing.validate import validate_well
                self._validator = validate_well
            except ImportError:
                logger.warning(
                    "Could not import validate_well; validation will be "
                    "skipped."
                )
                self._validator = lambda df, **kw: {"valid": True, "issues": []}
        return self._validator

    def _get_transformer(self, tier_name: str, model_bundle: Dict[str, Any]):
        """Lazily import and return a FaciesTransformer for the given tier.

        If the model bundle includes a serialised transformer, that is used.
        Otherwise a fresh ``FaciesTransformer`` is created from config.
        """
        if tier_name in self._transformer_cache:
            return self._transformer_cache[tier_name]

        # Prefer the transformer stored inside the model bundle.
        if "transformer" in model_bundle:
            transformer = model_bundle["transformer"]
            self._transformer_cache[tier_name] = transformer
            logger.info(
                "Using transformer from model bundle for tier '%s'.",
                tier_name,
            )
            return transformer

        # Fallback: build from the transform module.
        try:
            from electrofacies.preprocessing.transform import FaciesTransformer
            transformer = FaciesTransformer(config=self.config)
            self._transformer_cache[tier_name] = transformer
            logger.info(
                "Created FaciesTransformer from config for tier '%s'.",
                tier_name,
            )
            return transformer
        except ImportError:
            logger.warning(
                "Could not import FaciesTransformer; using identity "
                "transform."
            )

            class _IdentityTransformer:
                def transform(self, df: pd.DataFrame) -> pd.DataFrame:
                    return df.copy()

            transformer = _IdentityTransformer()
            self._transformer_cache[tier_name] = transformer
            return transformer

    def _get_ood_detector(self, model_bundle: Dict[str, Any]):
        """Return an OOD detector if available in the model bundle."""
        if self._ood_detector is not None:
            return self._ood_detector

        if "ood_detector" in model_bundle:
            self._ood_detector = model_bundle["ood_detector"]
            logger.info("Using OOD detector from model bundle.")
            return self._ood_detector

        # Try loading from the qc module.
        try:
            from electrofacies.qc.ood import OODDetector
            ood_path = self._artifacts_dir / "ood_detector"
            if ood_path.is_dir() or ood_path.with_suffix(".pkl").is_file():
                # Attempt to load a pre-fitted detector.
                self._ood_detector = OODDetector.load(ood_path)
                logger.info("Loaded OOD detector from %s.", ood_path)
                return self._ood_detector
        except (ImportError, Exception) as exc:
            logger.info(
                "OOD detector not available: %s. OOD flagging will be "
                "skipped.",
                exc,
            )

        return None

    # ------------------------------------------------------------------
    # Single-well processing
    # ------------------------------------------------------------------

    def process_well(self, well_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a single well file end-to-end.

        Steps
        -----
        1. Read the LAS or CSV file.
        2. Standardise mnemonics to canonical names.
        3. Validate data quality.
        4. Determine available logs and select the best model tier.
        5. Load model bundles for the selected tier.
        6. Run predictions with all available algorithms.
        7. Select the best algorithm's predictions.
        8. Post-process: smooth, assign confidence/OOD flags, QC status.
        9. Generate per-well QC summary.

        Parameters
        ----------
        well_path : str or Path
            Path to the well file (``.las`` or ``.csv``).

        Returns
        -------
        dict
            Result dictionary with keys:

            - ``"well_name"`` : str
            - ``"uwi"`` : str
            - ``"source_path"`` : str
            - ``"status"`` : ``"success"`` or ``"failed"``
            - ``"error"`` : str or ``None``
            - ``"tier_used"`` : str
            - ``"algorithm"`` : str
            - ``"predictions"`` : pd.DataFrame or ``None``
            - ``"qc_summary"`` : dict
            - ``"metadata"`` : dict
        """
        well_path = Path(well_path).resolve()
        well_name = well_path.stem
        result: Dict[str, Any] = {
            "well_name": well_name,
            "uwi": "",
            "source_path": str(well_path),
            "status": "failed",
            "error": None,
            "tier_used": "",
            "algorithm": "",
            "predictions": None,
            "qc_summary": {},
            "metadata": {},
        }

        logger.info("=" * 70)
        logger.info("Processing well: %s (%s)", well_name, well_path)
        logger.info("=" * 70)

        # ---- Step 1: Read --------------------------------------------------
        ext = well_path.suffix.lower()
        if ext in (".las",):
            raw = read_las(well_path)
        elif ext in (".csv",):
            raw = read_csv_well(well_path)
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}' for well {well_path}. "
                "Expected .las or .csv."
            )

        metadata = raw.get("metadata", {})
        curves_df = raw["curves"]
        well_name = metadata.get("well_name", well_name) or well_name
        result["well_name"] = well_name
        result["uwi"] = metadata.get("uwi", "")
        result["metadata"] = metadata

        if curves_df.empty:
            raise ValueError(
                f"Well '{well_name}' has no curve data after reading."
            )

        logger.info(
            "Read %d samples, %d curves for well '%s'.",
            len(curves_df),
            len(curves_df.columns),
            well_name,
        )

        # ---- Step 2: Standardise mnemonics ---------------------------------
        # Ensure DEPTH is a column (not just index) for downstream use.
        if curves_df.index.name == DEPTH:
            curves_df = curves_df.reset_index()

        curves_df, mapping_report = standardize_columns(
            curves_df, self.mnemonic_map
        )
        logger.info(
            "Mnemonic mapping for '%s': %s",
            well_name,
            {k: v for k, v in mapping_report.items() if v != "unmapped"},
        )

        # ---- Step 3: Validate data -----------------------------------------
        validate_well = self._get_validator()
        try:
            validation = validate_well(curves_df, self._ranges_config)
            if not validation.get("valid", True):
                issues = validation.get("issues", [])
                logger.warning(
                    "Validation issues for '%s': %s",
                    well_name,
                    issues,
                )
        except Exception:
            logger.warning(
                "Validation failed for '%s'; proceeding anyway.",
                well_name,
                exc_info=True,
            )

        # ---- Step 4: Select best tier --------------------------------------
        # Use the five core canonical logs for tier routing.
        core_logs = ["GR", "RESD", "RHOB", "NPHI", "DTC"]
        available_logs = determine_available_logs(
            curves_df,
            core_logs,
            min_coverage=self._min_log_coverage,
        )

        tier_result = select_best_tier(
            available_logs,
            self.tiers_config["tiers"],
        )

        if tier_result is None:
            raise ValueError(
                f"No model tier can be satisfied for well '{well_name}'. "
                f"Available logs: {available_logs}."
            )

        tier_name, tier_dict = tier_result
        result["tier_used"] = tier_name
        logger.info(
            "Well '%s': selected tier '%s' (%s).",
            well_name,
            tier_name,
            tier_dict.get("name", ""),
        )

        # ---- Step 5: Load model bundles ------------------------------------
        tier_models = load_tier_models(self._artifacts_dir, tier_name)
        if not tier_models:
            raise RuntimeError(
                f"No model bundles found for tier '{tier_name}' in "
                f"{self._artifacts_dir}."
            )

        # Get a reference bundle for transformer/metadata.
        reference_bundle = next(iter(tier_models.values()))

        # ---- Prepare well data for prediction ------------------------------
        # Keep only DEPTH + required canonical logs (plus any extra that the
        # transformer might need).
        required_logs = tier_dict.get("required_logs", [])
        keep_cols = [DEPTH] + [c for c in required_logs if c in curves_df.columns]
        # Also keep any extra canonical logs that may be present.
        for c in curves_df.columns:
            if c in CANONICAL_LOGS and c not in keep_cols:
                keep_cols.append(c)
        well_df = curves_df[[c for c in keep_cols if c in curves_df.columns]].copy()

        # ---- Step 6: Predict with all algorithms ---------------------------
        transformer = self._get_transformer(tier_name, reference_bundle)
        all_predictions = predict_all_algorithms(
            well_df, tier_models, transformer, self.config
        )
        if not all_predictions:
            raise RuntimeError(
                f"All algorithms failed for well '{well_name}', "
                f"tier '{tier_name}'."
            )

        # ---- Step 7: Select best prediction --------------------------------
        best_algo, best_preds = select_best_prediction(all_predictions)
        result["algorithm"] = best_algo
        logger.info(
            "Well '%s': best algorithm = '%s'.",
            well_name,
            best_algo,
        )

        # ---- Step 8: Post-process ------------------------------------------
        # 8a. Modal smoothing
        if "PREDICTED_FACIES" in best_preds.columns:
            best_preds["PREDICTED_FACIES"] = modal_filter(
                best_preds["PREDICTED_FACIES"],
                window=self._smoothing_window,
            )

        # 8b. Confidence flags
        best_preds = assign_confidence_flags(
            best_preds,
            threshold=self._confidence_threshold,
        )

        # 8c. OOD flags
        ood_detector = self._get_ood_detector(reference_bundle)
        if ood_detector is not None:
            # Build the feature matrix used for OOD detection.
            try:
                transform_result = transformer.transform(
                    well_df.drop(columns=[DEPTH], errors="ignore")
                )
                # transformer.transform() returns (DataFrame, list) tuple
                if isinstance(transform_result, tuple):
                    well_features, _feat_cols = transform_result
                else:
                    well_features = transform_result
                best_preds = assign_ood_flags(
                    best_preds, well_features, ood_detector
                )
            except Exception:
                logger.warning(
                    "OOD flagging failed for '%s'; skipping.",
                    well_name,
                    exc_info=True,
                )
                best_preds["OOD_FLAG"] = False
        else:
            best_preds["OOD_FLAG"] = False

        # 8d. Combined QC status
        best_preds = assign_qc_status(best_preds)

        result["predictions"] = best_preds

        # ---- Step 9: QC summary --------------------------------------------
        qc_summary = compute_well_summary(
            best_preds,
            well_name=well_name,
            tier_used=tier_name,
            algorithm=best_algo,
        )
        result["qc_summary"] = qc_summary
        result["status"] = "success"

        logger.info(
            "Well '%s' processed successfully: tier=%s, algorithm=%s, "
            "n_predictions=%d, qc_grade=%s.",
            well_name,
            tier_name,
            best_algo,
            len(best_preds),
            qc_summary.get("overall_qc_grade", "N/A"),
        )

        return result

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def run_batch(
        self,
        inbox_dir: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """Process all well files in *inbox_dir* and write outputs.

        Steps
        -----
        1. Scan *inbox_dir* for LAS/CSV files.
        2. Process each well via :meth:`process_well` (with per-well
           try/except).
        3. Move successful wells to ``processed/`` sub-directory.
        4. Move failed wells to ``failed/`` sub-directory.
        5. Write per-well CSV outputs and JSON reports.
        6. Write batch summary CSV.
        7. Write batch QC report JSON.

        Parameters
        ----------
        inbox_dir : str or Path
            Folder containing well files to process.
        output_dir : str or Path
            Root output directory.  Sub-directories will be created
            automatically.

        Returns
        -------
        list of dict
            One result dictionary per well (same structure as
            :meth:`process_well` output).
        """
        inbox_dir = Path(inbox_dir).resolve()
        output_dir = Path(output_dir).resolve()

        # Derive processed/failed directories from config or defaults.
        processed_dir = self._resolve_path(
            self.config.get("paths", {}).get(
                "wells_processed", "data/wells/processed"
            )
        )
        failed_dir = self._resolve_path(
            self.config.get("paths", {}).get(
                "wells_failed", "data/wells/failed"
            )
        )

        # Ensure directories exist.
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        failed_dir.mkdir(parents=True, exist_ok=True)

        # ---- Step 1: Scan inbox --------------------------------------------
        well_files = scan_wells_folder(inbox_dir)
        total = len(well_files)

        if total == 0:
            logger.warning(
                "No well files found in inbox: %s. Nothing to process.",
                inbox_dir,
            )
            return []

        logger.info(
            "Batch run started: %d well file(s) found in %s.",
            total,
            inbox_dir,
        )
        batch_start = datetime.now(timezone.utc)

        all_results: List[Dict[str, Any]] = []

        # ---- Step 2: Process each well ------------------------------------
        for idx, well_info in enumerate(well_files, start=1):
            well_path = Path(well_info["path"])
            well_name = well_path.stem

            self._log_progress(idx, total, well_name, "STARTED")

            try:
                result = self.process_well(well_path)
            except Exception as exc:
                # Catch-all: never crash the batch.
                tb = traceback.format_exc()
                logger.error(
                    "Well '%s' FAILED with exception: %s\n%s",
                    well_name,
                    exc,
                    tb,
                )
                result = {
                    "well_name": well_name,
                    "uwi": "",
                    "source_path": str(well_path),
                    "status": "failed",
                    "error": str(exc),
                    "tier_used": "",
                    "algorithm": "",
                    "predictions": None,
                    "qc_summary": {},
                    "metadata": {},
                }

            all_results.append(result)
            status = result.get("status", "unknown")

            # ---- Step 3/4: Move well file ----------------------------------
            if status == "success":
                dest = processed_dir / well_path.name
                self._safe_move(well_path, dest)
                self._log_progress(idx, total, well_name, "SUCCESS")
            else:
                dest = failed_dir / well_path.name
                self._safe_move(well_path, dest)
                self._log_progress(idx, total, well_name, "FAILED")

            # ---- Step 5: Write per-well outputs ----------------------------
            if status == "success" and result.get("predictions") is not None:
                preds_df = result["predictions"]
                safe_name = self._safe_filename(result["well_name"])

                # CSV output.
                csv_path = output_dir / f"{safe_name}_predictions.csv"
                try:
                    write_predictions_csv(preds_df, csv_path)
                except Exception:
                    logger.exception(
                        "Failed to write predictions CSV for '%s'.",
                        well_name,
                    )

                # LAS output (if original was LAS and config enables it).
                output_cfg = self.config.get("output", {})
                if (
                    output_cfg.get("save_las", False)
                    and well_path.suffix.lower() == ".las"
                ):
                    las_path = output_dir / f"{safe_name}_predictions.las"
                    try:
                        write_predictions_las(preds_df, well_path, las_path)
                    except Exception:
                        logger.exception(
                            "Failed to write predictions LAS for '%s'.",
                            well_name,
                        )

            # JSON report for every well (success or failure).
            try:
                write_well_report(result, output_dir / "reports")
            except Exception:
                logger.exception(
                    "Failed to write well report for '%s'.", well_name
                )

        # ---- Step 6: Batch summary ----------------------------------------
        batch_end = datetime.now(timezone.utc)
        summary_path = output_dir / "batch_summary.csv"
        try:
            write_batch_summary(all_results, summary_path)
        except Exception:
            logger.exception("Failed to write batch summary.")

        # ---- Step 7: Batch QC report --------------------------------------
        n_success = sum(1 for r in all_results if r["status"] == "success")
        n_failed = sum(1 for r in all_results if r["status"] == "failed")

        batch_qc = {
            "batch_start_utc": batch_start.isoformat(),
            "batch_end_utc": batch_end.isoformat(),
            "duration_seconds": (batch_end - batch_start).total_seconds(),
            "total_wells": total,
            "succeeded": n_success,
            "failed": n_failed,
            "success_rate": round(
                100.0 * n_success / max(total, 1), 2
            ),
            "tier_distribution": self._count_field(all_results, "tier_used"),
            "algorithm_distribution": self._count_field(
                all_results, "algorithm"
            ),
            "qc_grade_distribution": self._count_qc_grades(all_results),
            "failed_wells": [
                {
                    "well_name": r["well_name"],
                    "error": r.get("error", "unknown"),
                }
                for r in all_results
                if r["status"] == "failed"
            ],
        }

        qc_report_path = output_dir / "batch_qc_report.json"
        try:
            with open(qc_report_path, "w") as fh:
                json.dump(batch_qc, fh, indent=2, default=str)
            logger.info("Wrote batch QC report: %s", qc_report_path)
        except Exception:
            logger.exception("Failed to write batch QC report.")

        logger.info(
            "Batch run complete: %d / %d succeeded (%.1f%%) in %.1f s.",
            n_success,
            total,
            100.0 * n_success / max(total, 1),
            (batch_end - batch_start).total_seconds(),
        )

        return all_results

    # ------------------------------------------------------------------
    # Progress logging
    # ------------------------------------------------------------------

    def _log_progress(
        self,
        current: int,
        total: int,
        well_name: str,
        status: str,
    ) -> None:
        """Log a progress update for the batch run.

        Parameters
        ----------
        current : int
            1-based index of the current well.
        total : int
            Total number of wells in the batch.
        well_name : str
            Name of the current well.
        status : str
            Status string (e.g. ``"STARTED"``, ``"SUCCESS"``, ``"FAILED"``).
        """
        pct = 100.0 * current / max(total, 1)
        logger.info(
            "[%d / %d  %.0f%%] %s — %s",
            current,
            total,
            pct,
            well_name,
            status,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_move(src: Path, dest: Path) -> None:
        """Move a file, handling name collisions by appending a suffix."""
        if not src.is_file():
            logger.debug("Source file does not exist; skipping move: %s", src)
            return

        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            # Append a timestamp to avoid overwriting.
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            dest = dest.with_stem(f"{dest.stem}_{ts}")

        try:
            shutil.move(str(src), str(dest))
            logger.debug("Moved %s -> %s", src, dest)
        except Exception:
            logger.warning(
                "Could not move %s -> %s. File remains in inbox.",
                src,
                dest,
                exc_info=True,
            )

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Sanitise a well name for use as a filename."""
        return "".join(
            c if (c.isalnum() or c in "-_") else "_" for c in str(name)
        )

    @staticmethod
    def _count_field(results: List[Dict], field: str) -> Dict[str, int]:
        """Count unique values of *field* across successful results."""
        counts: Dict[str, int] = {}
        for r in results:
            if r.get("status") == "success":
                val = r.get(field, "")
                if val:
                    counts[val] = counts.get(val, 0) + 1
        return counts

    @staticmethod
    def _count_qc_grades(results: List[Dict]) -> Dict[str, int]:
        """Count overall QC grades from the qc_summary of each result."""
        counts: Dict[str, int] = {}
        for r in results:
            grade = r.get("qc_summary", {}).get("overall_qc_grade", "")
            if grade:
                counts[grade] = counts.get(grade, 0) + 1
        return counts
