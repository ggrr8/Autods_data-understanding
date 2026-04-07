from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class AgentConfig:
    output_dir: str
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    dataset_name: Optional[str] = None
    random_state: int = 42

    # Optional LLM enhancement
    use_llm_insights: bool = False
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0


class DataUnderstandingAgent:
    """
    AUTODS Data Understanding Agent

    Responsibilities:
    - Profile raw tabular data already loaded in memory
    - Infer schema and feature types
    - Diagnose data quality issues
    - Analyse target column when available
    - Export structured JSON artifacts for downstream agents
    - Optionally expose an LLM insights interface
    """

    AGENT_NAME = "AUTODS_DATA_UNDERSTANDING"

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("run() expects a pandas DataFrame.")

            data_profile = self._build_data_profile(df)
            data_quality_report = self._build_data_quality_report(df)
            target_analysis = self._build_target_analysis(df)
            data_understanding_summary = self._build_summary(
                df=df,
                data_profile=data_profile,
                data_quality_report=data_quality_report,
                target_analysis=target_analysis,
            )

            llm_insights = None
            generated_files = [
                "data_profile.json",
                "data_quality_report.json",
                "target_analysis.json",
                "data_understanding_summary.json",
                "data_understanding_metadata.json",
            ]

            if self.config.use_llm_insights:
                llm_insights = self._generate_llm_insights(
                    data_profile=data_profile,
                    data_quality_report=data_quality_report,
                    target_analysis=target_analysis,
                )
                self._write_json("llm_insights.json", llm_insights)
                generated_files.append("llm_insights.json")

            metadata = self._build_metadata(df, generated_files)

            self._write_json("data_profile.json", data_profile)
            self._write_json("data_quality_report.json", data_quality_report)
            self._write_json("target_analysis.json", target_analysis)
            self._write_json(
                "data_understanding_summary.json", data_understanding_summary
            )
            self._write_json("data_understanding_metadata.json", metadata)

            return {
                "status": "success",
                "agent_name": self.AGENT_NAME,
                "output_dir": str(self.output_dir.resolve()),
                "generated_files": generated_files,
                "result": {
                    "data_profile": data_profile,
                    "data_quality_report": data_quality_report,
                    "target_analysis": target_analysis,
                    "data_understanding_summary": data_understanding_summary,
                    "metadata": metadata,
                    "llm_insights": llm_insights,
                },
            }

        except Exception as e:
            return {
                "status": "failure",
                "agent_name": self.AGENT_NAME,
                "error_message": str(e),
            }

    @staticmethod
    def load_dataframe(data_path: str) -> pd.DataFrame:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)

        raise ValueError("Unsupported file type. Only CSV and Parquet are supported.")

    def _build_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        feature_types = self._infer_feature_types(df)

        numeric_stats = {}
        for col in feature_types["numeric_columns"]:
            s = pd.to_numeric(df[col], errors="coerce")
            numeric_stats[col] = {
                "count": self._safe_int(s.notna().sum()),
                "mean": self._safe_float(s.mean()),
                "std": self._safe_float(s.std()),
                "min": self._safe_float(s.min()),
                "p25": self._safe_float(s.quantile(0.25)),
                "median": self._safe_float(s.median()),
                "p75": self._safe_float(s.quantile(0.75)),
                "max": self._safe_float(s.max()),
            }

        categorical_preview = {}
        for col in feature_types["categorical_columns"][:20]:
            vc = df[col].astype("object").fillna("__MISSING__").value_counts().head(10)
            categorical_preview[col] = {
                str(k): self._safe_int(v) for k, v in vc.to_dict().items()
            }

        return {
            "dataset_name": self._resolve_dataset_name(),
            "shape": {
                "rows": self._safe_int(df.shape[0]),
                "columns": self._safe_int(df.shape[1]),
            },
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "feature_types": feature_types,
            "numeric_summary_statistics": numeric_stats,
            "categorical_value_preview": categorical_preview,
            "memory_usage_bytes": self._safe_int(df.memory_usage(deep=True).sum()),
        }

    def _build_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        missing_counts = df.isnull().sum()
        missing_ratios = df.isnull().mean().round(6)

        duplicate_rows = self._safe_int(df.duplicated().sum())
        constant_columns = [
            col for col in df.columns if df[col].nunique(dropna=False) <= 1
        ]
        all_missing_columns = [col for col in df.columns if df[col].isnull().all()]
        suspected_identifier_columns = self._detect_identifier_columns(df)
        high_cardinality_columns = self._detect_high_cardinality_columns(df)
        numeric_outliers = self._detect_numeric_outliers(df)

        high_missing_columns = [
            col for col in df.columns if float(df[col].isnull().mean()) >= 0.30
        ]

        leakage_risk_columns = []
        if self.config.target_column and self.config.target_column in df.columns:
            target_lower = self.config.target_column.lower()
            for col in df.columns:
                if col == self.config.target_column:
                    continue
                name_lower = col.lower()
                if (
                    target_lower in name_lower
                    or "target" in name_lower
                    or "label" in name_lower
                ):
                    leakage_risk_columns.append(col)

        recommended_actions = {
            "drop_or_review_constant_columns": constant_columns,
            "drop_or_review_all_missing_columns": all_missing_columns,
            "review_high_missing_columns": high_missing_columns,
            "review_identifier_columns_for_leakage": suspected_identifier_columns,
            "review_high_cardinality_columns_for_encoding": high_cardinality_columns,
            "review_possible_target_leakage_columns": leakage_risk_columns,
        }

        return {
            "missing_values": {
                "missing_count_by_column": {
                    col: self._safe_int(v)
                    for col, v in missing_counts.to_dict().items()
                },
                "missing_ratio_by_column": {
                    col: self._safe_float(v)
                    for col, v in missing_ratios.to_dict().items()
                },
            },
            "duplicate_rows": {
                "count": duplicate_rows,
                "ratio": (
                    self._safe_float(duplicate_rows / len(df)) if len(df) > 0 else 0.0
                ),
            },
            "constant_columns": constant_columns,
            "all_missing_columns": all_missing_columns,
            "high_missing_columns": high_missing_columns,
            "suspected_identifier_columns": suspected_identifier_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "numeric_outlier_report": numeric_outliers,
            "recommended_actions": recommended_actions,
        }

    def _build_target_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        target_column = self.config.target_column

        if not target_column:
            return {
                "target_column": None,
                "problem_type": self.config.problem_type,
                "inferred_problem_type": None,
                "status": "no_target_provided",
                "message": "Target analysis skipped because no target column was provided.",
            }

        if target_column not in df.columns:
            return {
                "target_column": target_column,
                "problem_type": self.config.problem_type,
                "inferred_problem_type": None,
                "status": "target_not_found",
                "message": f"Target column '{target_column}' was not found in dataset.",
            }

        y = df[target_column]
        inferred_problem_type = self._infer_problem_type(y)
        resolved_problem_type = self.config.problem_type or inferred_problem_type

        result = {
            "target_column": target_column,
            "problem_type": resolved_problem_type,
            "inferred_problem_type": inferred_problem_type,
            "missing_count": self._safe_int(y.isnull().sum()),
            "missing_ratio": self._safe_float(y.isnull().mean()),
            "n_unique": self._safe_int(y.nunique(dropna=True)),
            "status": "success",
        }

        if resolved_problem_type == "classification":
            vc = y.astype("object").fillna("__MISSING__").value_counts(dropna=False)
            class_dist = {str(k): self._safe_int(v) for k, v in vc.to_dict().items()}
            valid_counts = [v for v in class_dist.values() if v > 0]
            imbalance_ratio = (
                max(valid_counts) / min(valid_counts) if len(valid_counts) >= 2 else 1.0
            )

            result.update(
                {
                    "class_distribution": class_dist,
                    "imbalance_ratio_max_over_min": self._safe_float(imbalance_ratio),
                    "is_binary": self._safe_bool(y.nunique(dropna=True) == 2),
                    "recommended_primary_metrics": [
                        "f1",
                        "roc_auc",
                        "precision",
                        "recall",
                    ],
                }
            )
        else:
            y_num = pd.to_numeric(y, errors="coerce")
            result.update(
                {
                    "summary_statistics": {
                        "count": self._safe_int(y_num.notna().sum()),
                        "mean": self._safe_float(y_num.mean()),
                        "std": self._safe_float(y_num.std()),
                        "min": self._safe_float(y_num.min()),
                        "p25": self._safe_float(y_num.quantile(0.25)),
                        "median": self._safe_float(y_num.median()),
                        "p75": self._safe_float(y_num.quantile(0.75)),
                        "max": self._safe_float(y_num.max()),
                    },
                    "recommended_primary_metrics": ["rmse", "mae", "r2"],
                }
            )

        return result

    def _build_summary(
        self,
        df: pd.DataFrame,
        data_profile: Dict[str, Any],
        data_quality_report: Dict[str, Any],
        target_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        major_findings = []

        rows = df.shape[0]
        cols = df.shape[1]
        major_findings.append(f"Dataset contains {rows} rows and {cols} columns.")

        high_missing_columns = data_quality_report["high_missing_columns"]
        if high_missing_columns:
            major_findings.append(
                f"{len(high_missing_columns)} columns have at least 30% missing values and should be reviewed."
            )

        constant_columns = data_quality_report["constant_columns"]
        if constant_columns:
            major_findings.append(
                f"{len(constant_columns)} constant columns were detected and may be removable."
            )

        identifier_columns = data_quality_report["suspected_identifier_columns"]
        if identifier_columns:
            major_findings.append(
                f"{len(identifier_columns)} identifier-like columns were detected and should be reviewed for leakage."
            )

        if target_analysis.get("status") not in {
            "no_target_provided",
            "target_not_found",
        }:
            if target_analysis.get("problem_type") == "classification":
                ratio = target_analysis.get("imbalance_ratio_max_over_min", 1.0)
                major_findings.append(
                    f"Target is treated as classification with imbalance ratio {ratio:.3f}."
                )
            elif target_analysis.get("problem_type") == "regression":
                major_findings.append(
                    "Target is treated as regression based on configuration or inference."
                )

        downstream_handoff = {
            "cleaning_agent": {
                "priority_columns_for_imputation_or_review": high_missing_columns,
                "drop_candidate_columns": (
                    data_quality_report["all_missing_columns"]
                    + data_quality_report["constant_columns"]
                ),
            },
            "feature_engineering_agent": {
                "categorical_columns": data_profile["feature_types"][
                    "categorical_columns"
                ],
                "numeric_columns": data_profile["feature_types"]["numeric_columns"],
                "high_cardinality_columns": data_quality_report[
                    "high_cardinality_columns"
                ],
                "suspected_identifier_columns": data_quality_report[
                    "suspected_identifier_columns"
                ],
            },
            "modelling_agent": {
                "target_column": self.config.target_column,
                "problem_type": target_analysis.get("problem_type"),
                "class_imbalance_flag": (
                    target_analysis.get("imbalance_ratio_max_over_min", 1.0) >= 3.0
                    if target_analysis.get("problem_type") == "classification"
                    else False
                ),
                "recommended_metrics": target_analysis.get(
                    "recommended_primary_metrics", []
                ),
            },
        }

        return {
            "status": "success",
            "dataset_name": self._resolve_dataset_name(),
            "executive_summary": self._generate_executive_summary(
                data_profile=data_profile,
                data_quality_report=data_quality_report,
                target_analysis=target_analysis,
            ),
            "major_findings": major_findings,
            "primary_risks": self._build_primary_risks(
                data_quality_report, target_analysis
            ),
            "recommended_next_steps": self._build_next_steps(
                data_quality_report, target_analysis
            ),
            "downstream_handoff": downstream_handoff,
        }

    def _build_metadata(
        self, df: pd.DataFrame, generated_files: List[str]
    ) -> Dict[str, Any]:
        return {
            "agent_name": self.AGENT_NAME,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_name": self._resolve_dataset_name(),
            "output_dir": str(self.output_dir.resolve()),
            "target_column": self.config.target_column,
            "problem_type_config": self.config.problem_type,
            "random_state": self.config.random_state,
            "shape": {
                "rows": self._safe_int(df.shape[0]),
                "columns": self._safe_int(df.shape[1]),
            },
            "generated_files": generated_files,
            "use_llm_insights": self.config.use_llm_insights,
            "llm_model": (
                self.config.llm_model if self.config.use_llm_insights else None
            ),
        }

    def _infer_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        boolean_columns = df.select_dtypes(include=["bool"]).columns.tolist()
        datetime_columns = df.select_dtypes(
            include=["datetime64[ns]", "datetime64[ns, UTC]", "datetimetz"]
        ).columns.tolist()

        categorical_columns = [
            col
            for col in df.columns
            if col not in numeric_columns + boolean_columns + datetime_columns
        ]

        return {
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "boolean_columns": boolean_columns,
            "datetime_columns": datetime_columns,
        }

    def _detect_identifier_columns(self, df: pd.DataFrame) -> List[str]:
        suspected = []
        n_rows = len(df)

        for col in df.columns:
            name_lower = col.lower()
            unique_ratio = df[col].nunique(dropna=True) / max(n_rows, 1)

            if (
                name_lower == "id"
                or name_lower.endswith("_id")
                or "uuid" in name_lower
                or "identifier" in name_lower
                or unique_ratio >= 0.95
            ):
                suspected.append(col)

        return suspected

    def _detect_high_cardinality_columns(self, df: pd.DataFrame) -> List[str]:
        high_card_cols = []
        feature_types = self._infer_feature_types(df)

        for col in feature_types["categorical_columns"]:
            nunique = df[col].nunique(dropna=True)
            ratio = nunique / max(len(df), 1)
            if nunique >= 20 or ratio >= 0.30:
                high_card_cols.append(col)

        return high_card_cols

    def _detect_numeric_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        report = {}
        feature_types = self._infer_feature_types(df)

        for col in feature_types["numeric_columns"]:
            s = pd.to_numeric(df[col], errors="coerce").dropna()

            if len(s) < 5:
                report[col] = {
                    "status": "skipped_too_few_values",
                    "outlier_count": 0,
                    "outlier_ratio": 0.0,
                }
                continue

            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                report[col] = {
                    "status": "skipped_zero_iqr",
                    "outlier_count": 0,
                    "outlier_ratio": 0.0,
                }
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((s < lower) | (s > upper)).sum()

            report[col] = {
                "status": "completed",
                "lower_bound": self._safe_float(lower),
                "upper_bound": self._safe_float(upper),
                "outlier_count": self._safe_int(outliers),
                "outlier_ratio": self._safe_float(outliers / len(s)),
            }

        return report

    def _infer_problem_type(self, y: pd.Series) -> str:
        """Pure inference only. Does not read config."""
        if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 20:
            return "regression"
        return "classification"

    def _generate_executive_summary(
        self,
        data_profile: Dict[str, Any],
        data_quality_report: Dict[str, Any],
        target_analysis: Dict[str, Any],
    ) -> str:
        rows = data_profile["shape"]["rows"]
        cols = data_profile["shape"]["columns"]
        num_cols = len(data_profile["feature_types"]["numeric_columns"])
        cat_cols = len(data_profile["feature_types"]["categorical_columns"])

        summary = (
            f"The dataset contains {rows} rows and {cols} columns, "
            f"including {num_cols} numeric columns and {cat_cols} categorical columns. "
        )

        if data_quality_report["high_missing_columns"]:
            summary += (
                f"There are {len(data_quality_report['high_missing_columns'])} columns with high missingness "
                f"that require review before downstream modelling. "
            )

        if data_quality_report["suspected_identifier_columns"]:
            summary += "Identifier-like columns were detected and should be assessed for leakage risk. "

        if target_analysis.get("status") == "no_target_provided":
            summary += "No target column was provided, so target-specific analysis was skipped."
        elif target_analysis.get("status") == "target_not_found":
            summary += "The configured target column was not found in the dataset."
        else:
            summary += (
                f"The target is treated as {target_analysis.get('problem_type')} "
                f"for downstream workflow planning."
            )

        return summary

    def _generate_llm_insights(
        self,
        data_profile: Dict[str, Any],
        data_quality_report: Dict[str, Any],
        target_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "LLM insights were enabled, but no real LLM client has been implemented yet."
        )

    def _build_primary_risks(
        self,
        data_quality_report: Dict[str, Any],
        target_analysis: Dict[str, Any],
    ) -> List[str]:
        risks = []

        if data_quality_report["high_missing_columns"]:
            risks.append(
                "High missingness may reduce usable signal or introduce unstable imputation."
            )
        if data_quality_report["suspected_identifier_columns"]:
            risks.append(
                "Identifier-like columns may create leakage or spurious predictive performance."
            )
        if data_quality_report["high_cardinality_columns"]:
            risks.append(
                "High-cardinality categorical variables may require dedicated encoding strategy."
            )
        if (
            target_analysis.get("problem_type") == "classification"
            and target_analysis.get("imbalance_ratio_max_over_min", 1.0) >= 3.0
        ):
            risks.append(
                "Class imbalance may distort naive accuracy and requires metric selection discipline."
            )

        return risks

    def _build_next_steps(
        self,
        data_quality_report: Dict[str, Any],
        target_analysis: Dict[str, Any],
    ) -> List[str]:
        steps = [
            "Review missingness pattern and define imputation strategy before feature engineering.",
            "Review identifier-like fields and exclude leakage-prone columns before modelling.",
            "Confirm target column semantics with business context before downstream training.",
        ]

        if data_quality_report["high_cardinality_columns"]:
            steps.append(
                "Prepare encoding plan for high-cardinality categorical columns."
            )
        if (
            target_analysis.get("problem_type") == "classification"
            and target_analysis.get("imbalance_ratio_max_over_min", 1.0) >= 3.0
        ):
            steps.append(
                "Use stratified splitting and imbalance-aware metrics for downstream modelling."
            )

        return steps

    def _write_json(self, filename: str, payload: Dict[str, Any]) -> None:
        output_path = self.output_dir / filename
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                indent=2,
                ensure_ascii=False,
                default=self._json_default,
            )

    def _resolve_dataset_name(self) -> str:
        return self.config.dataset_name or "in_memory_dataset"

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return str(obj)

    @staticmethod
    def _safe_int(value: Any) -> int:
        if pd.isna(value):
            return 0
        return int(value)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _safe_bool(value: Any) -> bool:
        return bool(value)
