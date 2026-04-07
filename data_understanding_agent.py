from __future__ import annotations

import json
from dataclasses import dataclass, field, replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

    # Optional business context supplied by an upstream planner
    # Keys: normalised_description, primary_metric, reasoning,
    #       constraints (dict), focus_columns (list)
    business_context: Optional[Dict] = None


@dataclass
class PlannerInput:
    """Lightweight planner contract for the Data Understanding Agent."""

    source: str = "unknown"
    schema_version: str = "1.0"
    rationale: str = ""

    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    dataset_name: Optional[str] = None

    drop_columns: Optional[List[str]] = None
    use_columns: Optional[List[str]] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannerInput":
        known_keys = {
            "source", "schema_version", "rationale", "target_column",
            "problem_type", "dataset_name", "drop_columns", "use_columns",
        }
        return cls(
            source=data.get("source", "unknown"),
            schema_version=data.get("schema_version", "1.0"),
            rationale=data.get("rationale", ""),
            target_column=data.get("target_column"),
            problem_type=data.get("problem_type"),
            dataset_name=data.get("dataset_name"),
            drop_columns=data.get("drop_columns"),
            use_columns=data.get("use_columns"),
            extra={k: v for k, v in data.items() if k not in known_keys},
        )

    @classmethod
    def from_json_file(cls, path: str) -> "PlannerInput":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def apply_to_config(self, config: AgentConfig) -> AgentConfig:
        overrides: Dict[str, Any] = {}
        if self.target_column is not None:
            overrides["target_column"] = self.target_column
        if self.problem_type is not None:
            overrides["problem_type"] = self.problem_type
        if self.dataset_name is not None:
            overrides["dataset_name"] = self.dataset_name
        return dataclass_replace(config, **overrides) if overrides else config


def load_planner_input(path: str) -> PlannerInput:
    planner_path = Path(path)
    if not planner_path.exists():
        raise FileNotFoundError(f"Planner input file not found: {path}")
    return PlannerInput.from_json_file(path)


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

    def __init__(
        self,
        config: AgentConfig,
        planner_input: Optional[PlannerInput] = None,
    ) -> None:
        self.planner_input_: Optional[PlannerInput] = planner_input

        if planner_input is not None:
            config = planner_input.apply_to_config(config)

        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("run() expects a pandas DataFrame.")

            df = self._apply_planner_feature_hints(df)

            data_profile = self._build_data_profile(df)
            data_quality_report = self._build_data_quality_report(df)
            target_analysis = self._build_target_analysis(df)

            # Business alignment — only produced when business_context is set
            business_alignment: Optional[Dict[str, Any]] = None
            if self.config.business_context:
                business_alignment = self._build_business_alignment(df, target_analysis)

            data_understanding_summary = self._build_summary(
                df=df,
                data_profile=data_profile,
                data_quality_report=data_quality_report,
                target_analysis=target_analysis,
                business_alignment=business_alignment,
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

            if self.planner_input_ is not None:
                self._write_json("planner_input.json", self._planner_input_as_dict())
                generated_files.append("planner_input.json")

            metadata = self._build_metadata(df, generated_files)

            self._write_json("data_profile.json", data_profile)
            self._write_json("data_quality_report.json", data_quality_report)
            self._write_json("target_analysis.json", target_analysis)
            self._write_json("data_understanding_summary.json", data_understanding_summary)
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
                    "business_alignment": business_alignment,
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

    def _apply_planner_feature_hints(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.planner_input_ is None:
            return df

        df = df.copy()

        if self.planner_input_.use_columns:
            keep_cols = [c for c in self.planner_input_.use_columns if c in df.columns]
            if self.config.target_column and self.config.target_column in df.columns:
                if self.config.target_column not in keep_cols:
                    keep_cols.append(self.config.target_column)
            if keep_cols:
                df = df[keep_cols].copy()

        elif self.planner_input_.drop_columns:
            drop_cols = [c for c in self.planner_input_.drop_columns if c in df.columns]
            if self.config.target_column and self.config.target_column in drop_cols:
                drop_cols.remove(self.config.target_column)
            if drop_cols:
                df = df.drop(columns=drop_cols).copy()

        return df

    def _planner_input_as_dict(self) -> Dict[str, Any]:
        if self.planner_input_ is None:
            return {}
        return {
            "source": self.planner_input_.source,
            "schema_version": self.planner_input_.schema_version,
            "rationale": self.planner_input_.rationale,
            "target_column": self.planner_input_.target_column,
            "problem_type": self.planner_input_.problem_type,
            "dataset_name": self.planner_input_.dataset_name,
            "drop_columns": self.planner_input_.drop_columns,
            "use_columns": self.planner_input_.use_columns,
            "extra": self.planner_input_.extra,
        }

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

        profile: Dict[str, Any] = {
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

        # Business-context-aware: detailed stats for focus_columns
        if self.config.business_context:
            focus_columns: List[str] = self.config.business_context.get("focus_columns", [])
            if focus_columns:
                focus_analysis: Dict[str, Any] = {}
                target_col = self.config.target_column
                for col in focus_columns:
                    if col not in df.columns:
                        continue
                    col_info: Dict[str, Any] = {"is_business_key_column": True}
                    if col in feature_types["numeric_columns"]:
                        s = pd.to_numeric(df[col], errors="coerce")
                        col_info["distribution"] = {
                            "p10": self._safe_float(s.quantile(0.10)),
                            "p25": self._safe_float(s.quantile(0.25)),
                            "p50": self._safe_float(s.quantile(0.50)),
                            "p75": self._safe_float(s.quantile(0.75)),
                            "p90": self._safe_float(s.quantile(0.90)),
                            "p99": self._safe_float(s.quantile(0.99)),
                        }
                        q1 = s.quantile(0.25)
                        q3 = s.quantile(0.75)
                        iqr = q3 - q1
                        if iqr > 0:
                            n_valid = len(s.dropna())
                            if n_valid > 0:
                                outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                                col_info["outlier_ratio"] = self._safe_float(outliers / n_valid)
                        if target_col and target_col in df.columns:
                            try:
                                y_num = pd.to_numeric(df[target_col], errors="coerce")
                                col_info["correlation_with_target"] = self._safe_float(
                                    s.corr(y_num)
                                )
                            except Exception:
                                pass
                    focus_analysis[col] = col_info
                profile["focus_columns_analysis"] = focus_analysis

        return profile

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
                    col: self._safe_int(v) for col, v in missing_counts.to_dict().items()
                },
                "missing_ratio_by_column": {
                    col: self._safe_float(v) for col, v in missing_ratios.to_dict().items()
                },
            },
            "duplicate_rows": {
                "count": duplicate_rows,
                "ratio": self._safe_float(duplicate_rows / len(df)) if len(df) > 0 else 0.0,
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

        result: Dict[str, Any] = {
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
            # Business-context-aware: warn when roc_auc is chosen with severe imbalance
            if self.config.business_context:
                primary_metric = self.config.business_context.get("primary_metric", "")
                if primary_metric == "roc_auc" and imbalance_ratio > 10.0:
                    result["primary_metric_note"] = (
                        f"当前数据分布下 roc_auc 可能虚高（类别不平衡比例 {imbalance_ratio:.1f}:1），"
                        "建议关注 f1"
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
        business_alignment: Optional[Dict[str, Any]] = None,
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

        if target_analysis.get("status") not in {"no_target_provided", "target_not_found"}:
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
                "categorical_columns": data_profile["feature_types"]["categorical_columns"],
                "numeric_columns": data_profile["feature_types"]["numeric_columns"],
                "high_cardinality_columns": data_quality_report["high_cardinality_columns"],
                "suspected_identifier_columns": data_quality_report["suspected_identifier_columns"],
            },
            "modelling_agent": {
                "target_column": self.config.target_column,
                "problem_type": target_analysis.get("problem_type"),
                "class_imbalance_flag": (
                    target_analysis.get("imbalance_ratio_max_over_min", 1.0) >= 3.0
                    if target_analysis.get("problem_type") == "classification"
                    else False
                ),
                "recommended_metrics": target_analysis.get("recommended_primary_metrics", []),
            },
        }

        summary: Dict[str, Any] = {
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

        # Only add when business_context was provided — keeps existing readers unbroken
        if business_alignment is not None:
            summary["business_alignment"] = business_alignment

        return summary

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
            "llm_model": self.config.llm_model if self.config.use_llm_insights else None,
            "planner_input_source": self.planner_input_.source if self.planner_input_ is not None else None,
            "planner_input_available": self.planner_input_ is not None,
            "business_context_provided": self.config.business_context is not None,
        }

    def _build_business_alignment(
        self, df: pd.DataFrame, target_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Produce the business_alignment block (section 1.3 of the spec).

        Only called when config.business_context is not None.
        """
        bc = self.config.business_context or {}
        target_column = self.config.target_column

        # target column existence (case-insensitive fallback)
        target_column_found = False
        target_column_actual = target_column
        if target_column:
            if target_column in df.columns:
                target_column_found = True
            else:
                for col in df.columns:
                    if col.lower() == target_column.lower():
                        target_column_found = True
                        target_column_actual = col
                        break

        # imbalance metrics
        imbalance_ratio = float(target_analysis.get("imbalance_ratio_max_over_min") or 1.0)
        class_dist = target_analysis.get("class_distribution", {})
        total_samples = sum(class_dist.values()) if class_dist else df.shape[0]
        majority_count = max(class_dist.values()) if class_dist else 0
        imbalance_ratio_majority = (
            self._safe_float(majority_count / total_samples) if total_samples > 0 else None
        )

        if imbalance_ratio >= 10.0:
            imbalance_severity = "severe"
        elif imbalance_ratio >= 3.0:
            imbalance_severity = "moderate"
        else:
            imbalance_severity = "none"

        # primary metric suitability
        primary_metric = bc.get("primary_metric", "")
        primary_metric_suitable = True
        metric_suitability_reason = ""
        if (
            primary_metric == "roc_auc"
            and target_analysis.get("problem_type") == "classification"
            and imbalance_ratio > 10.0
        ):
            primary_metric_suitable = False
            metric_suitability_reason = (
                f"严重类别不平衡（比例 {imbalance_ratio:.1f}:1），"
                "roc_auc 可能虚高，建议关注 f1"
            )

        # data volume assessment
        n_rows = df.shape[0]
        if n_rows >= 1000:
            data_volume_assessment = "sufficient"
            data_volume_reason = f"样本量 {n_rows} 足以支撑稳健建模"
        elif n_rows >= 200:
            data_volume_assessment = "marginal"
            data_volume_reason = f"样本量 {n_rows} 偏少，建模结果可能不稳定，建议增加数据"
        else:
            data_volume_assessment = "insufficient"
            data_volume_reason = f"样本量 {n_rows} 严重不足，建议优先扩充数据集后再建模"

        # business concerns
        business_concerns: List[str] = []
        if imbalance_severity == "severe":
            business_concerns.append(
                "目标列存在严重类别不平衡，需考虑过采样/欠采样策略或调整评估指标"
            )
        if data_volume_assessment == "marginal":
            business_concerns.append("数据量偏少，模型泛化能力可能受限，建议交叉验证评估稳定性")
        elif data_volume_assessment == "insufficient":
            business_concerns.append("数据量严重不足，模型结果可信度低，建议优先扩充数据集")
        if not primary_metric_suitable:
            business_concerns.append(metric_suitability_reason)

        return {
            "target_column_found": target_column_found,
            "target_column_actual": target_column_actual,
            "primary_metric_suitable": primary_metric_suitable,
            "metric_suitability_reason": metric_suitability_reason,
            "data_volume_assessment": data_volume_assessment,
            "data_volume_reason": data_volume_reason,
            "imbalance_severity": imbalance_severity,
            "imbalance_ratio": imbalance_ratio_majority,
            "business_concerns": business_concerns,
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
            summary += (
                "Identifier-like columns were detected and should be assessed for leakage risk. "
            )

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
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        """Build a structured LLM insights payload.

        When business_context is set the prompt includes a business context
        section (normalised_description, primary_metric, constraints) and
        requests a Business Relevance Assessment block from the LLM.

        Args:
            llm_fn: Optional callable (prompt: str) -> str.  If provided it
                    is invoked with the assembled prompt and its response is
                    stored under the llm_response key.  When None the prompt
                    is assembled but no call is made.

        Raises:
            NotImplementedError: when use_llm_insights=True but no llm_fn was
                                 supplied and no default client has been wired in.
        """
        prompt_parts: List[str] = [
            "You are a senior data scientist. Analyse the following dataset "
            "statistics and provide concise insights.\n",
            "## Dataset Overview",
            (
                f"Shape: {data_profile['shape']['rows']} rows x "
                f"{data_profile['shape']['columns']} columns"
            ),
            "",
            "## Data Quality Summary",
            f"High-missing columns (>=30%): {data_quality_report['high_missing_columns']}",
            f"Duplicate rows: {data_quality_report['duplicate_rows']['count']}",
            f"Suspected identifier columns: {data_quality_report['suspected_identifier_columns']}",
            "",
            "## Target Analysis",
            f"Problem type: {target_analysis.get('problem_type', 'unknown')}",
        ]

        if target_analysis.get("problem_type") == "classification":
            prompt_parts.append(
                f"Class imbalance ratio (max/min): "
                f"{target_analysis.get('imbalance_ratio_max_over_min', 'N/A')}"
            )

        # Inject business context when available
        if self.config.business_context:
            bc = self.config.business_context
            prompt_parts += [
                "",
                "## Business Context",
                f"Task description: {bc.get('normalised_description', 'N/A')}",
                f"Primary evaluation metric: {bc.get('primary_metric', 'N/A')}",
                f"Constraints: {bc.get('constraints', {})}",
                "",
                "Please include a **Business Relevance Assessment** section that "
                "evaluates whether the data quality and features support the stated "
                "business goal, and highlights any findings that require "
                "business-side attention.",
            ]

        prompt = "\n".join(prompt_parts)

        if llm_fn is not None:
            llm_response = llm_fn(prompt)
            return {"prompt": prompt, "llm_response": llm_response}

        raise NotImplementedError(
            "LLM insights were enabled, but no real LLM client has been implemented yet. "
            "Pass a callable llm_fn or wire in a client before calling run()."
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
