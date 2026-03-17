# AUTODS Data Understanding Agent

This module implements the upstream data understanding stage for the AUTODS workflow.

## Current scope

- Raw tabular dataset ingestion
- Schema profiling and feature typing
- Data quality diagnostics
- Target analysis for classification or regression
- Standardised artifact export for downstream agent handoff

## Current phase

This version is the first deliverable data-understanding agent for the project.

Primary objective:

- Deliver a stable and reproducible data-understanding workflow that can be chained before cleaning, feature engineering, and modelling.

Secondary objective:

- Preserve clean interfaces and artifact semantics for future workflow orchestration.

## Required input

- Raw dataset file in CSV or Parquet format

## Optional input

- `target_column`
- `problem_type`
- `dataset_name`

## Runtime configuration

- `data_path`
- `target_column`
- `problem_type`
- `output_dir`
- `dataset_name`
- `random_state`

## Current downstream outputs

- `data_profile.json`
- `data_quality_report.json`
- `target_analysis.json`
- `data_understanding_summary.json`
- `data_understanding_metadata.json`

## Output semantics

### Core outputs

- `data_profile.json`: dataset shape, dtypes, feature types, descriptive profile
- `data_quality_report.json`: missingness, duplicates, constant columns, identifier risks, outlier report
- `target_analysis.json`: target distribution or regression summary, imbalance analysis, task interpretation

### Metadata outputs

- `data_understanding_summary.json`: executive summary, major findings, risks, recommended next steps, downstream handoff
- `data_understanding_metadata.json`: run configuration and generated file tracking

## Handoff to downstream agents

### Cleaning agent

Consume:
- missingness diagnostics
- all-missing columns
- constant columns
- duplicate information

### Feature Engineering agent

Consume:
- numeric columns
- categorical columns
- high-cardinality columns
- suspected identifier columns

### Modelling agent

Consume:
- target column
- inferred or configured problem type
- class imbalance diagnostics
- recommended metrics

## Design principle

- Python handles deterministic data profiling and artifact export.
- The agent produces structured outputs as source-of-truth artifacts for downstream stages.
- Current phase prioritises reproducibility and clean workflow handoff over broad data modality support.

## Current limitations

- Only CSV and Parquet are supported.
- Datetime semantic parsing is currently conservative.
- Leakage checks are heuristic and should be reviewed with domain context.
- This stage does not perform cleaning or feature transformation.

## Run example

```bash
python main.py \
  --data-path mydata.csv \
  --target-column target \
  --problem-type classification \
  --output-dir data_understanding_outputs