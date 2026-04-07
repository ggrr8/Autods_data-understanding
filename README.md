# AUTODS Data Understanding Agent

A modular data understanding component for a multi-agent automated data science pipeline.

---

## Overview

This repository implements the Data Understanding Agent, responsible for:

- profiling tabular datasets  
- identifying data quality issues  
- detecting target leakage risk columns  
- analysing the target column  
- generating downstream handoff context for cleaning, feature engineering, and modelling agents  
- producing structured JSON artifacts for pipeline consumption  

---

## Key Design

### Input Interface

The agent accepts an in-memory pandas DataFrame:

```python
agent.run(df)
```

File loading is handled externally (e.g. by `main.py`), keeping the agent decoupled from I/O.

---

### AgentConfig

```python
from data_understanding_agent import AgentConfig

config = AgentConfig(
    output_dir="data_understanding_outputs",  # required
    target_column="label",                    # optional
    problem_type="classification",            # optional: overrides inference
    dataset_name="my_dataset",               # optional: defaults to "in_memory_dataset"
    random_state=42,                          # optional

    # Optional LLM enhancement (reserved for future implementation)
    use_llm_insights=False,
    llm_model="gpt-4o-mini",
    llm_temperature=0.0,
)
```

---

### Output Artifacts

The agent writes the following JSON files to `output_dir`:

| File | Description |
|---|---|
| `data_profile.json` | Schema, dtypes, feature types, numeric stats, categorical previews |
| `data_quality_report.json` | Missing values, duplicates, outliers, recommended actions |
| `target_analysis.json` | Class distribution or regression statistics, recommended metrics |
| `data_understanding_summary.json` | Executive summary, major findings, risks, next steps, downstream handoff |
| `data_understanding_metadata.json` | Run metadata (timestamp, config, generated files) |
| `llm_insights.json` | *(only when `use_llm_insights=True`)* |

---

### Standardized Response

```json
{
  "status": "success" | "failure",
  "agent_name": "AUTODS_DATA_UNDERSTANDING",
  "output_dir": "/absolute/path/to/output_dir",
  "generated_files": ["data_profile.json", "..."],
  "result": {
    "data_profile": { ... },
    "data_quality_report": { ... },
    "target_analysis": { ... },
    "data_understanding_summary": { ... },
    "metadata": { ... },
    "llm_insights": null
  }
}
```

On failure:

```json
{
  "status": "failure",
  "agent_name": "AUTODS_DATA_UNDERSTANDING",
  "error_message": "..."
}
```

---

### Problem Type Logic

- `_infer_problem_type()` performs pure inference: numeric target with > 20 unique values → `regression`, otherwise → `classification`  
- `config.problem_type` overrides inference if explicitly provided  

---

### Data Quality Checks

The agent detects:

- missing values (count and ratio per column)  
- duplicate rows  
- constant columns (≤ 1 unique value)  
- all-missing columns  
- high-missing columns (≥ 30% missing)  
- identifier-like columns (by name pattern or unique ratio ≥ 95%)  
- high-cardinality columns (≥ 20 unique values or ≥ 30% unique ratio, categorical only)  
- numeric outliers (IQR method, skipped for < 5 values or zero IQR)  
- target leakage risk columns (by name similarity to the target column)  

`recommended_actions` in the quality report maps each issue category to the affected columns.

---

### Target Analysis

If a target column is provided:

#### Classification
- class distribution  
- imbalance ratio (max class count / min class count)  
- binary flag  
- recommended metrics: `f1`, `roc_auc`, `precision`, `recall`  

#### Regression
- summary statistics (count, mean, std, min, p25, median, p75, max)  
- recommended metrics: `rmse`, `mae`, `r2`  

---

### Downstream Handoff

`data_understanding_summary.json` includes a `downstream_handoff` section consumed by:

- **Cleaning Agent**: priority imputation columns, drop candidates  
- **Feature Engineering Agent**: numeric/categorical/high-cardinality/identifier columns  
- **Modelling Agent**: target column, problem type, class imbalance flag, recommended metrics  

---

### LLM Extension (Optional)

Set `use_llm_insights=True` in `AgentConfig` to enable LLM-generated insights. The interface is reserved — `_generate_llm_insights()` currently raises `NotImplementedError` until a real LLM client is wired in.

---

## Usage

### CLI

```bash
python main.py --data-path mydata.csv
```

All arguments:

```bash
python main.py \
  --data-path mydata.csv \
  --target-column label \
  --problem-type classification \
  --output-dir data_understanding_outputs \
  --dataset-name my_dataset \
  --random-state 42
```

Supported file formats: `.csv`, `.parquet` / `.pq`

### Python API

```python
import pandas as pd
from data_understanding_agent import DataUnderstandingAgent, AgentConfig

df = pd.read_csv("mydata.csv")

config = AgentConfig(
    output_dir="data_understanding_outputs",
    target_column="label",
    problem_type="classification",
    dataset_name="my_dataset",
)

agent = DataUnderstandingAgent(config)
result = agent.run(df)
```

---

## Project Structure

```text
.
├── data_understanding_agent.py   # Agent logic and AgentConfig
├── main.py                       # CLI entry point
├── mydata.csv                    # Sample data
├── README.md
├── requirements.txt
└── data_understanding_outputs/   # Auto-generated (not committed)
```

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Design Rationale

- **Decoupled input**: agent operates on DataFrames only; file I/O is the caller's responsibility  
- **Standardized outputs**: consistent JSON schema for all downstream agents  
- **Graceful failure**: all exceptions are caught and returned as a structured failure response  
- **Extensible**: LLM insights slot is reserved without breaking the existing interface  

---

## Author

Data Understanding Module  
Multi-Agent Data Science System
