# AUTODS Data Understanding Agent

A modular data understanding component for a multi-agent automated data science pipeline.

---

## Overview

This repository implements the Data Understanding Agent, responsible for:

- profiling tabular datasets  
- identifying data quality issues  
- analysing the target column  
- producing structured outputs for downstream agents  

The agent is designed to be integrated into a multi-agent system.

---

## Key Design

### Input Interface

The agent accepts an in-memory pandas DataFrame:

```python
agent.run(df)
```

This removes dependency on file paths and allows seamless pipeline integration.

---

### Output Artifacts

The agent generates the following JSON files:

- `data_profile.json`  
- `data_quality_report.json`  
- `target_analysis.json`  
- `data_understanding_summary.json`  
- `data_understanding_metadata.json`  

These are consumed by downstream agents:

- Cleaning Agent  
- Feature Engineering Agent  
- Modelling Agent  

---

### Standardized Response

```json
{
  "status": "success" | "failure",
  "agent_name": "AUTODS_DATA_UNDERSTANDING",
  "result": { ... }
}
```

This ensures consistent communication across the pipeline.

---

### Problem Type Logic

- `_infer_problem_type()` performs pure inference from data  
- `config.problem_type` overrides inference if provided  

---

### Data Quality Checks

The agent detects:

- missing values  
- duplicate rows  
- constant columns  
- high-missing columns  
- identifier-like columns  
- high-cardinality columns  
- numeric outliers (IQR method)  

---

### Target Analysis

If a target column is provided:

#### Classification
- class distribution  
- imbalance ratio  
- recommended metrics (F1, ROC-AUC)

#### Regression
- summary statistics  
- recommended metrics (RMSE, MAE, R²)

---

### LLM Extension (Optional)

```python
_generate_llm_insights(...)
```

Reserved for future extension.

---

## Usage

```bash
python main.py --data-path mydata.csv
```

Optional arguments:

```bash
--target-column target
--problem-type classification
--output-dir data_understanding_outputs
```

---

## Example

```python
import pandas as pd
from data_understanding_agent import DataUnderstandingAgent, AgentConfig

df = pd.read_csv("mydata.csv")

config = AgentConfig(output_dir="data_understanding_outputs")
agent = DataUnderstandingAgent(config)

result = agent.run(df)
```

---

## Project Structure

```text
.
├── data_understanding_agent.py
├── main.py
├── mydata.csv
├── README.md
├── requirements.txt
└── data_understanding_outputs/
```

---

## Notes

- `main.py` handles file loading  
- agent operates on DataFrame only  
- outputs are auto-generated  
- `data_understanding_outputs/` is ignored by Git  

---

## Design Rationale

This design enables:

- decoupled data input  
- standardized outputs  
- pipeline integration  
- easier testing and extension  

---

## Author

Data Understanding Module  
Multi-Agent Data Science System