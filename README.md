## Output Artifacts

The agent produces the following JSON files:

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

## Standardized Response

```json
{
  "status": "success" | "failure",
  "agent_name": "AUTODS_DATA_UNDERSTANDING",
  "result": { ... }
}