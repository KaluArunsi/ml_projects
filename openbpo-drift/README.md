# OpenBPO Drift

OpenBPO Drift is a local-first Streamlit app for detecting KPI drift in BPO and contact center operations. Upload CSV or Excel KPI data, map your columns, run explainable drift detection, view trend charts, and export alerts without sending data to external services.

## Screenshot

_Screenshot placeholder: add a local app screenshot after the first polished UI pass._

## What v1 includes

- CSV and Excel upload
- Interactive schema mapping
- Canonical KPI normalization
- Data quality checks
- Rolling baseline drift detection
- KPI trend charts
- CSV, YAML, and Markdown export
- Sample BPO KPI dataset

## What v1 does not include

- LLMs
- Cloud hosting
- Authentication
- Multi-user workspaces
- Scheduled jobs
- Enterprise connectors
- Slack or email alerts

## Repository structure

```text
openbpo-drift/
├── app.py
├── LICENSE
├── README.md
├── requirements.txt
├── .streamlit/config.toml
├── configs/
│   ├── default_kpi_rules.yaml
│   └── sample_mapping.yaml
├── data/
│   └── sample_bpo_kpis.csv
├── reports/
│   └── .gitkeep
├── src/
│   ├── charts.py
│   ├── config.py
│   ├── drift.py
│   ├── loaders.py
│   ├── mapper.py
│   ├── report.py
│   ├── sample_data.py
│   ├── schema.py
│   └── validation.py
└── tests/
    ├── test_drift.py
    ├── test_mapper.py
    └── test_validation.py
```

## Quickstart

```bash
git clone https://github.com/<owner>/openbpo-drift.git
cd openbpo-drift
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Try the sample workflow

The repository includes `data/sample_bpo_kpis.csv` and a matching YAML mapping in [configs/sample_mapping.yaml](/Users/kaluarunsi/Desktop/ml_projects/openbpo-drift/configs/sample_mapping.yaml:1).

Inside the app:

1. Click `Load sample data` in the sidebar if it is not already active.
2. Review the raw file in `Data Preview`.
3. Confirm the guessed field mapping and KPI mapping in `Schema Mapper`.
4. Review normalization and data quality results.
5. Inspect drift alerts and KPI trends.
6. Export normalized data, alerts, mapping YAML, or the Markdown report.

## Input format

OpenBPO Drift accepts `.csv`, `.xlsx`, and `.xls` files. V1 uses the first Excel sheet by default, with optional sheet selection in the UI.

Minimum required fields:

- `date`
- `entity_id` or equivalent mapped source column
- at least one KPI column

Optional metadata fields:

- `team`
- `site`
- `account`
- `shift`

Sample wide input:

```csv
date,agent_id,team,site,account,shift,aht,qa,csat
2026-06-01,A001,Team Manila A,Manila,Telco,Night,420,91,4.6
```

The app normalizes this into a canonical long format with `date`, `entity_id`, `kpi_name`, `kpi_value`, and the mapped metadata fields.

## Schema mapping

The Streamlit UI keeps schema mapping in the app layer, but the reusable normalization logic lives in [src/mapper.py](/Users/kaluarunsi/Desktop/ml_projects/openbpo-drift/src/mapper.py:1). `app.py` only collects mapping choices and calls into `src/`.

The mapping export format looks like this:

```yaml
field_mapping:
  date: date
  entity_id: agent_id
  team: team
  site: site
  account: account
  shift: shift

kpis:
  - source_column: aht
    kpi_name: aht
    unit: seconds
    direction_bad: up
    drift_threshold_pct: 15
    include: true
```

## Exports

The export tab provides:

- `normalized_kpis.csv`
- `drift_alerts.csv`
- `mapping.yaml`
- `openbpo_drift_report.md`

## Tests

```bash
pytest
```

## License

OpenBPO Drift is licensed under the GNU Affero General Public License v3.0 or later.

This means you can use, study, modify, and share the software under the terms of the AGPL. If you modify and deploy the software as a network-accessible service, the AGPL requires that you make the corresponding source code available to users of that service.

Commercial licensing is available for teams that want to embed, modify, or deploy OpenBPO Drift without AGPL obligations. Contact the maintainer for commercial licensing, private integrations, or production deployment support.

## Contributing

Contributions are welcome, but keep the v1 boundary intact. The non-negotiable rule is architectural: `app.py` stays focused on Streamlit UI, and reusable logic belongs in `src/`.
