# Security Policy

## Local-processing model

OpenBPO Drift v1 is designed to run locally with Streamlit. Uploaded files are parsed, normalized, analyzed, and exported on the machine running the app.

The application does not intentionally send uploaded data, normalized KPI rows, drift alerts, schema mappings, or generated reports to external services.

## Network posture

The app has no built-in telemetry, analytics beacon, LLM call, hosted API dependency, or cloud storage integration. External network access is not required for the sample workflow after dependencies are installed.

If you deploy OpenBPO Drift behind a network-accessible endpoint, you are responsible for the deployment boundary, authentication, authorization, TLS, logging, access controls, and AGPL source-availability obligations.

## Supported versions

Security fixes are accepted for the current `main` branch.

## Reporting a vulnerability

Please report vulnerabilities privately to the maintainer before public disclosure. Include:

- Affected commit or release
- Reproduction steps
- Expected and actual behavior
- Any affected file types, inputs, or generated outputs

Do not include real customer, employee, or production BPO data in reports.

## V1 limitations

OpenBPO Drift v1 is a local-first reference app. It does not provide enterprise controls such as SSO, role-based access control, audit logs, centralized secrets management, encrypted server-side persistence, or tenant isolation.
