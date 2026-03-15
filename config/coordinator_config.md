# coordinator_config.json Documentation

This document describes the configuration parameters used by `coordinator_config.json`.
The coordinator reads this file at runtime to control orchestration behavior,
timeouts, metric selection, and query planning behavior.

The configuration file must remain **valid JSON without comments**. All explanations
are contained in this document.

---

# Configuration Overview

The configuration is divided into the following sections:

- run
- fault_tolerance
- db
- metrics
- residency_rules
- query_templates
- logging

Each section is explained below.

---

# run

Controls the parameters for a specific execution of the coordinator.

### time_window

Defines the timeframe used when determining if a resident should be included in
the aggregation results.

Fields:

- **start_date**
  - Start of the time window
  - Format: `YYYY-MM-DD`

- **end_date**
  - End of the time window
  - Format: `YYYY-MM-DD`

Workers use this time window to determine if a resident is still present in the county
based on fields such as:

- moved_in_date
- move_out_date
- death_date
- active_status

---

### category

Defines the type of dataset being requested.

This allows the architecture to expand in the future to support additional categories
such as:

- demographic
- health
- housing
- education

For the current project iteration the category should be:

- demographic

---

### output_dir

Directory where the coordinator will write the final report JSON.

The output report includes:

- aggregated statistics
- node failures
- query plans used
- dropped rows
- quality metrics

---

### include_node_payloads

Boolean flag that determines whether the final report should include the
raw worker responses.

Setting this to `true` makes debugging easier but increases the size of the
final JSON report.

---

# fault_tolerance

Controls how the coordinator handles failures or slow worker nodes.

---

### worker_response_timeout_sec

Maximum amount of time the coordinator will wait for a worker response before
declaring that worker as failed.

If exceeded, the node will be marked as:

- timeout

---

### worker_query_timeout_sec

Maximum amount of time a worker is allowed to run its database query.

If exceeded, the worker should abort execution and return a failure response.

---

### max_retries_per_phase

Number of retry attempts allowed for a failed worker phase.

Worker phases include:

- schema discovery
- query execution

Example:

- 0 = no retries
- 1 = retry once

---

# db

Database execution settings applied by worker nodes.

These are best-effort settings and may not always succeed depending on database permissions.

---

### statement_timeout_ms

Maximum amount of time a SQL statement is allowed to run.

Applied using:

- SET statement_timeout

---

### lock_timeout_ms

Maximum time a query waits for database locks.

Applied using:

- SET lock_timeout

---

# metrics

Controls which statistical modules are requested by the coordinator.

Each metric module produces partial results that can be merged by the coordinator.

---

### enabled

Each boolean flag determines whether a metric is included in the execution plan.

Available modules:

- **population**
- **demographics**
- **income**
- **age**
- **unemployment**
- **quality**

---

### bucket_config_path

Path to the histogram bucket configuration file.

Buckets are used to calculate:

- age distributions
- income distributions
- approximate median values

The file referenced is:
- Config/buckets.json

---

# residency_rules

Defines the rules used to determine whether a resident should be included
in the results during the given time window.

Workers translate these rules into schema-specific SQL filters.

---

### require_county_match

If true, only records with matching `county_fips` should be included.

---

### strict_missing_fields

Determines how rows with missing data should be handled.

If true:

- rows with missing required fields are dropped
- dropped rows contribute to bad-data counts

If false:

- missing values may be interpreted as unknown

---

### date_logic

Defines how residency date fields are interpreted.

---

#### require_moved_in_on_or_before_end

Residents must have moved into the county on or before the end of the time window.

---

#### exclude_if_moved_out_before_start

Residents who moved out before the start of the time window are excluded.

---

#### exclude_if_dead_on_or_before_end

Residents who died on or before the end of the time window are excluded.

---

# query_templates

Defines how the coordinator selects SQL templates for worker nodes.

---

### registry_path

Path to the template registry file.

The registry maps:
- metric + schema type → SQL template file

Example location:
- queries/registry.json

---

### allow_probe_fallback

If enabled, the coordinator may attempt multiple query templates when the
schema type cannot be determined with certainty.

The worker may run an `EXPLAIN` test query to verify compatibility.

---

# logging

Controls logging behavior.

---

### log_dir

Directory where coordinator logs will be written.

---

### level

Logging verbosity level.

Available levels:
- trace
- debug
- info
- warn
- error

Recommended default:

- info

---

# Summary

`coordinator_config.json` controls:

- runtime execution parameters
- worker fault tolerance
- database query limits
- enabled metrics
- residency filtering rules
- query template selection
- logging behavior

All documentation for the configuration fields is stored in this file to keep
the JSON configuration strictly compliant with the JSON specification.