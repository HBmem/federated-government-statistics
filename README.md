# federated-government-statistics

A federated distributed statistics reporting system built with **C++**, **MPI**, **PostgreSQL**, and **Docker**.  
Each local node retains its own database and performs local aggregation. The coordinator gathers only approved aggregate-level metrics and produces a final JSON report.

---

## Project Overview

This project simulates a federated reporting environment in which local agencies keep ownership of their raw data while still contributing aggregate statistics for higher-level reporting.

Instead of centralizing raw resident records, each worker node:
- stores its own county dataset in PostgreSQL,
- detects its local schema structure,
- runs schema-appropriate aggregation queries locally,
- returns only summary results and quality metrics.

The coordinator node:
- reads the node configuration,
- requests schema discovery from each worker,
- selects query templates based on the detected schema,
- gathers worker responses,
- handles worker failures and timeouts,
- merges aggregate results,
- writes a final JSON report.

---

## Current Features

- Distributed coordinator/worker execution using MPI
- PostgreSQL-backed worker nodes running in Docker containers
- Synthetic data generation for multiple counties
- Three schema variants:
  - **Schema A**: multi-table relational schema for join-heavy queries
  - **Schema B**: flat single-table schema for scan-heavy queries
  - **Schema C**: JSONB-based schema for nested document queries
- Controlled bad-data injection for testing validation behavior
- Query-template-based execution for modularity
- Worker fault and timeout reporting
- Final coordinator output as JSON

---

## Directory Structure

```text
Config/
  nodes.txt
  coordinator_config.json
  buckets.json

DataGeneration/
  generator.py
  generator_config.json
  Data/

Init/
  metro/
  medium1/
  medium2/
  rural1/
  rural2/

queries/
  registry.json
  templates/
    schema_a/
    schema_b/
    schema_c/

Src/
  main.cpp
  common/
  coordinator/
  worker/
  db/

Output/
  runs/
  logs/

Docker-compose.yml
CMakeLists.txt
README.md
```

---

## Data Generation

Synthetic data is generated for multiple counties and written into CSV files for all three schema designs.

### Supported Schemas

#### Schema A

A multi-table design intended to test join-heavy aggregation logic.

Tables:
- `household`
- `address`
- `resident`

#### Schema B

A single-table design intended to test scan-heavy queries and string parsing.

Table:
- `person_record`

#### Schema C

A single-table JSONB design intended to test JSON extraction and nested field access.

Table:
- `citizen`

### Bad Data Injection

The generator can inject controlled bad data such as:
- wrong data types
- invalid ages
- negative income
- nonsense ZIP codes
- inconsistent demographic codes
- stale timestamps
- move-out before move-in
- death before birth
- active-status inconsistencies
- missing verification fields

This allows the aggregation logic to test row validation, dropping behavior, and quality reporting.

### Running the Generator

From the project root:

```Bash
python3 DataGeneration/generator.py --config DataGeneration/generator_config.json
```

Override output path, seed, or bad-data rate:

```Bash
python3 DataGeneration/generator.py \
  --config DataGeneration/generator_config.json \
  --out DataGeneration/Data \
  --seed 123 \
  --bad-rate 0.10
```

---

## Node Configuration

Worker database connections are defined in `Config/nodes.txt`.

```INI
# rank 0 (coordinator)
COORDINATOR

# rank 1 (metro_db)
host=127.0.0.1 port=5541 dbname=county user=federated password=federated county_fips=53033
```

### Important Notes

- Rank `0` must always be the coordinator.
- Each worker node must include:
    - `host`
    - port
    - dbname
    - user
    - password
    - county_fips

The county_fips value from nodes.txt is used by the worker when labeling results, including for Schema B where the database stores county name instead of county FIPS.

---

## Coordinator Configuration

Global coordinator behavior is stored in:

```
Config/coordinator_config.json
```

This file controls:
- default time window
- fault tolerance and timeouts
- DB statement timeout settings
- enabled metrics
- output directory
- logging settings
- template registry path

### CLI Override Behavior

The coordinator uses the time window from the config file by default, but it can be overridden from the command line:

```Bash
--start YYYY-MM-DD
--end YYYY-MM-DD
```

---

## Data Aggregation

The coordinator requests aggregate-level statistics from worker nodes for a specified time window.

### Requested Inputs

- time window
    - start date
    - end date

### Grouping

- county FIPS
- requested time window

### Metrics

#### Population

- population count

#### Demographics

- sex counts
- race counts
- ethnicity counts

#### Income

- income sum
- income count
- income min
- income max
- income buckets

#### Age

- age sum
- age count
- age min
- age max
- age buckets

#### Employment

- count with jobs
- count without jobs

#### Quality

- rows scanned
- rows used
- rows dropped
- dropped-row reason counts

---

## Residency and Time Window Logic

The time window is used to determine whether a resident should be counted for the requested reporting period.

A resident may be excluded if:
- the resident moved into the county after the end of the requested window,
- the resident moved out before the start of the requested window,
- the resident died on or before the end of the requested window,
- required fields are missing,
- data values are invalid or inconsistent.

Each worker applies schema-specific validation and reports:
- how many rows were scanned,
- how many rows were used,
- how many rows were dropped,
- why rows were dropped.

---

## Query Execution Model

The system uses a coordinator/worker model.

### Phase 1: Discovery

Each worker inspects its local PostgreSQL schema and returns:
- detected schema type (`A`, `B`, `C`, or `Unknown`)
- table names
- column names
- column data types

### Phase 2: Planning

The coordinator selects the correct SQL templates using:
- schema type
- enabled metric list
- configured time window

### Phase 3: Execution

Each worker:
- executes its assigned templates locally,
- computes aggregate-level results,
- returns metric payloads and quality counters.

### Phase 4: Aggregation

The coordinator:
- merges worker outputs,
- records failures and timeouts,
- writes a final JSON report.

---

## Docker Setup

Each worker node runs in its own PostgreSQL container.

### Start the Databases

```Bash
docker compose up -d
```

### Check Container Status

```Bash
docker compose ps -a
```

### Rebuild Database State from Scratch

```Bash
docker compose down -v
docker compose up -d
```

### Notes

- The `Init/` directory for each node is mounted into `/docker-entrypoint-initdb.d`.
- PostgreSQL automatically runs the SQL initialization scripts in that directory on first container startup.
- If you change schema/load scripts and want them to re-run, use `docker compose down -v` to remove existing volumes first.

---

## Building the Project

Create and enter a build directory:

```Bash
mkdir -p build
cd build
```

Configure and build:

```Bash
cmake ..
make -j
```

If you need to rebuild from a clean state:

```Bash
make clean
cmake ..
make -j
```

---

## Running the MPI Program

From the build/ directory:

```Bash
mpirun -np 6 ./federated
```

With Explicit Config Paths

```Bash
mpirun -np 6 ./federated \
  --config ../Config/coordinator_config.json \
  --nodes ../Config/nodes.txt
```

With Time Window Override

```Bash
mpirun -np 6 ./federated \
  --config ../Config/coordinator_config.json \
  --nodes ../Config/nodes.txt \
  --start 2026-01-01 \
  --end 2026-12-31
```

Rank Layout
- rank `0`: coordinator
- rank `1`: metro worker
- rank `2`: medium1 worker
- rank `3`: medium2 worker
- rank `4`: rural1 worker
- rank `5`: rural2 worker

---

## Output

The coordinator writes final JSON reports to:

```
Output/runs/
```

The report includes:
- run metadata
- requested time window
- enabled metrics
- per-node schema discovery results
- per-node aggregation results
- failures and timeouts
- merged global aggregates

---

## rst Test Scenario

The initial test uses five worker databases and one coordinator process.

| Database | County    | Size   |
| -------- | --------- | ------ |
| metro    | King      | 25,000 |
| medium1  | Pierce    | 10,000 |
| medium2  | Snohomish | 10,000 |
| rural1   | Ferry     | 2,000  |
| rural2   | San Juan  | 3,000  |

Total rows: **50,000**

This first test is intended to evaluate:
- correctness of distributed aggregation,
- schema-specific execution behavior,
- row validation and dropped-row reporting,
- failure handling,
- performance differences across schema designs and node resource levels.

---

## Current Execution Sequence

Generate synthetic CSV data

Copy generated CSVs into the appropriate Init/ node folders

Start PostgreSQL containers with Docker Compose

Build the C++ MPI application

Run the program with mpirun

Inspect the generated coordinator JSON report

---

## Dependencies

You will need:
- C++17 compiler
- CMake 3.16+
- MPI implementation
- PostgreSQL client library (`libpq`)
- Docker and Docker Compose
- Python 3 for data generation
- `nlohmann/json` available either:
    - as a system package, or
    - as a vendored header in `ThirdParty/nlohmann/json.hpp`

## Notes for Development

- The current design uses predefined SQL templates stored under queries/.
- Workers perform local aggregation and never send raw row-level data to the coordinator.
- Schema B relies on worker configuration from nodes.txt for canonical county FIPS labeling.
- Future work may include AI-assisted query generation and query-template caching.

---