# data-generation

This directory generates synthetic county-level datasets for a federated statistics project.

## Overview

The generator creates synthetic county datasets for multiple database schema designs used in the federated government statistics system. It produces CSV files that can be loaded into PostgreSQL worker nodes and supports configurable bad-data injection for validation and fault-tolerance testing.

The generator uses the **Faker** library to create more realistic synthetic data such as:
- first and last names
- street addresses
- city names
- ZIP codes
- UUIDs
- timestamps

---

## Goals

- Produce synthetic CSV data for multiple counties.
- Support three schema layouts:
  - **Schema A**: multi-table (`address`, `household`, `resident`) to stress joins
  - **Schema B**: single table (`person_record`) to stress scans and parsing
  - **Schema C**: single table (`citizen`) with JSONB to stress JSON extraction
- Inject controllable bad data to test validation and row-dropping logic.
- Generate reproducible outputs using a configurable seed.

---

## Requirements

Install Python dependencies before running the generator.

### Install Faker

```bash
pip install Faker
```

If you are using a virtual environment, activate it first and then install Faker.

## Usage

From the repository root:

```bash
python3 data-generation/generator.py --config data-generation/generator_config.json
```

Override common parameters:

```bash
python3 data-generation/generator.py \
  --config data-generation/generator_config.json \
  --out data-generation/data \
  --seed 123 \
  --bad-rate 0.10
```

---

### Command Line Options

- `--config`
Path to the generator configuration file.

- `--out`
Override the output root directory.

- `--seed`
Override the RNG seed used for reproducible generation.

- `--bad-rate`
Override the bad data injection rate. Must be between 0 and 1.

---

### Output Layout

The generator writes files into the following structure:

```
data-generation/data/
  schema_A/<county_name_or_label>/
    address.csv
    household.csv
    resident.csv
  schema_B/<county_name_or_label>/
    personal_record.csv
  schema_C/<county_name_or_label>/
    citizen.csv
```

Counties and record counts are configured in `generator_config.json.`

---

### Schema Details

#### Schema A

Schema A is a multi-table relational design intended to stress join-heavy workloads.

Generated files:

- `address.csv`
- `household.csv`
- `resident.csv`

#### Schema B

Schema B is a flat single-table design intended to stress scans, string parsing, and validation logic.

Generated files:

- `personal_record.csv`

#### Schema C

Schema C is a JSONB-style single-table design intended to stress nested extraction and JSON-based querying.

Generated files:

- `citizen.csv`

---

### Bad Data Injection

The generator can inject several forms of bad data into records. These are selected using weighted probabilities from `generator_config.json`.

Examples include:

- wrong data types
- invalid ages
- negative income
- nonsense ZIP codes
- inconsistent demographic codes
- stale timestamps
- move-out before move-in
- death before birth
- active-status inconsistencies
- missing verification data

This helps test:

- row validation logic
- dropped-row reporting
- schema-specific parsing behavior
- data quality metrics

---

### Reproducibility

The generator uses:

- Python’s `random.Random(seed)` for distributions, record selection, and corruption decisions
- Faker seeded with the same value for realistic but repeatable synthetic values

Using the same config and seed should produce reproducible outputs.

---

Example Workflow

Generate datasets with the default configuration:

```bash
python3 data-generation/generator.py --config data-generation/generator_config.json
```

Generate datasets with a custom seed and higher corruption rate:

```bash
python3 data-generation/generator.py \
  --config data-generation/generator_config.json \
  --seed 42 \
  --bad-rate 0.15
```

Write results to a different output directory:

```bash
python3 data-generation/generator.py \
  --config data-generation/generator_config.json \
  --out ./tmp/generated_data
```
