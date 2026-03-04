# data-generation

This directory generates synthetic county-level datasets for a federated statistics project.

## Goals
- Produce synthetic CSV data for multiple counties.
- Support three schema layouts:
  - Schema A: multi-table (address, household, resident) to stress JOINs
  - Schema B: single table (person_record) to stress scans and parsing
  - Schema C: single table (citizen) with JSONB to stress JSON extraction
- Inject controllable "bad data" to test validation and row-dropping logic.

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

## Output Layout

- data-generation/data/
    - Schema_A/<county_name_or_label>/
        - Address.csv
        - Household.csv
        - Resident.csv
    - Schema_B/<county_name_or_label>/
        - Personal_record.csv
    - Schema_C/<county_name_or_label>/
        - Citizen.csv

Counties and record counts are configured in generator_config.json.