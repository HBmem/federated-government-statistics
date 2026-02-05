"""
Synthetic county=level demographic generator

 - 3 schema variants (A, B, C)
 - Bad-data injection
 - Outputs: pandas DataFrame and CSV file

Requires:
 - pip install faker pandas numpy
"""
import random
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker("en_US")

# ----------------------------
# County config
# ----------------------------

@dataclass(frozen=True)
class CountySpec:
    name: str
    fips: str
    tier: str # "metro", "mid", "rural"
    median_income: int
    income_spread: int
    hh_size_lambda: float


COUNTIES: List[CountySpec] = [
    CountySpec("King", "53033", "metro", median_income=105000, income_spread=45000, hh_size_lambda=2.2),
    CountySpec("Pierce", "53053", "mid",  median_income=85000,  income_spread=35000, hh_size_lambda=2.4),
    CountySpec("Snohomish", "53061", "mid", median_income=95000, income_spread=38000, hh_size_lambda=2.3),
    CountySpec("Ferry", "53019", "rural", median_income=60000,  income_spread=25000, hh_size_lambda=2.2),
    CountySpec("Garfield", "53023", "rural", median_income=65000, income_spread=22000, hh_size_lambda=2.1),
    CountySpec("San Juan", "53055", "rural", median_income=90000, income_spread=42000, hh_size_lambda=2.0),
]


# ----------------------------
# Demographic value pools
# ----------------------------

SEX_VALUES = ["M", "F", "X"]
SEX_CODES = {"M": "M", "F": "F", "X": "U"}

RACE_VALUES = ["White", "Black", "Asian", "Native", "Pacific", "Other", "TwoOrMore"]
RACE_CODES = {
    "White": "W",
    "Black": "B",
    "Asian": "A",
    "Native": "N",
    "Pacific": "P",
    "Other": "O",
    "TwoOrMore": "T",
}

ETHNICITY_VALUES = ["Hispanic", "Non-Hispanic"]
ETH_FLAG = {"Hispanic": "Y", "Non-Hispanic": "N"}

HOUSING_TYPES = ["SFH", "Apartment", "Townhome", "Mobile", "Other"]

INCOME_BRACKETS = [
    "<25k", "25-50k", "50-75k", "75-100k", "100-150k", "150-200k", "200k+"
]
BRACKET_BOUNDS = {
    "<25k": (0, 25000),
    "25-50k": (25000, 50000),
    "50-75k": (50000, 75000),
    "75-100k": (75000, 100000),
    "100-150k": (100000, 150000),
    "150-200k": (150000, 200000),
    "200k+": (200000, 500000),
}

# ----------------------------
# Helpers
# ----------------------------

def _random_dob(min_age: int = 0, max_age: int = 95) -> date:
    """Random DOB using age bounds."""
    today = date.today()
    age = random.randint(min_age, max_age)
    # random day within age year
    start = today - timedelta(days=365 * (age + 1))
    end = today - timedelta(days=365 * age)
    return fake.date_between(start_date=start, end_date=end)

def _income_from_county(spec: CountySpec) -> int:
    """Lognormal-ish income with county-specific location/scale; clipped."""
    # Use normal around median with spread, then clip
    val = int(np.random.normal(loc=spec.median_income, scale=spec.income_spread))
    return int(np.clip(val, 0, 500000))

def _income_to_bracket(income: int) -> str:
    for b, (lo, hi) in BRACKET_BOUNDS.items():
        if lo <= income < hi:
            return b
    return "200k+"

def _household_size(spec: CountySpec) -> int:
    # Poisson skewed towards 1-3, clipped to [1..8]
    val = 1 + np.random.poisson(lam=max(0.8, spec.hh_size_lambda - 1.0))
    return int(np.clip(val, 1, 8))

def _zip_for_county(county_name: str) -> str:
    """Not exact; just WA-ish 98xxx / 99xxx vibe to avoid needing real zip mapping."""
    if county_name in ("King", "Snohomish", "Pierce"):
        base = random.choice([98, 98, 98, 99])  # more 98xxx
    else:
        base = random.choice([98, 99, 99])  # more 99xxx
    return f"{base}{random.randint(0, 999):03d}"

def _dob_to_mixed_string(d: date) -> str:
    """Schema B: mixed date formats (on purpose, even before 'bad data')."""
    fmt = random.choice(["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d"])
    return d.strftime(fmt)

def _now_ts() -> datetime:
    return datetime.utcnow().replace(microsecond=0)

def _normalize_bad_rate(bad_rate: float) -> float:
    # If user passes 3 meaning "3%", convert to 0.03
    if bad_rate > 1.0:
        if bad_rate <= 100.0:
            bad_rate = bad_rate / 100.0
        else:
            raise ValueError("bad_rate must be in [0,1] or [0,100] as percent.")
    return max(0.0, min(1.0, float(bad_rate)))

# ----------------------------
# Bad data injection
# ----------------------------

BAD_ERROR_TYPES = [
    "missing_required",
    "invalid_age",
    "negative_income",
    "nonsense_zip",
    "wrong_type",
    "inconsistent_codes",
    "stale_timestamp",
]

def inject_bad_data_record(record: dict,
                    schema: str,
                    bad_rate: float,
                    error_weights: Optional[Dict[str, float]] = None,
                    ) -> Tuple[dict, Optional[str]]:
    """
    With probability bad_rate, mutate record to contain a data-quality issue.
    Returns (record, error_type_or_None).
    """
    bad_rate = _normalize_bad_rate(bad_rate)

    if bad_rate <= 0:
        return record, None
    if random.random() > bad_rate:
        return record, None
    
    weights = error_weights or {k: 1.0 for k in BAD_ERROR_TYPES}
    # normalize weights
    keys = list(weights.keys())
    w = np.array([max(0.0, float(weights[k])) for k in keys], dtype=float)
    if w.sum() <= 0:
        keys = BAD_ERROR_TYPES
        w = np.ones(len(keys))
    w = w / w.sum()

    error_type = np.random.choice(keys, p=w)

    # Apply an error depending on schema
    if error_type == "missing_required":
        # knock out a common required field
        candidates = {
            "A": ["resident_id", "dob", "household_id"],
            "B": ["person_uuid", "dob_str", "county"],
            "C": ["doc_id", "profile_json", "county_fips"],
        }[schema]
        k = random.choice(candidates)
        record[k] = None
    
    elif error_type == "invalid_age":
        # set DOB to future date or absurd old
        if schema == "A":
            record["dob"] = date.today() + timedelta(days=random.randint(1, 3650))
        elif schema == "B":
            future = date.today() + timedelta(days=random.randint(1, 3650))
            record["dob_str"] = future.strftime("%Y-%m-%d")
        else:  # C
            pj = record["profile_json"]
            pj["demographics"]["dob"] = (date.today() + timedelta(days=999)).isoformat()
            record["profile_json"] = pj

    elif error_type == "negative_income":
        if schema == "A":
            record["income_usd"] = -abs(int(record.get("income_usd") or 1000))
        elif schema == "B":
            # bracket that doesn't exist
            record["income_bracket"] = "-50k"
        else:
            pj = record["profile_json"]
            pj["household"]["income_usd"] = -5000
            record["profile_json"] = pj

    elif error_type == "nonsense_zip":
        if schema == "A":
            record["zip5"] = "ABCDE"
        elif schema == "B":
            # embed nonsense in address
            record["full_address"] = (record.get("full_address") or "") + " ZIP=XXXXX"
        else:
            pj = record["profile_json"]
            pj["location"]["zip5"] = "12"
            record["profile_json"] = pj

    elif error_type == "wrong_type":
        # store numbers as strings, dates as ints, etc.
        if schema == "A":
            record["hh_size"] = str(record.get("hh_size", "2"))
        elif schema == "B":
            record["household_size"] = "two"
        else:
            pj = record["profile_json"]
            pj["household"]["hh_size"] = "3"
            record["profile_json"] = pj

    elif error_type == "inconsistent_codes":
        if schema == "B":
            # invalid codes
            record["gender_code"] = "Z"
            record["race_code"] = "?"
        elif schema == "A":
            record["sex"] = "Unknownish"
        else:
            pj = record["profile_json"]
            pj["demographics"]["race"] = 999
            record["profile_json"] = pj

    elif error_type == "stale_timestamp":
        # updated timestamp far in the past
        stale = datetime(1999, 1, 1, 0, 0, 0)
        if schema == "A":
            record["created_at"] = stale
        elif schema == "B":
            record["updated_ts"] = stale
        else:
            record["created_at"] = stale

    return record, error_type

# ----------------------------
# Schema emitters
# ----------------------------

def generate_schema_A(county: CountySpec, n_households: int, bad_rate: float, error_weights=None):
    """
    Returns DataFrames: residents, households, addresses, plus a data_quality log.
    """
    households = []
    addresses = []
    residents = []
    dq = []

    for _ in range(n_households):
        household_id = uuid.uuid4()
        hh_size = _household_size(county)
        income = _income_from_county(county)
        housing_type = random.choice(HOUSING_TYPES)

        address_id = uuid.uuid4()
        street = fake.street_address()
        city = fake.city()
        state = "WA"
        zip5 = _zip_for_county(county.name)

        hrow = {
            "household_id": household_id,
            "county_fips": county.fips,
            "hh_size": hh_size,
            "income_usd": income,
            "housing_type": housing_badge(housing_type),
            "created_at": _now_ts(),
        }
        arow = {
            "address_id": address_id,
            "street": street,
            "city": city,
            "state": state,
            "zip5": zip5,
        }

        # allow bad data mutations on household/address too
        hrow, e1 = inject_bad_data_record(hrow, "A", bad_rate, error_weights)
        if e1:
            dq.append({"county": county.name, "schema": "A", "table": "household", "error_type": e1})
        arow, e2 = inject_bad_data_record(arow, "A", bad_rate * 0.5, error_weights)
        if e2:
            dq.append({"county": county.name, "schema": "A", "table": "address", "error_type": e2})

        households.append(hrow)
        addresses.append(arow)

        # residents for household
        for _ in range(int(hh_size)):
            dob = _random_dob(0, 95)
            sex = random.choice(SEX_VALUES)
            race = random.choice(RACE_VALUES)
            ethnicity = random.choice(ETHNICITY_VALUES)

            rrow = {
                "resident_id": uuid.uuid4(),
                "household_id": household_id,
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "dob": dob,
                "sex": sex,
                "race": race,
                "ethnicity": ethnicity,
                "address_id": address_id,
                "moved_in_date": fake.date_between(start_date="-10y", end_date="today"),
            }

            rrow, e3 = inject_bad_data_record(rrow, "A", bad_rate, error_weights)
            if e3:
                dq.append({"county": county.name, "schema": "A", "table": "resident", "error_type": e3})

            residents.append(rrow)

    return (
        pd.DataFrame(residents),
        pd.DataFrame(households),
        pd.DataFrame(addresses),
        pd.DataFrame(dq),
    )

def generate_schema_B(county: CountySpec, n_records: int, bad_rate: float, error_weights=None):
    """
    Returns DataFrame: person_record, plus data_quality log.
    """
    rows = []
    dq = []

    for _ in range(n_records):
        dob = _random_dob(0, 95)
        sex = random.choice(SEX_VALUES)
        race = random.choice(RACE_VALUES)
        ethnicity = random.choice(ETHNICITY_VALUES)
        income = _income_from_county(county)

        full_addr = f"{fake.street_address()}, {fake.city()}, WA {_zip_for_county(county.name)}"

        row = {
            "person_uuid": str(uuid.uuid4()),
            "county": county.name,
            "dob_str": _dob_to_mixed_string(dob),
            "gender_code": SEX_CODES[sex],
            "race_code": RACE_CODES[race],
            "eth_flag": ETH_FLAG[ethnicity],
            "income_bracket": _income_to_bracket(income),
            "household_size": _household_size(county),
            "full_address": full_addr,
            "updated_ts": _now_ts(),
        }

        row, e = inject_bad_data_record(row, "B", bad_rate, error_weights)
        if e:
            dq.append({"county": county.name, "schema": "B", "table": "person_record", "error_type": e})

        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(dq)


def generate_schema_C(county: CountySpec, n_docs: int, bad_rate: float, error_weights=None):
    """
    Returns DataFrame: citizen docs, plus data_quality log.
    """
    rows = []
    dq = []

    for _ in range(n_docs):
        dob = _random_dob(0, 95)
        sex = random.choice(SEX_VALUES)
        race = random.choice(RACE_VALUES)
        ethnicity = random.choice(ETHNICITY_VALUES)
        income = _income_from_county(county)

        profile = {
            "name": {"first": fake.first_name(), "last": fake.last_name()},
            "demographics": {
                "dob": dob.isoformat(),
                "sex": sex,
                "race": race,
                "ethnicity": ethnicity,
            },
            "household": {
                "hh_id": str(uuid.uuid4()),
                "hh_size": _household_size(county),
                "income_usd": income,
            },
            "location": {
                "street": fake.street_address(),
                "city": fake.city(),
                "state": "WA",
                "zip5": _zip_for_county(county.name),
            },
        }

        row = {
            "doc_id": str(uuid.uuid4()),
            "county_fips": county.fips,
            "profile_json": profile,
            "created_at": _now_ts(),
        }

        row, e = inject_bad_data_record(row, "C", bad_rate, error_weights)
        if e:
            dq.append({"county": county.name, "schema": "C", "table": "citizen", "error_type": e})

        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(dq)

# Small helper to allow a “type drift” within allowed values too
def housing_badge(housing_type: str) -> str:
    # a tiny realism touch: some systems store “Single Family” vs “SFH”
    if random.random() < 0.15 and housing_type == "SFH":
        return "Single Family"
    return housing_type

# ----------------------------
# Orchestrator
# ----------------------------

def generate_county_data(
    schema: str,
    county: CountySpec,
    scale: int,
    bad_rate: float = 0.02,
    error_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    schema: "A" | "B" | "C"
    scale:
      - schema A: number of households
      - schema B: number of person_records
      - schema C: number of citizen docs
    bad_rate: fraction of rows to corrupt (approx), 0..1

    returns dict of DataFrames
    """
    schema = schema.upper().strip()
    if schema == "A":
        residents, households, addresses, dq = generate_schema_A(county, scale, bad_rate, error_weights)
        return {
            "resident": residents,
            "household": households,
            "address": addresses,
            "data_quality": dq,
        }
    if schema == "B":
        person, dq = generate_schema_B(county, scale, bad_rate, error_weights)
        return {"person_record": person, "data_quality": dq}
    if schema == "C":
        citizen, dq = generate_schema_C(county, scale, bad_rate, error_weights)
        return {"citizen": citizen, "data_quality": dq}
    raise ValueError("schema must be one of: A, B, C")

def generate_all_nodes(schema: str,
                       county_scales: Dict[str, int],
                       bad_rate: float = 0.02,
                       error_weights: Optional[Dict[str, float]] = None,
                       seed: int = 42,
                       ) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    county_scales: dict like {"King": 80000, "Pierce": 30000, ...}
    returns {county_name: {table: df, ...}, ...}
    """
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)

    by_name = {c.name: c for c in COUNTIES}
    out = {}
    for county_name, scale in county_scales.items():
        county = by_name[county_name]
        out[county_name] = generate_county_data(schema, county, scale, bad_rate, error_weights)
    return out


def export_to_csv(node_data: Dict[str, Dict[str, pd.DataFrame]], out_dir: str, schema: str):
    """
    Writes each county/table to CSV.
    For schema C, profile_json will be stringified.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    for county, tables in node_data.items():
        for table, df in tables.items():
            fpath = os.path.join(out_dir, f"{schema}_{county}_{table}.csv")
            df2 = df.copy()
            if "profile_json" in df2.columns:
                df2["profile_json"] = df2["profile_json"].apply(lambda x: str(x))
            df2.to_csv(fpath, index=False)


if __name__ == "__main__":
    # One metro, two mid, three rural
    scales_small_demo = {
        "King": 25000,       # A: households, B: people, C: docs
        "Pierce": 10000,
        "Snohomish": 10000,
        "Ferry": 2000,
        "Garfield": 1500,
        "San Juan": 2500,
    }

    # Control *types* of bad data (optional)
    # e.g., more missing required + wrong types, fewer stale timestamps
    weights = {
        "missing_required": 3.0,
        "wrong_type": 2.0,
        "invalid_age": 1.5,
        "negative_income": 1.0,
        "nonsense_zip": 1.0,
        "inconsistent_codes": 1.0,
        "stale_timestamp": 0.3,
    }

    # Generate Schema A
    nodes_A = generate_all_nodes("A", scales_small_demo, bad_rate=0.03, error_weights=weights, seed=7)
    export_to_csv(nodes_A, out_dir="data-generation/data/schema_A", schema="A")
    # Generate Schema B
    nodes_B = generate_all_nodes("B", scales_small_demo, bad_rate=0.03, error_weights=weights, seed=7)
    export_to_csv(nodes_B, out_dir="data-generation/data/schema_B", schema="B")
    # Generate Schema C
    nodes_C = generate_all_nodes("C", scales_small_demo, bad_rate=0.03, error_weights=weights, seed=7)
    export_to_csv(nodes_C, out_dir="data-generation/data/schema_C", schema="C")

    # Quick sanity prints
    print("Schema A (King) rows:",
          {k: len(v) for k, v in nodes_A["King"].items() if k != "data_quality"},
          "bad:", len(nodes_A["King"]["data_quality"]))

    print("Schema B (King) rows:",
          {k: len(v) for k, v in nodes_B["King"].items() if k != "data_quality"},
          "bad:", len(nodes_B["King"]["data_quality"]))

    print("Schema C (King) rows:",
          {k: len(v) for k, v in nodes_C["King"].items() if k != "data_quality"},
          "bad:", len(nodes_C["King"]["data_quality"]))