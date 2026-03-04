"""
Synthetic county-level demographic data generator (PostgreSQL-friendly)

Features
--------
- 3 schema variants (A, B, C) to simulate heterogeneous local systems
- Residency lifecycle modeling for snapshot-style reporting:
    - moved_in_date, move_out_date, death_date, active_status
    - last_verified_date, verification_source
- Lightweight employment modeling to support unemployment metrics
- Bad-data injection with configurable error types and weights
- Outputs: pandas DataFrames per county + CSV export (Schema C JSON is serialized)

Dependencies
------------
pip install faker pandas numpy

Notes
-----
- This generator is intentionally "realistic-shaped" rather than actuarially exact.
- Residency validity is determined by dates and/or active flag as-of a chosen snapshot date.
- For Schema B, several fields are intentionally stored as strings in mixed formats.
"""

import random
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
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
    tier: str  # "metro", "mid", "rural"
    median_income: int
    income_spread: int
    hh_size_lambda: float

    # Realism knobs (simple, tunable)
    annual_move_rate: float          # approximate probability of moving out per year (tenure-adjusted)
    annual_death_rate_base: float    # baseline death probability per year (scaled by age curve)
    verification_mean_days: int      # average days between "record verification" events
    unemployment_rate: float         # approximate unemployment among labor-force participants (working-age)


COUNTIES: List[CountySpec] = [
    CountySpec(
        "King", "53033", "metro",
        median_income=105000, income_spread=45000, hh_size_lambda=2.2,
        annual_move_rate=0.12, annual_death_rate_base=0.0060, verification_mean_days=240,
        unemployment_rate=0.045,
    ),
    CountySpec(
        "Pierce", "53053", "mid",
        median_income=85000, income_spread=35000, hh_size_lambda=2.4,
        annual_move_rate=0.10, annual_death_rate_base=0.0060, verification_mean_days=270,
        unemployment_rate=0.050,
    ),
    CountySpec(
        "Snohomish", "53061", "mid",
        median_income=95000, income_spread=38000, hh_size_lambda=2.3,
        annual_move_rate=0.10, annual_death_rate_base=0.0060, verification_mean_days=270,
        unemployment_rate=0.048,
    ),
    CountySpec(
        "Ferry", "53019", "rural",
        median_income=60000, income_spread=25000, hh_size_lambda=2.2,
        annual_move_rate=0.08, annual_death_rate_base=0.0065, verification_mean_days=320,
        unemployment_rate=0.055,
    ),
    CountySpec(
        "Garfield", "53023", "rural",
        median_income=65000, income_spread=22000, hh_size_lambda=2.1,
        annual_move_rate=0.08, annual_death_rate_base=0.0065, verification_mean_days=320,
        unemployment_rate=0.055,
    ),
    CountySpec(
        "San Juan", "53055", "rural",
        median_income=90000, income_spread=42000, hh_size_lambda=2.0,
        annual_move_rate=0.09, annual_death_rate_base=0.0062, verification_mean_days=300,
        unemployment_rate=0.050,
    ),
]

# ----------------------------
# Demographic value pools
# ----------------------------

SEX_VALUES = ["M", "F", "X"]
SEX_CODES = {"M": "M", "F": "F", "X": "U"}  # Schema B uses U for unknown

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

INCOME_BRACKETS = ["<25k", "25-50k", "50-75k", "75-100k", "100-150k", "150-200k", "200k+"]
BRACKET_BOUNDS = {
    "<25k": (0, 25000),
    "25-50k": (25000, 50000),
    "50-75k": (50000, 75000),
    "75-100k": (75000, 100000),
    "100-150k": (100000, 150000),
    "150-200k": (150000, 200000),
    "200k+": (200000, 500000),
}

VERIFICATION_SOURCES = ["DMV", "Utility", "School", "Census", "Medicaid", "VoterReg", "Other"]

# ----------------------------
# Helpers
# ----------------------------

def _random_dob(min_age: int = 0, max_age: int = 95) -> date:
    """Random DOB using age bounds (inclusive)."""
    today = date.today()
    age = random.randint(min_age, max_age)
    start = today - timedelta(days=365 * (age + 1))
    end = today - timedelta(days=365 * age)
    return fake.date_between(start_date=start, end_date=end)

def _age_on(dob: date, as_of: date) -> int:
    """Compute age in years as-of a date."""
    years = as_of.year - dob.year
    if (as_of.month, as_of.day) < (dob.month, dob.day):
        years -= 1
    return max(0, years)

def _income_from_county(spec: CountySpec) -> int:
    """Income with county-specific location/scale; clipped to [0..500k]."""
    val = int(np.random.normal(loc=spec.median_income, scale=spec.income_spread))
    return int(np.clip(val, 0, 500000))

def _income_to_bracket(income: int) -> str:
    """Map numeric income to a bracket label."""
    for b, (lo, hi) in BRACKET_BOUNDS.items():
        if lo <= income < hi:
            return b
    return "200k+"

def _household_size(spec: CountySpec) -> int:
    """Poisson-skewed household size, clipped to [1..8]."""
    val = 1 + np.random.poisson(lam=max(0.8, spec.hh_size_lambda - 1.0))
    return int(np.clip(val, 1, 8))

def _zip_for_county(county_name: str) -> str:
    """WA-ish ZIP generator to avoid real ZIP mappings."""
    if county_name in ("King", "Snohomish", "Pierce"):
        base = random.choice([98, 98, 98, 99])  # more 98xxx
    else:
        base = random.choice([98, 99, 99])  # more 99xxx
    return f"{base}{random.randint(0, 999):03d}"

def _dob_to_mixed_string(d: date) -> str:
    """Schema B: mixed date formats by design."""
    fmt = random.choice(["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d"])
    return d.strftime(fmt)

def _now_ts() -> datetime:
    """UTC timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0)

def _normalize_bad_rate(bad_rate: float) -> float:
    """Normalize bad_rate to [0..1], accepting 0..100 as percent."""
    if bad_rate > 1.0:
        if bad_rate <= 100.0:
            bad_rate = bad_rate / 100.0
        else:
            raise ValueError("bad_rate must be in [0,1] or [0,100] as percent.")
    return max(0.0, min(1.0, float(bad_rate)))

def should_inject_bad_data(bad_rate: float) -> bool:
    """Decide whether to inject bad data given bad_rate."""
    bad_rate = _normalize_bad_rate(bad_rate)
    return bad_rate > 0 and (random.random() < bad_rate)

def _death_probability(county: CountySpec, age: int) -> float:
    """
    Simple age-based mortality curve.
    This is not actuarially accurate, but it's shaped to be plausible.
    """
    if age < 1:
        mult = 0.5
    elif age < 15:
        mult = 0.2
    elif age < 45:
        mult = 0.6
    elif age < 65:
        mult = 1.5
    elif age < 80:
        mult = 4.0
    else:
        mult = 8.0
    return min(0.25, county.annual_death_rate_base * mult)

def _move_multiplier_by_age(age: int) -> float:
    """Age-based mobility multiplier to make moves more realistic."""
    if 0 <= age < 18:
        return 0.7
    if 18 <= age <= 35:
        return 1.5
    if 36 <= age <= 64:
        return 1.0
    return 0.6  # 65+

def generate_residency_fields(county: CountySpec, dob: date, moved_in: date, as_of: date) -> dict:
    """
    Generate consistent residency lifecycle fields for a record.

    Fields returned
    --------------
    - active_status: bool
    - move_out_date: Optional[date]
    - death_date: Optional[date]
    - last_verified_date: date
    - verification_source: str
    """
    age = _age_on(dob, as_of)

    tenure_days = max(1, (as_of - moved_in).days)
    tenure_years = tenure_days / 365.0

    # Move-out probability over tenure using annual move rate; adjust by age.
    annual_move = max(0.0, min(0.9, county.annual_move_rate * _move_multiplier_by_age(age)))
    move_prob = 1.0 - (1.0 - annual_move) ** tenure_years
    moved_out = (random.random() < move_prob)

    move_out_date: Optional[date] = None
    if moved_out:
        max_days = max(1, (as_of - moved_in).days)
        move_out_date = moved_in + timedelta(days=random.randint(1, max_days))

    # Death probability accumulates loosely over tenure (scaled by age curve).
    death_prob = _death_probability(county, age) * tenure_years
    died = (random.random() < min(0.90, death_prob))

    death_date: Optional[date] = None
    if died:
        earliest = max(moved_in, dob)
        max_days = max(1, (as_of - earliest).days)
        death_date = earliest + timedelta(days=random.randint(1, max_days))

    # Active as-of derived from move_out / death.
    active = True
    if death_date is not None and death_date <= as_of:
        active = False
    if move_out_date is not None and move_out_date <= as_of:
        active = False

    # Verification recency: exponential-ish distribution around mean.
    mean = max(30, int(county.verification_mean_days))
    delta_days = int(np.clip(np.random.exponential(scale=mean), 0, 3 * mean))
    last_verified = as_of - timedelta(days=delta_days)

    return {
        "active_status": active,
        "move_out_date": move_out_date,
        "death_date": death_date,
        "last_verified_date": last_verified,
        "verification_source": random.choice(VERIFICATION_SOURCES),
    }

def generate_employment_fields(county: CountySpec, dob: date, as_of: date) -> dict:
    """
    Generate simple employment status for unemployment-rate aggregation.

    - Under 16: NotInLaborForce
    - 16-64: Employed / Unemployed / NotInLaborForce
    - 65+: NotInLaborForce (mostly)

    Returns:
    - employment_status: str in {"Employed","Unemployed","NotInLaborForce"}
    - has_job: bool (True only when employed)
    """
    age = _age_on(dob, as_of)

    if age < 16:
        status = "NotInLaborForce"
    elif age >= 65:
        # Some seniors still work; keep small chance for realism.
        if random.random() < 0.08:
            status = "Employed"
        else:
            status = "NotInLaborForce"
    else:
        # Working-age: participation + unemployment
        # Participation: mostly in labor force, but not all
        participation = 0.88 if county.tier == "metro" else (0.85 if county.tier == "mid" else 0.83)
        if random.random() > participation:
            status = "NotInLaborForce"
        else:
            # In labor force: unemployed_rate chance unemployed
            if random.random() < max(0.0, min(0.5, county.unemployment_rate)):
                status = "Unemployed"
            else:
                status = "Employed"

    return {"employment_status": status, "has_job": (status == "Employed")}

# Small helper to allow "type drift" within allowed values too
def housing_badge(housing_type: str) -> str:
    """Occasionally store a more verbose label for realism."""
    if random.random() < 0.15 and housing_type == "SFH":
        return "Single Family"
    return housing_type

# ----------------------------
# Bad data injection
# ----------------------------

BAD_ERROR_TYPES = [
    "invalid_age",
    "negative_income",
    "nonsense_zip",
    "wrong_type",
    "inconsistent_codes",
    "stale_timestamp",
    # Residency-focused issues
    "move_out_before_move_in",
    "death_before_birth",
    "active_inconsistent_with_dates",
    "missing_verification",
]

def inject_bad_data_record(
    record: dict,
    schema: str,
    bad_rate: float,
    error_weights: Optional[Dict[str, float]] = None,
) -> Tuple[dict, Optional[str]]:
    """
    With probability bad_rate, mutate record to contain a data-quality issue.
    Returns (record, error_type_or_None).
    """
    bad_rate = _normalize_bad_rate(bad_rate)
    if bad_rate <= 0 or random.random() > bad_rate:
        return record, None

    weights = error_weights or {k: 1.0 for k in BAD_ERROR_TYPES}
    keys = list(weights.keys())
    w = np.array([max(0.0, float(weights.get(k, 0.0))) for k in keys], dtype=float)
    if w.sum() <= 0:
        keys = BAD_ERROR_TYPES
        w = np.ones(len(keys), dtype=float)
    w = w / w.sum()

    error_type = str(np.random.choice(keys, p=w))

    # Apply an error depending on schema. This function is table-agnostic; it may add extra keys.
    if error_type == "invalid_age":
        # set DOB to future date
        if schema == "A":
            record["dob"] = date.today() + timedelta(days=random.randint(1, 3650))
        elif schema == "B":
            future = date.today() + timedelta(days=random.randint(1, 3650))
            record["dob_str"] = future.strftime("%Y-%m-%d")
        else:  # C
            pj = record.get("profile_json", {})
            pj.setdefault("demographics", {})
            pj["demographics"]["dob"] = (date.today() + timedelta(days=999)).isoformat()
            record["profile_json"] = pj

    elif error_type == "negative_income":
        if schema == "A":
            # household income or other income field
            if "income_usd" in record and record["income_usd"] is not None:
                record["income_usd"] = -abs(int(record["income_usd"]))
            else:
                record["income_usd"] = -5000
        elif schema == "B":
            record["income_bracket"] = "-50k"  # invalid bracket label
        else:
            pj = record.get("profile_json", {})
            pj.setdefault("household", {})
            pj["household"]["income_usd"] = -5000
            record["profile_json"] = pj

    elif error_type == "nonsense_zip":
        if schema == "A":
            record["zip"] = "ABCDE"
        elif schema == "B":
            record["full_address"] = (record.get("full_address") or "") + " ZIP=XXXXX"
        else:
            pj = record.get("profile_json", {})
            pj.setdefault("location", {})
            pj["location"]["zip"] = "12"
            record["profile_json"] = pj

    elif error_type == "wrong_type":
        # store numbers as strings, etc.
        if schema == "A":
            if "hh_size" in record:
                record["hh_size"] = str(record.get("hh_size", "2"))
            else:
                record["hh_size"] = "two"
        elif schema == "B":
            record["household_size"] = "two"
        else:
            pj = record.get("profile_json", {})
            pj.setdefault("household", {})
            pj["household"]["hh_size"] = "3"
            record["profile_json"] = pj

    elif error_type == "inconsistent_codes":
        if schema == "B":
            record["gender_code"] = "Z"
            record["race_code"] = "?"
        elif schema == "A":
            record["sex"] = "U"
        else:
            pj = record.get("profile_json", {})
            pj.setdefault("demographics", {})
            pj["demographics"]["race"] = 999  # invalid type
            record["profile_json"] = pj

    elif error_type == "stale_timestamp":
        stale = datetime(1999, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        if schema == "A":
            record["created_at_utc"] = stale
        elif schema == "B":
            record["updated_ts_utc"] = stale
        else:
            record["created_at_utc"] = stale

    elif error_type == "move_out_before_move_in":
        if schema == "A":
            # For A: move_out_date earlier than moved_in_date
            mi = record.get("moved_in_date", date.today())
            record["move_out_date"] = mi - timedelta(days=random.randint(1, 365))
        elif schema == "B":
            record["move_out_str"] = "01/01/2010"
            record["moved_in_str"] = "01/01/2020"
        else:
            pj = record.get("profile_json", {})
            pj.setdefault("residency", {})
            pj["residency"]["move_out_date"] = "2010-01-01"
            pj["residency"]["moved_in_date"] = "2020-01-01"
            record["profile_json"] = pj

    elif error_type == "death_before_birth":
        if schema == "A":
            dob = record.get("dob", date(2000, 1, 1))
            record["death_date"] = dob - timedelta(days=random.randint(1, 3650))
        elif schema == "B":
            record["death_str"] = "01/01/1980"
            record["dob_str"] = "01/01/2000"
        else:
            pj = record.get("profile_json", {})
            pj.setdefault("demographics", {})
            pj.setdefault("residency", {})
            pj["demographics"]["dob"] = "2000-01-01"
            pj["residency"]["death_date"] = "1980-01-01"
            record["profile_json"] = pj

    elif error_type == "active_inconsistent_with_dates":
        # Active flag contradicts move-out/death
        if schema == "A":
            record["active_status"] = True
            record["move_out_date"] = date.today() - timedelta(days=30)
        elif schema == "B":
            record["active_flag"] = "Y"
            record["move_out_str"] = "01/01/2020"
        else:
            pj = record.get("profile_json", {})
            pj.setdefault("residency", {})
            pj["residency"]["active_status"] = True
            pj["residency"]["move_out_date"] = "2020-01-01"
            record["profile_json"] = pj

    elif error_type == "missing_verification":
        if schema == "A":
            record["last_verified_date"] = None
            record["verification_source"] = None
        elif schema == "B":
            record["last_verified_str"] = ""
            record["verification_source"] = ""
        else:
            pj = record.get("profile_json", {})
            pj.setdefault("residency", {})
            pj["residency"]["last_verified_date"] = None
            pj["residency"]["verification_source"] = None
            record["profile_json"] = pj

    return record, error_type

# ----------------------------
# Schema emitters
# ----------------------------

def generate_schema_A(
    county: CountySpec,
    n_households: int,
    bad_rate: float,
    error_weights: Optional[Dict[str, float]] = None,
    as_of: Optional[date] = None,
):
    """
    Schema A (normalized-ish):
      - household, address, resident
      - adds residency + employment fields to resident for snapshot validation

    Returns: residents_df, households_df, addresses_df, dq_df
    """
    as_of = as_of or date.today()

    households: List[dict] = []
    addresses: List[dict] = []
    residents: List[dict] = []
    dq: List[dict] = []

    for _ in range(n_households):
        household_id = uuid.uuid4()
        hh_size = _household_size(county)
        income = _income_from_county(county)
        housing_type = random.choice(HOUSING_TYPES)

        address_id = uuid.uuid4()
        street = fake.street_address()
        city = fake.city()
        state = "WA"
        zip_code = _zip_for_county(county.name)

        # operational timestamp (kept for realism; not required for residency)
        created_at_utc = _now_ts()

        hrow = {
            "household_id": household_id,
            "county_fips": county.fips,
            "hh_size": hh_size,
            "income_usd": income,
            "housing_type": housing_badge(housing_type),
            "created_at_utc": created_at_utc,
        }
        arow = {
            "address_id": address_id,
            "street": street,
            "city": city,
            "state": state,
            "zip": zip_code,
        }

        # Optionally inject correlated badness per household cluster
        cluster_bad = should_inject_bad_data(bad_rate)
        if cluster_bad:
            hrow, e1 = inject_bad_data_record(hrow, "A", bad_rate=1.0, error_weights=error_weights)
        else:
            e1 = None
        if e1:
            dq.append({"county": county.name, "schema": "A", "table": "household", "error_type": e1})

        if cluster_bad and random.random() < 0.5:
            arow, e2 = inject_bad_data_record(arow, "A", bad_rate=1.0, error_weights=error_weights)
        else:
            e2 = None
        if e2:
            dq.append({"county": county.name, "schema": "A", "table": "address", "error_type": e2})

        households.append(hrow)
        addresses.append(arow)

        # Create residents for the household
        # Household move-in anchor date (helps realism); individual residents get small jitter
        household_moved_in = fake.date_between(start_date="-10y", end_date=as_of)

        for _r in range(int(hh_size)):
            dob = _random_dob(0, 95)
            sex = random.choice(SEX_VALUES)
            race = random.choice(RACE_VALUES)
            ethnicity = random.choice(ETHNICITY_VALUES)

            # Jitter moved-in by up to +/- 60 days (keep within plausible bounds)
            jitter = random.randint(-30, 60)
            moved_in_date = household_moved_in + timedelta(days=jitter)
            # Prevent moved_in after as_of
            if moved_in_date > as_of:
                moved_in_date = as_of - timedelta(days=random.randint(0, 30))
            # Prevent moved_in before 10 years window too aggressively
            if moved_in_date < (as_of - timedelta(days=365 * 10)):
                moved_in_date = as_of - timedelta(days=365 * 10)

            residency = generate_residency_fields(county, dob, moved_in_date, as_of)
            employment = generate_employment_fields(county, dob, as_of)

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

                # Residency lifecycle
                "moved_in_date": moved_in_date,
                "move_out_date": residency["move_out_date"],
                "death_date": residency["death_date"],
                "active_status": residency["active_status"],
                "last_verified_date": residency["last_verified_date"],
                "verification_source": residency["verification_source"],

                # Employment
                "employment_status": employment["employment_status"],
                "has_job": employment["has_job"],
            }

            if cluster_bad and random.random() < 0.5:
                rrow, e3 = inject_bad_data_record(rrow, "A", bad_rate=1.0, error_weights=error_weights)
            else:
                e3 = None
            if e3:
                dq.append({"county": county.name, "schema": "A", "table": "resident", "error_type": e3})

            residents.append(rrow)

    RESIDENT_COLS = [
        "resident_id", "household_id", "first_name", "last_name",
        "dob", "sex", "race", "ethnicity", "address_id",
        "moved_in_date", "move_out_date", "death_date",
        "active_status", "last_verified_date", "verification_source",
        "employment_status", "has_job",
    ]
    HOUSEHOLD_COLS = ["household_id", "county_fips", "hh_size", "income_usd", "housing_type", "created_at_utc"]
    ADDRESS_COLS = ["address_id", "street", "city", "state", "zip"]

    res_df = pd.DataFrame(residents)
    hh_df = pd.DataFrame(households)
    addr_df = pd.DataFrame(addresses)
    dq_df = pd.DataFrame(dq)

    # Ensure stable column order (missing cols handled gracefully)
    res_df = res_df.reindex(columns=RESIDENT_COLS)
    hh_df = hh_df.reindex(columns=HOUSEHOLD_COLS)
    addr_df = addr_df.reindex(columns=ADDRESS_COLS)

    return res_df, hh_df, addr_df, dq_df


def generate_schema_B(
    county: CountySpec,
    n_records: int,
    bad_rate: float,
    error_weights: Optional[Dict[str, float]] = None,
    as_of: Optional[date] = None,
):
    """
    Schema B (single table, messy strings and codes):
      - Adds string-based residency and employment fields for normalization challenges.

    Returns: person_df, dq_df
    """
    as_of = as_of or date.today()

    rows: List[dict] = []
    dq: List[dict] = []

    for _ in range(n_records):
        dob = _random_dob(0, 95)
        sex = random.choice(SEX_VALUES)
        race = random.choice(RACE_VALUES)
        ethnicity = random.choice(ETHNICITY_VALUES)
        income = _income_from_county(county)

        full_addr = f"{fake.street_address()}, {fake.city()}, WA {_zip_for_county(county.name)}"

        moved_in = fake.date_between(start_date="-10y", end_date=as_of)
        residency = generate_residency_fields(county, dob, moved_in, as_of)
        employment = generate_employment_fields(county, dob, as_of)

        row = {
            "person_uuid": str(uuid.uuid4()),
            "county": county.name,  # not FIPS
            "dob_str": _dob_to_mixed_string(dob),
            "gender_code": SEX_CODES[sex],
            "race_code": RACE_CODES[race],
            "eth_flag": ETH_FLAG[ethnicity],
            "income_bracket": _income_to_bracket(income),
            # household_size intentionally allowed to drift to wrong types via bad data
            "household_size": _household_size(county),
            "full_address": full_addr,

            # operational timestamp (kept for realism)
            "updated_ts_utc": _now_ts(),

            # Residency lifecycle stored as strings (mixed formats)
            "moved_in_str": _dob_to_mixed_string(moved_in),
            "move_out_str": _dob_to_mixed_string(residency["move_out_date"]) if residency["move_out_date"] else "",
            "death_str": _dob_to_mixed_string(residency["death_date"]) if residency["death_date"] else "",
            "active_flag": "Y" if residency["active_status"] else "N",
            "last_verified_str": _dob_to_mixed_string(residency["last_verified_date"]),
            "verification_source": residency["verification_source"],

            # Employment (codes/strings)
            "employment_status": employment["employment_status"],
            "has_job_flag": "Y" if employment["has_job"] else "N",
        }

        row, e = inject_bad_data_record(row, "B", bad_rate, error_weights)
        if e:
            dq.append({"county": county.name, "schema": "B", "table": "person_record", "error_type": e})

        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(dq)


def generate_schema_C(
    county: CountySpec,
    n_docs: int,
    bad_rate: float,
    error_weights: Optional[Dict[str, float]] = None,
    as_of: Optional[date] = None,
):
    """
    Schema C (JSON documents):
      - profile_json contains nested demographics/household/location/residency/employment
      - Adds residency and employment sections for snapshot validation

    Returns: citizen_df, dq_df
    """
    as_of = as_of or date.today()

    rows: List[dict] = []
    dq: List[dict] = []

    for _ in range(n_docs):
        dob = _random_dob(0, 95)
        sex = random.choice(SEX_VALUES)
        race = random.choice(RACE_VALUES)
        ethnicity = random.choice(ETHNICITY_VALUES)
        income = _income_from_county(county)

        moved_in = fake.date_between(start_date="-10y", end_date=as_of)
        residency = generate_residency_fields(county, dob, moved_in, as_of)
        employment = generate_employment_fields(county, dob, as_of)

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
                "zip": _zip_for_county(county.name),
            },
            "residency": {
                "active_status": residency["active_status"],
                "moved_in_date": moved_in.isoformat(),
                "move_out_date": residency["move_out_date"].isoformat() if residency["move_out_date"] else None,
                "death_date": residency["death_date"].isoformat() if residency["death_date"] else None,
                "last_verified_date": residency["last_verified_date"].isoformat(),
                "verification_source": residency["verification_source"],
            },
            "employment": {
                "employment_status": employment["employment_status"],
                "has_job": employment["has_job"],
            }
        }

        row = {
            "doc_id": str(uuid.uuid4()),
            "county_fips": county.fips,
            "profile_json": profile,
            "created_at_utc": _now_ts(),  # operational timestamp
        }

        row, e = inject_bad_data_record(row, "C", bad_rate, error_weights)
        if e:
            dq.append({"county": county.name, "schema": "C", "table": "citizen", "error_type": e})

        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(dq)

# ----------------------------
# Orchestrator
# ----------------------------

def generate_county_data(
    schema: str,
    county: CountySpec,
    scale: int,
    bad_rate: float = 0.02,
    error_weights: Optional[Dict[str, float]] = None,
    as_of: Optional[date] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate per-county data for one schema.

    Parameters
    ----------
    schema : "A" | "B" | "C"
    scale :
      - schema A: number of households
      - schema B: number of person_records
      - schema C: number of citizen docs
    bad_rate : fraction of rows to corrupt (approx), 0..1 (or 0..100 for percent)
    as_of : snapshot date for residency/employment generation

    Returns
    -------
    dict of DataFrames by table name
    """
    as_of = as_of or date.today()
    schema = schema.upper().strip()

    if schema == "A":
        residents, households, addresses, dq = generate_schema_A(
            county, scale, bad_rate, error_weights=error_weights, as_of=as_of
        )
        return {
            "resident": residents,
            "household": households,
            "address": addresses,
            "data_quality": dq,
        }

    if schema == "B":
        person, dq = generate_schema_B(
            county, scale, bad_rate, error_weights=error_weights, as_of=as_of
        )
        return {"person_record": person, "data_quality": dq}

    if schema == "C":
        citizen, dq = generate_schema_C(
            county, scale, bad_rate, error_weights=error_weights, as_of=as_of
        )
        return {"citizen": citizen, "data_quality": dq}

    raise ValueError("schema must be one of: A, B, C")

def generate_all_nodes(
    schema: str,
    county_scales: Dict[str, int],
    bad_rate: float = 0.02,
    error_weights: Optional[Dict[str, float]] = None,
    seed: int = 42,
    as_of: Optional[date] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate data for multiple counties for a given schema.

    Parameters
    ----------
    county_scales : dict like {"King": 80000, "Pierce": 30000, ...}
    seed : ensures reproducibility (random, numpy, faker)

    Returns
    -------
    {county_name: {table: df, ...}, ...}
    """
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)

    as_of = as_of or date.today()

    by_name = {c.name: c for c in COUNTIES}
    out: Dict[str, Dict[str, pd.DataFrame]] = {}

    for county_name, scale in county_scales.items():
        if county_name not in by_name:
            raise KeyError(f"Unknown county '{county_name}'. Valid: {sorted(by_name.keys())}")
        county = by_name[county_name]
        out[county_name] = generate_county_data(
            schema, county, scale, bad_rate=bad_rate, error_weights=error_weights, as_of=as_of
        )

    return out

def export_to_csv(node_data: Dict[str, Dict[str, pd.DataFrame]], out_dir: str, schema: str):
    """
    Write each county/table to CSV.
    For schema C, profile_json is serialized to a JSON string.

    Output filenames: "{schema}_{county}_{table}.csv"
    """
    import os
    import json

    os.makedirs(out_dir, exist_ok=True)

    for county, tables in node_data.items():
        for table, df in tables.items():
            fpath = os.path.join(out_dir, f"{schema}_{county}_{table}.csv")
            df2 = df.copy()

            if "profile_json" in df2.columns:
                df2["profile_json"] = df2["profile_json"].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
                )

            df2.to_csv(fpath, index=False)

# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    # Example: one metro, two mid, and two rural nodes (scale interpreted per schema)
    scales_small_demo = {
        "King": 25000,       # A: households, B: people, C: docs
        "Pierce": 10000,
        "Snohomish": 10000,
        "Ferry": 2000,
        "San Juan": 3000,
    }

    # Weighting for bad data injection; set to 0.0 to disable a specific error type
    weights = {
        "wrong_type": 2.0,
        "invalid_age": 1.5,
        "negative_income": 1.0,
        "nonsense_zip": 1.0,
        "inconsistent_codes": 1.0,
        "stale_timestamp": 0.3,
        "move_out_before_move_in": 0.8,
        "death_before_birth": 0.3,
        "active_inconsistent_with_dates": 0.6,
        "missing_verification": 0.6,
    }

    # Choose a stable snapshot date for reproducible "as-of" reporting
    as_of_snapshot = date(2026, 2, 1)

    # Generate Schema A
    nodes_A = generate_all_nodes(
        "A", scales_small_demo, bad_rate=0.03, error_weights=weights, seed=7, as_of=as_of_snapshot
    )
    export_to_csv(nodes_A, out_dir="data-generation/data/schema_A", schema="A")

    # Generate Schema B
    nodes_B = generate_all_nodes(
        "B", scales_small_demo, bad_rate=0.03, error_weights=weights, seed=7, as_of=as_of_snapshot
    )
    export_to_csv(nodes_B, out_dir="data-generation/data/schema_B", schema="B")

    # Generate Schema C
    nodes_C = generate_all_nodes(
        "C", scales_small_demo, bad_rate=0.03, error_weights=weights, seed=7, as_of=as_of_snapshot
    )
    export_to_csv(nodes_C, out_dir="data-generation/data/schema_C", schema="C")

    # Summary output
    for county in scales_small_demo.keys():
        print(f"County: {county}")
        print("  Schema A:", {k: len(v) for k, v in nodes_A[county].items() if k != "data_quality"},
              "bad:", len(nodes_A[county]["data_quality"]))
        print("  Schema B:", {k: len(v) for k, v in nodes_B[county].items() if k != "data_quality"},
              "bad:", len(nodes_B[county]["data_quality"]))
        print("  Schema C:", {k: len(v) for k, v in nodes_C[county].items() if k != "data_quality"},
              "bad:", len(nodes_C[county]["data_quality"]))