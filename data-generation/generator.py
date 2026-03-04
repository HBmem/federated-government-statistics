#!/usr/bin/env python3
"""
Synthetic data generator for federated county datasets.

This script generates synthetic CSV files for three schema designs:
- Schema A: address.csv, household.csv, resident.csv
- Schema B: personal_record.csv (single table, many fields are strings)
- Schema C: citizen.csv (single table with JSON profile)

It also injects controllable "bad data" into a fraction of rows using weighted types.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import string
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Helpers: filesystem + rng
# -------------------------

def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    """Read a JSON file and return parsed data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, header: List[str], rows: List[Dict[str, Any]]) -> None:
    """Write rows to CSV using a fixed header ordering."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# -------------------------
# Weighted selection helpers
# -------------------------

def weighted_choice(weights: Dict[str, float], rng: random.Random) -> str:
    """Pick a key from a weight dict."""
    keys = list(weights.keys())
    vals = [float(weights[k]) for k in keys]
    total = sum(vals)
    if total <= 0:
        return keys[0]
    pick = rng.uniform(0, total)
    acc = 0.0
    for k, w in zip(keys, vals):
        acc += w
        if pick <= acc:
            return k
    return keys[-1]


def prob(rng: random.Random, p: float) -> bool:
    """Return True with probability p."""
    return rng.random() < p


# -------------------------
# Synthetic value generation
# -------------------------

FIRST_NAMES = [
    "Ariana", "Marc", "Jordan", "Priya", "Noah", "Mia", "Liam", "Sofia", "Ethan",
    "Zoe", "Camila", "Kai", "Ava", "Leo", "Nina", "Mateo", "Iris", "Amir"
]
LAST_NAMES = [
    "Meza", "Salas", "Nguyen", "Smith", "Patel", "Kim", "Johnson", "Brown", "Lopez",
    "Garcia", "Chen", "Williams", "Davis", "Martinez", "Taylor"
]
HOUSING_TYPES = ["Apartment", "House", "Townhome", "MobileHome", "Condo"]
VERIFICATION_SOURCES = ["DMV", "Medicaid", "School", "Utilities", "Tax", "Employer"]
GENDER_CODES_B = ["M", "F", "X"]
RACE_CODES_B = ["W", "B", "A", "P", "N", "O"]  # White, Black, Asian, Pacific, Native, Other
ETH_FLAGS_B = ["Y", "N"]
EMPLOYMENT_STATUSES = ["Employed", "Unemployed", "NotInLaborForce"]


def rand_zip(rng: random.Random) -> str:
    """Generate a 5-digit ZIP code string."""
    return "".join(rng.choice(string.digits) for _ in range(5))


def rand_street(rng: random.Random) -> str:
    """Generate a synthetic street address line."""
    num = rng.randint(1, 99999)
    street = rng.choice(["Cervantes", "Smith", "Aguilar", "Pine", "Union", "Rainier", "Madison"])
    suffix = rng.choice(["St", "Ave", "Blvd", "Rd", "Ln", "Way", "Common", "Forks"])
    extra = rng.choice(["", " Apt. " + str(rng.randint(1, 999)), " Unit " + str(rng.randint(1, 999))])
    return f"{num} {street} {suffix}{extra}".strip()


def rand_city(rng: random.Random) -> str:
    """Generate a synthetic city name."""
    return rng.choice(["Seattle", "Tacoma", "Everett", "Spokane", "Olympia", "South Donna", "Thomasbury", "Jonesland"])


def rand_timestamp_utc(rng: random.Random, min_year: int, max_year: int) -> str:
    """Generate an ISO-ish timestamp string with UTC offset."""
    start = datetime(min_year, 1, 1)
    end = datetime(max_year, 12, 31, 23, 59, 59)
    delta = end - start
    dt = start + timedelta(seconds=rng.randint(0, int(delta.total_seconds())))
    # Store with "+00:00" to match your sample.
    return dt.strftime("%Y-%m-%d %H:%M:%S+00:00")


def rand_date(rng: random.Random, min_year: int, max_year: int) -> date:
    """Generate a random date in [min_year, max_year]."""
    start = date(min_year, 1, 1)
    end = date(max_year, 12, 31)
    days = (end - start).days
    return start + timedelta(days=rng.randint(0, days))


def format_date_multiple(d: Optional[date], rng: random.Random) -> Optional[str]:
    """
    Format a date into multiple string formats.
    Used for Schema B to stress parsing.

    Returns None if d is None.
    """
    if d is None:
        return None
    fmts = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"]
    fmt = rng.choice(fmts)
    return d.strftime(fmt)


def pick_from_dist(dist: Dict[str, float], rng: random.Random) -> str:
    """Pick a key based on probability distribution."""
    keys = list(dist.keys())
    weights = [dist[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def income_usd(rng: random.Random) -> int:
    """Generate a plausible annual household income."""
    # Simple log-ish distribution: many middle incomes, fewer extremes.
    base = int(rng.triangular(15000, 220000, 65000))
    return max(0, base)


def income_bracket_from_income(inc: int) -> str:
    """Map numeric income to a bracket string similar to Schema B example."""
    if inc < 25000:
        return "<25k"
    if inc < 50000:
        return "25-50k"
    if inc < 75000:
        return "50-75k"
    if inc < 100000:
        return "75-100k"
    if inc < 150000:
        return "100-150k"
    if inc < 200000:
        return "150-200k"
    return "200k+"


def compute_has_job(employment_status: str, rng: random.Random) -> bool:
    """Derive has_job from employment status with a small inconsistency chance handled later."""
    if employment_status == "Employed":
        return True
    if employment_status == "Unemployed":
        return False
    # Not in labor force => usually false
    return False


def compute_age_years(dob: date, ref: date) -> int:
    """Compute integer age in years at reference date."""
    years = ref.year - dob.year
    if (ref.month, ref.day) < (dob.month, dob.day):
        years -= 1
    return years


# -------------------------
# Bad data injection
# -------------------------

def inject_bad_data_row(row: Dict[str, Any], schema: str, bad_type: str, rng: random.Random) -> None:
    """
    Mutate a row in-place to introduce a specific kind of bad data.

    The mutation is schema-aware but intentionally imperfect.
    """
    # Helper fields: try multiple possible keys depending on schema output.
    def set_if_exists(keys: List[str], value: Any) -> bool:
        for k in keys:
            if k in row:
                row[k] = value
                return True
        return False

    if bad_type == "wrong_type":
        # Put a non-date string in a date field or a non-numeric in income.
        if not set_if_exists(["dob", "dob_str"], "not-a-date"):
            set_if_exists(["income_usd", "income_bracket"], "NaN")

    elif bad_type == "invalid_age":
        # Set DOB to a future date.
        future = date.today() + timedelta(days=rng.randint(30, 3650))
        set_if_exists(["dob"], future.isoformat())
        set_if_exists(["dob_str"], future.isoformat())

    elif bad_type == "negative_income":
        set_if_exists(["income_usd"], -rng.randint(1, 50000))
        # Schema B stores bracket; simulate nonsense bracket
        set_if_exists(["income_bracket"], "-10k")

    elif bad_type == "nonsense_zip":
        # Invalid ZIP (non-digits or wrong length)
        set_if_exists(["zip"], "ABCDE")
        # Schema B full address is one string; append garbage zip
        if "full_address" in row:
            row["full_address"] = row["full_address"].rsplit(" ", 1)[0] + " Z1P!!"

    elif bad_type == "inconsistent_codes":
        # Race/gender/eth combos that don't match the normal coding scheme.
        set_if_exists(["sex"], "Z")
        set_if_exists(["gender_code"], "Q")
        set_if_exists(["race"], "???")
        set_if_exists(["race_code"], "9")
        set_if_exists(["ethnicity"], "??")
        set_if_exists(["eth_flag"], "U")

    elif bad_type == "stale_timestamp":
        # Updated timestamp far in the past.
        set_if_exists(["updated_ts_utc", "created_at_utc"], "1990-01-01 00:00:00+00:00")

    elif bad_type == "move_out_before_move_in":
        # Swap moved-in/out ordering by setting move_out earlier.
        # Works best on schema A/B where these fields exist as strings.
        if "moved_in_str" in row and row["moved_in_str"]:
            row["move_out_str"] = "01/01/2010"
        if "moved_in_date" in row and row["moved_in_date"]:
            row["move_out_date"] = "2010-01-01"

    elif bad_type == "death_before_birth":
        # Death date earlier than DOB.
        set_if_exists(["death_date", "death_str"], "1900-01-01")
        set_if_exists(["dob", "dob_str"], "2000-01-01")

    elif bad_type == "active_inconsistent_with_dates":
        # Mark active but also set move_out_date.
        set_if_exists(["active_status"], True)
        set_if_exists(["active_flag"], "Y")
        set_if_exists(["move_out_date", "move_out_str"], "2020-01-01")

    elif bad_type == "missing_verification":
        # Remove verification info.
        set_if_exists(["verification_source"], "")
        set_if_exists(["last_verified_date", "last_verified_str"], "")

    # For Schema C (JSON), the "row" we write contains a JSON string in profile_json.
    # We inject JSON badness separately where profile_json is constructed.


def inject_bad_data_into_profile(profile: Dict[str, Any], bad_type: str, rng: random.Random) -> None:
    """Introduce bad data directly into Schema C JSON profile object."""
    if bad_type == "wrong_type":
        profile.setdefault("demographics", {})["dob"] = "not-a-date"
    elif bad_type == "invalid_age":
        future = (date.today() + timedelta(days=rng.randint(30, 3650))).isoformat()
        profile.setdefault("demographics", {})["dob"] = future
    elif bad_type == "negative_income":
        profile.setdefault("household", {})["income_usd"] = -rng.randint(1, 50000)
    elif bad_type == "nonsense_zip":
        profile.setdefault("location", {})["zip"] = "ABCDE"
    elif bad_type == "inconsistent_codes":
        profile.setdefault("demographics", {})["sex"] = "Z"
        profile.setdefault("demographics", {})["race"] = "???"
    elif bad_type == "stale_timestamp":
        # Timestamp stored outside profile; handled at row level.
        pass
    elif bad_type == "move_out_before_move_in":
        profile.setdefault("residency", {})["moved_in_date"] = "2022-01-01"
        profile.setdefault("residency", {})["move_out_date"] = "2010-01-01"
    elif bad_type == "death_before_birth":
        profile.setdefault("demographics", {})["dob"] = "2000-01-01"
        profile.setdefault("residency", {})["death_date"] = "1900-01-01"
    elif bad_type == "active_inconsistent_with_dates":
        profile.setdefault("residency", {})["active_status"] = True
        profile.setdefault("residency", {})["move_out_date"] = "2020-01-01"
    elif bad_type == "missing_verification":
        profile.setdefault("residency", {})["verification_source"] = ""
        profile.setdefault("residency", {})["last_verified_date"] = ""


# -------------------------
# Schema row builders
# -------------------------

@dataclass
class CountySpec:
    label: str
    fips: str
    name: str
    rows: int


def build_schema_a(county: CountySpec, cfg: Dict[str, Any], rng: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build Schema A tables:
    - address(address_id, street, city, state, zip)
    - household(household_id, county_fips, hh_size, income_usd, housing_type, created_at_utc)
    - resident(resident_id, household_id, first_name, last_name, dob, sex, race, ethnicity, address_id,
               moved_in_date, move_out_date, death_date, active_status, last_verified_date, verification_source,
               employment_status, has_job_flag)
    """
    min_year = cfg["date_bounds"]["min_year"]
    max_year = cfg["date_bounds"]["max_year"]
    dist = cfg["distributions"]

    addresses: List[Dict[str, Any]] = []
    households: List[Dict[str, Any]] = []
    residents: List[Dict[str, Any]] = []

    # Simple approach: each resident gets its own household/address with a chance of sharing.
    # This avoids needing complex household size consistency logic, while still enabling joins.
    share_household_prob = 0.30

    household_pool: List[Dict[str, Any]] = []
    address_pool: List[Dict[str, Any]] = []

    for _ in range(county.rows):
        # Possibly reuse household/address to create joins.
        if household_pool and prob(rng, share_household_prob):
            hh = rng.choice(household_pool)
        else:
            hh_id = str(uuid.uuid4())
            inc = income_usd(rng)
            hh = {
                "household_id": hh_id,
                "county_fips": county.fips,
                "hh_size": rng.randint(1, 6),
                "income_usd": inc,
                "housing_type": rng.choice(HOUSING_TYPES),
                "created_at_utc": rand_timestamp_utc(rng, min_year, max_year),
            }
            households.append(hh)
            household_pool.append(hh)

        if address_pool and prob(rng, share_household_prob):
            addr = rng.choice(address_pool)
        else:
            addr_id = str(uuid.uuid4())
            addr = {
                "address_id": addr_id,
                "street": rand_street(rng),
                "city": rand_city(rng),
                "state": "WA",
                "zip": rand_zip(rng)
            }
            addresses.append(addr)
            address_pool.append(addr)

        dob = rand_date(rng, 1930, 2008)
        ref = rand_date(rng, 2024, max_year)
        emp_status = pick_from_dist(dist["employment_status"], rng)
        has_job = compute_has_job(emp_status, rng)

        moved_in = rand_date(rng, min_year, max_year)
        # Move out sometimes.
        move_out = rand_date(rng, moved_in.year, max_year) if prob(rng, 0.22) else None
        # Death sometimes.
        death = rand_date(rng, 2000, max_year) if prob(rng, 0.08) else None

        active = True
        if move_out is not None or death is not None:
            active = False

        resident = {
            "resident_id": str(uuid.uuid4()),
            "household_id": hh["household_id"],
            "first_name": rng.choice(FIRST_NAMES),
            "last_name": rng.choice(LAST_NAMES),
            "dob": dob.isoformat(),
            "sex": pick_from_dist(dist["sex"], rng),
            "race": pick_from_dist(dist["race"], rng),
            "ethnicity": pick_from_dist(dist["ethnicity"], rng),
            "address_id": addr["address_id"],
            "moved_in_date": moved_in.isoformat(),
            "move_out_date": move_out.isoformat() if move_out else "",
            "death_date": death.isoformat() if death else "",
            "active_status": active,
            "last_verified_date": rand_date(rng, 2020, max_year).isoformat(),
            "verification_source": rng.choice(VERIFICATION_SOURCES),
            "employment_status": emp_status,
            "has_job_flag": has_job
        }
        residents.append(resident)

    return addresses, households, residents


def build_schema_b(county: CountySpec, cfg: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """
    Build Schema B table person_record with many string fields and mixed date formats.
    """
    min_year = cfg["date_bounds"]["min_year"]
    max_year = cfg["date_bounds"]["max_year"]
    dist = cfg["distributions"]

    rows: List[Dict[str, Any]] = []

    for _ in range(county.rows):
        dob = rand_date(rng, 1930, 2008)
        moved_in = rand_date(rng, min_year, max_year)
        move_out = rand_date(rng, moved_in.year, max_year) if prob(rng, 0.22) else None
        death = rand_date(rng, 2000, max_year) if prob(rng, 0.08) else None

        emp_status = pick_from_dist(dist["employment_status"], rng)
        has_job = compute_has_job(emp_status, rng)

        # Income stored as bracket in Schema B
        inc = income_usd(rng)
        bracket = income_bracket_from_income(inc)

        active_flag = "Y"
        if move_out is not None or death is not None:
            active_flag = "N"

        full_addr = f"{rand_street(rng)}, {rand_city(rng)}, WA {rand_zip(rng)}"

        row = {
            "person_uuid": str(uuid.uuid4()),
            "county": county.name,  # Schema B stores county name, not fips
            "dob_str": dob.isoformat(),  # often ISO, but bad data may change it
            "gender_code": rng.choice(GENDER_CODES_B),
            "race_code": rng.choice(RACE_CODES_B),
            "eth_flag": rng.choice(ETH_FLAGS_B),
            "income_bracket": bracket,
            "household_size": str(rng.randint(1, 6)),
            "full_address": full_addr,
            "updated_ts_utc": rand_timestamp_utc(rng, min_year, max_year),
            "moved_in_str": format_date_multiple(moved_in, rng) or "",
            "move_out_str": format_date_multiple(move_out, rng) or "",
            "death_str": format_date_multiple(death, rng) or "",
            "active_flag": active_flag,
            "last_verified_str": format_date_multiple(rand_date(rng, 2020, max_year), rng) or "",
            "verification_source": rng.choice(VERIFICATION_SOURCES),
            "employment_status": emp_status,
            "has_job_flag": "Y" if has_job else "N"
        }
        rows.append(row)

    return rows


def build_schema_c(county: CountySpec, cfg: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """
    Build Schema C table citizen(doc_id, county_fips, profile_json, created_at_utc).
    """
    min_year = cfg["date_bounds"]["min_year"]
    max_year = cfg["date_bounds"]["max_year"]
    dist = cfg["distributions"]

    rows: List[Dict[str, Any]] = []

    for _ in range(county.rows):
        dob = rand_date(rng, 1930, 2008)
        moved_in = rand_date(rng, min_year, max_year)
        move_out = rand_date(rng, moved_in.year, max_year) if prob(rng, 0.22) else None
        death = rand_date(rng, 2000, max_year) if prob(rng, 0.08) else None

        emp_status = pick_from_dist(dist["employment_status"], rng)
        has_job = compute_has_job(emp_status, rng)
        inc = income_usd(rng)

        active_status = True
        if move_out is not None or death is not None:
            active_status = False

        profile = {
            "name": {"first": rng.choice(FIRST_NAMES), "last": rng.choice(LAST_NAMES)},
            "demographics": {
                "dob": dob.isoformat(),
                "sex": pick_from_dist(dist["sex"], rng),
                "race": pick_from_dist(dist["race"], rng),
                "ethnicity": pick_from_dist(dist["ethnicity"], rng)
            },
            "household": {
                "hh_id": str(uuid.uuid4()),
                "hh_size": rng.randint(1, 6),
                "income_usd": inc
            },
            "location": {
                "street": rand_street(rng),
                "city": rand_city(rng),
                "state": "WA",
                "zip": rand_zip(rng)
            },
            "residency": {
                "active_status": active_status,
                "moved_in_date": moved_in.isoformat(),
                "move_out_date": move_out.isoformat() if move_out else None,
                "death_date": death.isoformat() if death else None,
                "last_verified_date": rand_date(rng, 2020, max_year).isoformat(),
                "verification_source": rng.choice(VERIFICATION_SOURCES)
            },
            "employment": {
                "employment_status": emp_status,
                "has_job": has_job
            }
        }

        row = {
            "doc_id": str(uuid.uuid4()),
            "county_fips": county.fips,
            "profile_json": json.dumps(profile, ensure_ascii=False),
            "created_at_utc": rand_timestamp_utc(rng, min_year, max_year)
        }
        rows.append(row)

    return rows


# -------------------------
# Main orchestration
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic CSV datasets for schemas A/B/C.")
    p.add_argument("--config", required=True, help="Path to generator_config.json")
    p.add_argument("--out", default=None, help="Override output_root")
    p.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    p.add_argument("--bad-rate", type=float, default=None, help="Override bad_data_rate (0..1)")
    return p.parse_args()


def maybe_corrupt_rows_schema_a(
    residents: List[Dict[str, Any]],
    households: List[Dict[str, Any]],
    addresses: List[Dict[str, Any]],
    bad_rate: float,
    weights: Dict[str, float],
    rng: random.Random
) -> None:
    """Inject bad data into Schema A by corrupting residents mostly, occasionally household/address."""
    for r in residents:
        if prob(rng, bad_rate):
            bad_type = weighted_choice(weights, rng)
            inject_bad_data_row(r, "A", bad_type, rng)

    # Small chance to corrupt household/address too.
    for h in households:
        if prob(rng, bad_rate * 0.15):
            bad_type = weighted_choice(weights, rng)
            inject_bad_data_row(h, "A", bad_type, rng)

    for a in addresses:
        if prob(rng, bad_rate * 0.15):
            bad_type = weighted_choice(weights, rng)
            inject_bad_data_row(a, "A", bad_type, rng)


def maybe_corrupt_rows_schema_b(
    rows: List[Dict[str, Any]],
    bad_rate: float,
    weights: Dict[str, float],
    rng: random.Random
) -> None:
    """Inject bad data into Schema B rows."""
    for row in rows:
        if prob(rng, bad_rate):
            bad_type = weighted_choice(weights, rng)
            inject_bad_data_row(row, "B", bad_type, rng)


def maybe_corrupt_rows_schema_c(
    rows: List[Dict[str, Any]],
    bad_rate: float,
    weights: Dict[str, float],
    rng: random.Random
) -> None:
    """Inject bad data into Schema C rows (JSON profile + sometimes created_at_utc)."""
    for row in rows:
        if prob(rng, bad_rate):
            bad_type = weighted_choice(weights, rng)
            # Profile JSON corruption
            try:
                profile = json.loads(row["profile_json"])
                inject_bad_data_into_profile(profile, bad_type, rng)
                row["profile_json"] = json.dumps(profile, ensure_ascii=False)
            except Exception:
                # If JSON already malformed, leave it; worker should treat as parse failure.
                row["profile_json"] = row["profile_json"] + "}"

        # Timestamp stale injection can be separate
        if prob(rng, bad_rate * 0.10):
            row["created_at_utc"] = "1990-01-01 00:00:00+00:00"


def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)

    output_root = args.out if args.out is not None else cfg["output_root"]
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    bad_rate = args.bad_rate if args.bad_rate is not None else float(cfg["bad_data_rate"])

    if bad_rate < 0.0 or bad_rate > 1.0:
        raise ValueError("bad_data_rate must be in [0,1]")

    rng = random.Random(seed)

    weights = cfg["bad_data_weights"]
    counties_cfg = cfg["counties"]
    counties = [CountySpec(**c) for c in counties_cfg]

    ensure_dir(output_root)

    for county in counties:
        # Schema A
        if cfg.get("generate_schema_a", True):
            out_dir = os.path.join(output_root, "schema_A", county.label)
            ensure_dir(out_dir)

            addresses, households, residents = build_schema_a(county, cfg, rng)
            maybe_corrupt_rows_schema_a(residents, households, addresses, bad_rate, weights, rng)

            write_csv(
                os.path.join(out_dir, "address.csv"),
                ["address_id", "street", "city", "state", "zip"],
                addresses
            )
            write_csv(
                os.path.join(out_dir, "household.csv"),
                ["household_id", "county_fips", "hh_size", "income_usd", "housing_type", "created_at_utc"],
                households
            )
            write_csv(
                os.path.join(out_dir, "resident.csv"),
                [
                    "resident_id", "household_id", "first_name", "last_name", "dob",
                    "sex", "race", "ethnicity", "address_id",
                    "moved_in_date", "move_out_date", "death_date", "active_status",
                    "last_verified_date", "verification_source",
                    "employment_status", "has_job_flag"
                ],
                residents
            )

        # Schema B
        if cfg.get("generate_schema_b", True):
            out_dir = os.path.join(output_root, "schema_B", county.label)
            ensure_dir(out_dir)

            rows = build_schema_b(county, cfg, rng)
            maybe_corrupt_rows_schema_b(rows, bad_rate, weights, rng)

            write_csv(
                os.path.join(out_dir, "personal_record.csv"),
                [
                    "person_uuid", "county", "dob_str", "gender_code", "race_code", "eth_flag",
                    "income_bracket", "household_size", "full_address", "updated_ts_utc",
                    "moved_in_str", "move_out_str", "death_str", "active_flag",
                    "last_verified_str", "verification_source", "employment_status", "has_job_flag"
                ],
                rows
            )

        # Schema C
        if cfg.get("generate_schema_c", True):
            out_dir = os.path.join(output_root, "schema_C", county.label)
            ensure_dir(out_dir)

            rows = build_schema_c(county, cfg, rng)
            maybe_corrupt_rows_schema_c(rows, bad_rate, weights, rng)

            write_csv(
                os.path.join(out_dir, "citizen.csv"),
                ["doc_id", "county_fips", "profile_json", "created_at_utc"],
                rows
            )

    print(f"Done. Output written to: {output_root}")


if __name__ == "__main__":
    main()