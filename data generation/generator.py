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