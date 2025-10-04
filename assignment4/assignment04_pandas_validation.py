

import pandas as pd
import numpy as np
from functools import partial
from datetime import datetime


CSV_FILENAME = "example_data.csv"
CLEANED_FILENAME = "cleaned_data.csv"


def create_example_csv(filename=CSV_FILENAME):
    """Create a small example CSV with a mix of correct and incorrect types.
    Includes duplicates and missing values on purpose for the assignment.
    """
    data = [
        {"user_id": 1, "name": "Luca", "age": "25", "signup_date": "2025-01-15", "last_login": "2025-09-30 14:22", "city": "Rome",  "temp_celsius": "23.5", "active": "True",  "notes": None,       "score": "85"},
        {"user_id": 2, "name": "Maya", "age": "30", "signup_date": "2024-12-01", "last_login": "2025-10-02 09:10", "city": None,    "temp_celsius": "18",   "active": "False", "notes": "Loyal",     "score": "92"},
        {"user_id": 3, "name": "Jon",  "age": "not_available", "signup_date": "not a date", "last_login": "2025-09-29",        "city": "Berlin","temp_celsius": "20",   "active": "yes",   "notes": "",           "score": "88"},
        
        {"user_id": 4, "name": "Luca", "age": "25", "signup_date": "2025-01-15", "last_login": "2025-09-30 14:22", "city": "Rome",  "temp_celsius": "23.5", "active": "True",  "notes": None,       "score": "85"},
        {"user_id": 5, "name": "Nora", "age": "45", "signup_date": "2020-05-20", "last_login": None,                 "city": "Tunis","temp_celsius": None,   "active": "False", "notes": "prefers email", "score": "100"},
        {"user_id": 6, "name": "Omar", "age": np.nan,  "signup_date": "2025-07-01", "last_login": "2025-09-01 08:00", "city": None,    "temp_celsius": "15.0", "active": "no",    "notes": None,        "score": "n/a"},
        {"user_id": 7, "name": "Zara", "age": "28", "signup_date": "2024-02-14", "last_login": "2024-02-20",        "city": "Paris","temp_celsius": "19.0", "active": "True",  "notes": "VIP",        "score": "90"},
    ]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"STEP 1 - Created example CSV -> {filename} (rows: {len(df)})")



def demo_series():
    ser = pd.Series([100, 200, 300], index=["alpha", "beta", "gamma"], name="example_series")
    print("\nSTEP 2 - Custom Pandas Series (with custom index):")
    print(ser)
    return ser



def load_dataframe(filename=CSV_FILENAME):
    df = pd.read_csv(filename)
    print("\nSTEP 3 - Loaded DataFrame from CSV")
    return df



def inspect_dataframe(df):
    print("\nSTEP 4 - Inspecting DataFrame (dtypes, head, tail, describe):")
    print("\n-- dtypes --")
    print(df.dtypes)
    print("\n-- head() --")
    print(df.head())
    print("\n-- tail() --")
    print(df.tail())
    print("\n-- describe(include='all') --")
    print(df.describe(include='all'))



def slicing_examples(df):
    print("\nSTEP 5 - Slicing Examples:")
    print("\n- By row position (iloc[1:4]):")
    print(df.iloc[1:4])

    print("\n- By column name (['name','age','city']):")
    print(df[["name", "age", "city"]])

    
    flags = np.array([True if i % 2 == 0 else False for i in range(len(df))])
    print("\n- By boolean flags array (pattern: True for even index rows):")
    print(df[flags])

   
    ages_numeric = pd.to_numeric(df["age"], errors="coerce")
    print("\n- Ages after pd.to_numeric(errors='coerce'):")
    print(ages_numeric)

    print("\n- Filter by age range (25 <= age <= 40):")
    age_filter = (ages_numeric >= 25) & (ages_numeric <= 40)
    print(df[age_filter])



def duplicates_and_uniques(df):
    print("\nSTEP 6 - Duplicates and unique counts:")
    print("\n-- duplicated() on full rows --")
    print(df.duplicated())
    print("\n-- duplicated().sum() (number of duplicate rows detected) --")
    print(df.duplicated().sum())

    print("\n-- nunique() for each column --")
    print(df.nunique(dropna=False))

    print("\n-- drop_duplicates(subset=['name','signup_date']): --")
    deduped = df.drop_duplicates(subset=["name", "signup_date"]) 
    print(deduped)
    return deduped



def safe_type_conversions(df):
    print("\nSTEP 7 - Safe type conversions with pd.to_numeric and pd.to_datetime:")

    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["temp_celsius"] = pd.to_numeric(df["temp_celsius"], errors="coerce")
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["last_login"] = pd.to_datetime(df["last_login"], errors="coerce")

    
    df["active_bool"] = (
        df["active"].astype(str).str.lower().map({"true": True, "false": False, "yes": True, "no": False})
    )

    print("\n-- dtypes after safe conversions --")
    print(df.dtypes)

    return df



def apply_default_values(df):
    print("\nSTEP 8 - Setting defaults for missing data using .apply():")

    def default_city(x):
        if pd.isna(x) or str(x).strip() == "":
            return "Unknown"
        return x

    def default_notes(x):
        if pd.isna(x) or str(x).strip() == "":
            return "No notes"
        return x

    df = df.copy()
    df["city"] = df["city"].apply(default_city)
    df["notes"] = df["notes"].apply(default_notes)

    print(df[["user_id", "name", "city", "notes"]])
    return df




def convert_columns_and_print(df):
    """Function to be used in a .pipe() pipeline. Converts types and prints the dtypes.
    It returns the DataFrame to allow chaining.
    """
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["temp_celsius"] = pd.to_numeric(df["temp_celsius"], errors="coerce")
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["last_login"] = pd.to_datetime(df["last_login"], errors="coerce")

    print("\n[PIPE] dtypes after convert_columns_and_print:")
    print(df.dtypes)
    return df


def print_null_counts(df):
    print("\n[PIPE] Null counts in each column:")
    print(df.isnull().sum())
    return df


def filter_min_age(df, min_age):
    """Filter by minimum age (assumes age is numeric). To be used with partial in .pipe()."""
    return df[df["age"] >= min_age]



if __name__ == "__main__":
    
    create_example_csv()

    
    demo_series()

    
    df = load_dataframe()

    
    inspect_dataframe(df)

    
    slicing_examples(df)

    
    deduped = duplicates_and_uniques(df)

    
    df_safe = safe_type_conversions(df)

    
    df_defaults = apply_default_values(df_safe)

    
    df_piped = (
        df_defaults
        .pipe(convert_columns_and_print)
        .pipe(print_null_counts)
    )

    
    print("\nSTEP 9 - Using .pipe() with partial to filter by min_age = 25")
    df_filtered_by_age = df_piped.pipe(partial(filter_min_age, min_age=25))
    print(df_filtered_by_age)

    
    df_piped.to_csv(CLEANED_FILENAME, index=False)
    print(f"\nFinal cleaned CSV saved -> {CLEANED_FILENAME}")

    print("\n--- Assignment tasks covered. If you need the same file but with comments in French or step outputs saved to a separate TXT, tell me and I will add them.")
