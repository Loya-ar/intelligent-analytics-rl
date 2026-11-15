"""Download, verify and load the four public datasets used in the benchmark.

Raw files go to data/raw/ and cleaned copies to data/processed/ (both gitignored).
Every raw file is checked against a fixed size and SHA-256 before it is used.
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# name -> (url, bytes, sha256)
FILES = {
    "hillstrom.csv": (
        "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
        3_964_977, "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"),
    "telco_churn.csv": (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        970_457, "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"),
    "online_retail_ii.zip": (
        "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        45_622_418, "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"),
    "green_tripdata_2024-01.parquet": (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-01.parquet",
        1_362_284, "1512a953bed564ac68dcb622827424dfb4f68ad93c2a4572e50ef6ab0b09c1ad"),
    "taxi_zone_lookup.csv": (
        "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
        12_331, "1a99e105092230f8620f301edcca7f80d3080642ff404d28ed957d3fa222c8ed"),
}


@dataclass
class Dataset:
    """A cleaned table plus the column roles the analyses use."""
    name: str
    df: pd.DataFrame
    targets: list[str]                 # numeric outcome columns, most important first
    numeric: list[str]                 # numeric columns for correlation / clustering
    date: str | None = None
    region: str | None = None
    product: str | None = None
    segments: list[str] = field(default_factory=list)   # columns used to cut segment slices


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_all(verbose: bool = True) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, size, digest) in FILES.items():
        dest = RAW / name
        if not dest.exists():
            if verbose:
                print(f"Downloading {url}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
        got_size, got = dest.stat().st_size, sha256(dest)
        if got_size != size or got != digest:
            raise RuntimeError(f"Checksum mismatch for {name}: {got_size} bytes, sha256 {got}")
        if verbose:
            print(f"OK  {name} ({got_size:,} bytes, sha256 verified)")


def _cached(name: str, build) -> pd.DataFrame:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = build()
    df.to_parquet(path, index=False)
    return df


def _hillstrom() -> pd.DataFrame:
    df = pd.read_csv(RAW / "hillstrom.csv")
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})
    return df


def _telco() -> pd.DataFrame:
    df = pd.read_csv(RAW / "telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df.dropna(subset=["TotalCharges"]).drop(columns=["customerID"])


def _retail() -> pd.DataFrame:
    with zipfile.ZipFile(RAW / "online_retail_ii.zip") as z, z.open("online_retail_II.xlsx") as f:
        sheets = pd.read_excel(f, sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    df["revenue"] = df["Quantity"] * df["Price"]
    top = df["Country"].value_counts().index[:8]
    df["Country"] = df["Country"].where(df["Country"].isin(top), "Other")
    df["StockCode"] = df["StockCode"].astype(str)
    df["hour"] = df["InvoiceDate"].dt.hour
    return df[["InvoiceDate", "Country", "StockCode", "Quantity", "Price", "revenue", "hour"]]


def _taxi() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "green_tripdata_2024-01.parquet")
    zones = pd.read_csv(RAW / "taxi_zone_lookup.csv")[["LocationID", "Borough"]]
    df = df.merge(zones, left_on="PULocationID", right_on="LocationID", how="left")
    df = df[df["lpep_pickup_datetime"].between("2024-01-01", "2024-02-01")]
    df["trip_minutes"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    df = df[(df["trip_minutes"].between(1, 180)) & (df["trip_distance"].between(0.01, 100))
            & (df["fare_amount"] > 0)].copy()
    df["payment_type"] = df["payment_type"].map({1: "card", 2: "cash", 3: "no_charge", 4: "dispute"}).fillna("other")
    df["passenger_count"] = df["passenger_count"].fillna(1)
    return df[["lpep_pickup_datetime", "Borough", "payment_type", "fare_amount", "trip_distance",
               "tip_amount", "total_amount", "trip_minutes", "passenger_count"]].reset_index(drop=True)


def load_all() -> dict[str, Dataset]:
    """Load the four cleaned datasets (downloads and caches on first use)."""
    download_all(verbose=False)
    return {
        "hillstrom": Dataset(
            "hillstrom", _cached("hillstrom", _hillstrom),
            targets=["spend", "visit", "conversion", "history", "recency"],
            numeric=["recency", "history", "mens", "womens", "newbie", "visit", "conversion", "spend"],
            region="zip_code", product="segment",
            segments=["channel", "history_segment", "newbie"]),
        "telco": Dataset(
            "telco", _cached("telco", _telco),
            targets=["MonthlyCharges", "Churn", "tenure", "TotalCharges"],
            numeric=["MonthlyCharges", "Churn", "tenure", "TotalCharges", "SeniorCitizen"],
            product="InternetService",
            segments=["Contract", "PaymentMethod", "gender", "PaperlessBilling", "Partner"]),
        "retail": Dataset(
            "retail", _cached("retail", _retail),
            targets=["revenue", "Quantity", "Price"],
            numeric=["revenue", "Quantity", "Price", "hour"],
            date="InvoiceDate", region="Country", product="StockCode",
            segments=["Country"]),
        "taxi": Dataset(
            "taxi", _cached("taxi", _taxi),
            targets=["total_amount", "tip_amount", "trip_distance", "trip_minutes", "passenger_count"],
            numeric=["fare_amount", "trip_distance", "tip_amount", "total_amount", "trip_minutes",
                     "passenger_count"],
            date="lpep_pickup_datetime", region="Borough", product="payment_type",
            segments=["Borough", "payment_type"]),
    }


if __name__ == "__main__":
    download_all()
    for name, ds in load_all().items():
        print(name, ds.df.shape, np.round(ds.df[ds.targets].mean().to_dict().get(ds.targets[0]), 3))
