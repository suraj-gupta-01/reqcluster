import pandas as pd
import numpy as np
import re
import io
from typing import Tuple, List, Dict


REQUIRED_COLUMNS_VARIANTS = {
    "id": ["id", "req_id", "requirement_id", "req id", "identifier"],
    "text": ["text", "requirement", "description", "requirement text", "req text", "content", "shall"],
    "module": ["module", "subsystem", "category", "domain", "system"],
    "section": ["section", "chapter", "area", "group", "heading"],
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to expected schema."""
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {}
    for canonical, variants in REQUIRED_COLUMNS_VARIANTS.items():
        for col in df.columns:
            if col in variants and canonical not in df.columns:
                rename_map[col] = canonical
                break
    df = df.rename(columns=rename_map)
    # Ensure required columns exist
    for col in ["id", "text", "module", "section"]:
        if col not in df.columns:
            df[col] = None if col != "text" else ""
    return df


def clean_text(text: str) -> str:
    """Clean a single requirement text."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove leading/trailing quotes
    text = text.strip('"\'')
    return text


def preprocess_requirements(
    file_bytes: bytes, filename: str
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load and clean requirements from CSV or XLSX.
    Returns cleaned DataFrame and stats dict.
    """
    stats = {"total_raw": 0, "empty_removed": 0, "duplicate_removed": 0, "final": 0}

    # Load file
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    elif filename.endswith(".csv"):
        # Try UTF-8 first, fall back to latin-1; use on_bad_lines='skip' to
        # tolerate rows with unquoted embedded commas rather than crashing.
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding=enc,
                    quotechar='"',
                    on_bad_lines="skip",
                )
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not decode CSV file. Try saving as UTF-8.")
    else:
        raise ValueError(f"Unsupported file format: {filename}. Use CSV or XLSX.")

    stats["total_raw"] = len(df)

    # Normalize columns
    df = normalize_column_names(df)

    # Validate text column exists
    if "text" not in df.columns or df["text"].isnull().all():
        raise ValueError(
            "Could not find a requirements text column. "
            "Expected columns: id, text, module, section"
        )

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Remove empty requirements
    before = len(df)
    df = df[df["text"].str.len() > 5].copy()
    stats["empty_removed"] = before - len(df)

    # Remove duplicates (by text)
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).copy()
    stats["duplicate_removed"] = before - len(df)

    # Reset index
    df = df.reset_index(drop=True)

    # Generate IDs if missing
    df["id"] = df["id"].fillna(
        pd.Series([f"REQ-{i+1:03d}" for i in range(len(df))])
    )
    df["id"] = df["id"].astype(str)

    # Fill optional columns
    df["module"] = df["module"].fillna("").astype(str)
    df["section"] = df["section"].fillna("").astype(str)

    stats["final"] = len(df)
    return df, stats
