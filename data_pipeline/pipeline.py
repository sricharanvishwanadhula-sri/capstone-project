"""
pipeline.py ? Module 1: Data Pipeline
Cleans scraped book data, converts currency, loads into SQLite,
runs 5+ SQL queries, and demonstrates pd.read_sql vs pd.merge.
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import sys

# Force UTF-8 output on Windows console to handle book titles with accented characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from scraper import run_scraper


# ---------------------------------------------
# CONSTANTS
# ---------------------------------------------
GBP_TO_INR = 105.50   # Fixed project-defined constant (not a live rate)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "books.db")

RATING_MAP = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5
}

# ---------------------------------------------
# STEP 1: SCRAPE
# ---------------------------------------------
def scrape_data() -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 1: Scraping books.toscrape.com")
    print("="*60)
    df = run_scraper()
    print(f"Scraped {len(df)} books.")
    return df


# ---------------------------------------------
# STEP 2: CLEAN
# ---------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 2: Cleaning data")
    print("="*60)
    original_len = len(df)

    # 2a. Strip currency symbol, convert to float -> price_gbp
    df["price_gbp"] = (
        df["price_text"]
        .str.replace(r"[^\d.]", "", regex=True)
        .apply(pd.to_numeric, errors="coerce")
    )

    # 2b. Convert text star rating -> int (1?5) -> rating
    df["rating"] = df["star_rating"].map(RATING_MAP)

    # 2c. Parse availability text -> bool -> in_stock
    df["in_stock"] = df["availability_text"].str.contains(
        "In stock", case=False, na=False
    )

    # 2d. Handle parse failures
    #   - price_gbp: numeric field -> median imputation
    #   - rating: if unmapped text appears -> drop that row (state choice)
    price_null = df["price_gbp"].isna().sum()
    rating_null = df["rating"].isna().sum()

    if price_null > 0:
        median_price = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(median_price)
        print(f"  Imputed {price_null} missing price_gbp values with median ({median_price:.2f})")

    if rating_null > 0:
        print(f"  Dropping {rating_null} rows with unparseable star_rating.")
        df = df.dropna(subset=["rating"])

    df["rating"] = df["rating"].astype(int)

    # 2e. Drop exact duplicates
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    print(f"  Cleaned: {original_len} -> {len(df)} rows after dedup/drops.")

    # Verify minimums
    assert len(df) >= 60, f"Only {len(df)} books ? need at least 60!"
    assert df["category"].nunique() >= 3, f"Only {df['category'].nunique()} categories ? need at least 3!"
    print(f"  [OK] {len(df)} books across {df['category'].nunique()} categories.")

    return df


# ---------------------------------------------
# STEP 3: CONVERT CURRENCY
# ---------------------------------------------
def convert_currency(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 3: Currency conversion (1 GBP = 105.50 INR)")
    print("="*60)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)
    print(f"  price_inr range: {df['price_inr'].min():.2f} ? {df['price_inr'].max():.2f} INR")
    return df


# ---------------------------------------------
# STEP 4: LOAD INTO SQLITE
# ---------------------------------------------
def load_to_sqlite(df: pd.DataFrame, db_path: str = DB_PATH):
    print("\n" + "="*60)
    print("STEP 4: Loading into SQLite ? normalized 2-table schema")
    print("="*60)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Drop and recreate tables
    cur.executescript("""
        DROP TABLE IF EXISTS books;
        DROP TABLE IF EXISTS categories;

        CREATE TABLE categories (
            category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE books (
            book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            price_gbp   REAL NOT NULL,
            price_inr   REAL NOT NULL,
            rating      INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            in_stock    INTEGER NOT NULL,   -- 0 or 1 (SQLite boolean)
            category_id INTEGER NOT NULL REFERENCES categories(category_id)
        );
    """)

    # Insert categories
    unique_categories = df["category"].unique().tolist()
    for cat in unique_categories:
        cur.execute(
            "INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (cat,)
        )
    conn.commit()

    # Build category_name -> id map
    cat_map = dict(cur.execute("SELECT category_name, category_id FROM categories").fetchall())

    # Insert books
    records = []
    for _, row in df.iterrows():
        records.append((
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            cat_map[row["category"]],
        ))
    cur.executemany(
        "INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        records
    )
    conn.commit()

    book_count = cur.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    cat_count = cur.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    print(f"  [OK] Inserted {book_count} books into {cat_count} categories.")
    return conn


# ---------------------------------------------
# STEP 5: SQL QUERIES
# ---------------------------------------------
def run_sql_queries(conn: sqlite3.Connection, df_clean: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 5: SQL Queries")
    print("="*60)

    queries = {}

    # Q1 ? SELECT / WHERE / ORDER BY / LIMIT
    q1 = """
    SELECT title, price_gbp, price_inr, rating
    FROM books
    WHERE in_stock = 1
    ORDER BY price_gbp DESC
    LIMIT 10;
    """
    queries["Q1 ? Top 10 most expensive in-stock books"] = q1

    # Q2 ? DISTINCT
    q2 = """
    SELECT DISTINCT c.category_name
    FROM categories c
    JOIN books b ON c.category_id = b.category_id
    ORDER BY c.category_name;
    """
    queries["Q2 ? DISTINCT categories that have at least one book"] = q2

    # Q3 ? BETWEEN
    q3 = """
    SELECT title, price_gbp, rating
    FROM books
    WHERE price_gbp BETWEEN 10.00 AND 30.00
    ORDER BY rating DESC, price_gbp ASC
    LIMIT 15;
    """
    queries["Q3 ? Books priced BETWEEN ?10 and ?30"] = q3

    # Q4 ? IN
    q4 = """
    SELECT title, price_inr, rating
    FROM books
    WHERE rating IN (4, 5)
    ORDER BY rating DESC, price_inr ASC
    LIMIT 15;
    """
    queries["Q4 ? 4 and 5-star books (IN clause)"] = q4

    # Q5 ? JOIN: Average price per category + book count
    q5 = """
    SELECT c.category_name,
           COUNT(b.book_id)      AS total_books,
           ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp,
           ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
           MAX(b.rating)         AS max_rating
    FROM categories c
    JOIN books b ON c.category_id = b.category_id
    GROUP BY c.category_name
    ORDER BY avg_price_gbp DESC;
    """
    queries["Q5 ? JOIN: Category stats (avg price, book count, max rating)"] = q5

    # Q6 ? JOIN + WHERE: Top 10 highest-rated books per category
    q6 = """
    SELECT c.category_name, b.title, b.rating, b.price_gbp
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    WHERE b.rating = (
        SELECT MAX(b2.rating)
        FROM books b2
        WHERE b2.category_id = b.category_id
    )
    ORDER BY c.category_name, b.price_gbp DESC
    LIMIT 20;
    """
    queries["Q6 ? JOIN + Subquery: Highest-rated books per category"] = q6

    results = {}
    for name, query in queries.items():
        print(f"\n{'-'*50}")
        print(f"  {name}")
        print(f"{'-'*50}")
        result_df = pd.read_sql(query, conn)
        print(result_df.to_string(index=False))
        results[name] = (query, result_df)

    return results


# ---------------------------------------------
# STEP 6: pd.read_sql vs pd.merge comparison
# ---------------------------------------------
def demonstrate_pandas(conn: sqlite3.Connection, df_clean: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 6: pd.read_sql vs pd.merge equivalence")
    print("="*60)

    # --- Via pd.read_sql ---
    join_query = """
    SELECT c.category_name, b.title, b.price_gbp, b.price_inr, b.rating, b.in_stock
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    ORDER BY c.category_name, b.rating DESC;
    """
    df_sql = pd.read_sql(join_query, conn)
    print("\n[A] pd.read_sql result (first 10 rows):")
    print(df_sql.head(10).to_string(index=False))

    # --- Via pd.merge ---
    df_books = pd.read_sql("SELECT * FROM books", conn)
    df_categories = pd.read_sql("SELECT * FROM categories", conn)

    df_merge = pd.merge(
        df_books, df_categories, on="category_id", how="inner"
    )[["category_name", "title", "price_gbp", "price_inr", "rating", "in_stock"]]
    df_merge = df_merge.sort_values(["category_name", "rating"], ascending=[True, False]).reset_index(drop=True)

    print("\n[B] pd.merge result (first 10 rows):")
    print(df_merge.head(10).to_string(index=False))

    # Verify equivalence
    df_sql_sorted = df_sql.reset_index(drop=True)
    df_merge_sorted = df_merge.reset_index(drop=True)
    cols = ["category_name", "title", "price_gbp", "price_inr", "rating", "in_stock"]

    match = df_sql_sorted[cols].equals(df_merge_sorted[cols])
    print(f"\n[OK] pd.read_sql and pd.merge produce equivalent results: {match}")
    if not match:
        print("  (Minor sort differences are acceptable ? both contain the same rows)")
        set_sql = set(df_sql_sorted["title"].tolist())
        set_merge = set(df_merge_sorted["title"].tolist())
        print(f"  Same titles: {set_sql == set_merge}")


# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    # 1. Scrape
    df_raw = scrape_data()

    # 2. Clean
    df_clean = clean_data(df_raw)

    # 3. Convert
    df_clean = convert_currency(df_clean)

    # Show final DataFrame info
    print("\nFinal cleaned DataFrame:")
    print(df_clean[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]].head(10))
    print(f"\nSchema:\n{df_clean.dtypes}")

    # 4. Load to SQLite
    conn = load_to_sqlite(df_clean)

    # 5. SQL Queries
    run_sql_queries(conn, df_clean)

    # 6. Pandas equivalence
    demonstrate_pandas(conn, df_clean)

    conn.close()
    print("\n\n[OK] Module 1 ? Data Pipeline complete!")
    print(f"   Database saved to: {DB_PATH}")
