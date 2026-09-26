# Module 1 — Data Pipeline

## Overview
Scrapes book data from [books.toscrape.com](http://books.toscrape.com/), cleans it,
converts GBP → INR at the fixed rate of **1 GBP = 105.50 INR**, loads it into a normalized
SQLite database, and demonstrates SQL queries + pandas equivalence.

## Files
| File | Purpose |
|------|---------|
| `scraper.py` | Scrapes ≥60 books across ≥3 categories from books.toscrape.com |
| `pipeline.py` | Cleans data, converts currency, loads SQLite, runs 6 SQL queries, pd.merge demo |
| `books.db` | SQLite database (generated; also recreatable by running pipeline.py) |

## How to Run
```bash
cd data_pipeline
python pipeline.py
```

## Design Decisions

### Scraping Scope
Scraped the first 5 paginated pages of the "All Products" catalogue. Each page lists 20 books,
yielding up to 100 books. Each book's detail page is followed to extract the category from
the breadcrumb trail. This gives ≥60 books spanning many categories without bias.

### Parse Failure Handling
- **price_gbp**: `pd.to_numeric(errors='coerce')` converts any unparseable value to NaN.
  NaN values are imputed with the **median price** — median is preferred over mean because
  the price distribution is right-skewed.
- **rating**: Rows with an unrecognised text rating (not One…Five) are **dropped**.
  There is no meaningful numeric default for an unmapped word rating, and these rows
  represent genuine data quality issues.

### Currency Conversion
Fixed rate: **1 GBP = 105.50 INR** — a project-defined constant, not a live market rate.
No external API, no network call, no date reference. The rate is hard-coded in `pipeline.py`
and stated here as required.

### SQLite Schema (2-table PK/FK)
```sql
categories(category_id INTEGER PRIMARY KEY AUTOINCREMENT,
           category_name TEXT UNIQUE NOT NULL)

books(book_id   INTEGER PRIMARY KEY AUTOINCREMENT,
      title     TEXT NOT NULL,
      price_gbp REAL NOT NULL,
      price_inr REAL NOT NULL,
      rating    INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
      in_stock  INTEGER NOT NULL,
      category_id INTEGER NOT NULL REFERENCES categories(category_id))
```

### SQL Queries Demonstrated
| # | Clauses Used | Description |
|---|-------------|-------------|
| Q1 | SELECT, WHERE, ORDER BY, LIMIT | Top 10 most expensive in-stock books |
| Q2 | SELECT, DISTINCT, JOIN | All distinct categories with at least one book |
| Q3 | SELECT, WHERE, BETWEEN, ORDER BY | Books priced between £10 and £30 |
| Q4 | SELECT, WHERE, IN | 4 and 5-star books |
| Q5 | SELECT, JOIN, GROUP BY, AVG | Category stats: avg price, book count, max rating |
| Q6 | SELECT, JOIN, Subquery, LIMIT | Highest-rated books per category |
