"""
scraper.py ? Module 1: Data Pipeline
Scrapes books.toscrape.com across all catalogue pages (up to 5 pages minimum)
and multiple categories, returning a raw DataFrame with >= 60 books.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "http://books.toscrape.com/"
CATALOGUE_URL = "http://books.toscrape.com/catalogue/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_soup(url: str) -> BeautifulSoup:
    """Fetch a URL and return a BeautifulSoup object."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_categories() -> dict:
    """Return a mapping of category_name -> category_url."""
    soup = get_soup(BASE_URL)
    nav = soup.find("ul", class_="nav-list")
    categories = {}
    for li in nav.find_all("li")[1:]:  # skip "Books" root
        a = li.find("a")
        name = a.text.strip()
        url = BASE_URL + a["href"]
        categories[name] = url
    return categories


def parse_books_on_page(soup: BeautifulSoup, category: str) -> list[dict]:
    """Parse all book entries on a single listing page."""
    books = []
    for article in soup.find_all("article", class_="product_pod"):
        try:
            title = article.find("h3").find("a")["title"]
            price_text = article.find("p", class_="price_color").text.strip()
            rating_class = article.find("p", class_="star-rating")["class"][1]
            availability_text = article.find("p", class_="availability").text.strip()
            books.append({
                "title": title,
                "price_text": price_text,
                "star_rating": rating_class,
                "availability_text": availability_text,
                "category": category,
            })
        except Exception:
            # Skip malformed entries silently; logged in pipeline.py
            continue
    return books


def scrape_category(category_name: str, category_url: str, max_pages: int = 3) -> list[dict]:
    """Scrape up to max_pages pages of a single category."""
    all_books = []
    url = category_url
    for page_num in range(1, max_pages + 1):
        try:
            soup = get_soup(url)
            books = parse_books_on_page(soup, category_name)
            all_books.extend(books)

            # Check for next page
            next_btn = soup.find("li", class_="next")
            if not next_btn:
                break

            # Build next page URL relative to catalogue or base
            next_href = next_btn.find("a")["href"]
            if "catalogue/" in url:
                # category URL like .../catalogue/category/books/mystery_3/index.html
                url = "/".join(url.split("/")[:-1]) + "/" + next_href
            else:
                url = CATALOGUE_URL + next_href

            time.sleep(0.3)  # polite delay
        except Exception as e:
            print(f"  Warning: error on page {page_num} of '{category_name}': {e}")
            break
    return all_books


def scrape_all_products_pages(max_pages: int = 5) -> list[dict]:
    """Scrape the first max_pages pages of the 'All products' catalogue."""
    all_books = []
    url = CATALOGUE_URL + "page-1.html"
    for page_num in range(1, max_pages + 1):
        try:
            soup = get_soup(url)
            # Each book on the all-products page doesn't show category directly;
            # we follow the book detail link to get category.
            for article in soup.find_all("article", class_="product_pod"):
                try:
                    title = article.find("h3").find("a")["title"]
                    price_text = article.find("p", class_="price_color").text.strip()
                    rating_class = article.find("p", class_="star-rating")["class"][1]
                    availability_text = article.find("p", class_="availability").text.strip()

                    # Get category from detail page breadcrumb
                    detail_href = article.find("h3").find("a")["href"]
                    if detail_href.startswith("../"):
                        detail_url = CATALOGUE_URL + detail_href.replace("../", "")
                    else:
                        detail_url = CATALOGUE_URL + detail_href

                    detail_soup = get_soup(detail_url)
                    breadcrumb = detail_soup.find("ul", class_="breadcrumb")
                    crumbs = breadcrumb.find_all("li")
                    # breadcrumb: Home > Books > Category > Title
                    category = crumbs[2].text.strip() if len(crumbs) >= 3 else "Unknown"

                    all_books.append({
                        "title": title,
                        "price_text": price_text,
                        "star_rating": rating_class,
                        "availability_text": availability_text,
                        "category": category,
                    })
                    time.sleep(0.1)
                except Exception as e:
                    print(f"  Warning: skipping book on page {page_num}: {e}")
                    continue

            # Next page
            next_btn = soup.find("li", class_="next")
            if not next_btn:
                break
            next_href = next_btn.find("a")["href"]
            url = CATALOGUE_URL + next_href
            time.sleep(0.3)
            print(f"  Scraped page {page_num}, total so far: {len(all_books)}")
        except Exception as e:
            print(f"  Error on all-products page {page_num}: {e}")
            break
    return all_books


def run_scraper() -> pd.DataFrame:
    """
    Main scraping entry point.
    Strategy: scrape 5 all-products pages (each has 20 books = up to 100 books),
    following detail links for category. Falls back to per-category scraping if needed.
    Returns raw DataFrame.
    """
    print("=== Starting scraper: books.toscrape.com ===")
    print("Scraping first 5 pages of All Products catalogue...")
    raw_books = scrape_all_products_pages(max_pages=5)

    df = pd.DataFrame(raw_books)
    categories_found = df["category"].nunique() if not df.empty else 0
    print(f"\nRaw scrape complete: {len(df)} books across {categories_found} categories.")

    # If we somehow got fewer than 60 books, supplement with targeted category scraping
    if len(df) < 60:
        print("Fewer than 60 books; supplementing with category scraping...")
        categories = get_categories()
        extra_books = []
        for cat_name, cat_url in list(categories.items())[:5]:
            cat_books = scrape_category(cat_name, cat_url, max_pages=2)
            extra_books.extend(cat_books)
            print(f"  {cat_name}: {len(cat_books)} books")
            if len(df) + len(extra_books) >= 60:
                break
        df = pd.concat([df, pd.DataFrame(extra_books)], ignore_index=True)
        df = df.drop_duplicates(subset=["title"])

    print(f"Final raw dataset: {len(df)} rows, {df['category'].nunique()} categories.")
    return df


if __name__ == "__main__":
    df = run_scraper()
    print(df.head())
