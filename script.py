import os
import time
import csv
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# --- Config ---
CSV_FILENAME = "scraped_articles_nannews.csv"
PROGRESS_FILENAME = "progress.txt"

# --- Progress Tracking ---
def read_progress():
    try:
        with open(PROGRESS_FILENAME, 'r') as f:
            page = f.read().strip()
            if page.isdigit():
                print(f"[Resume] Last completed page: {page}")
                return int(page)
        return 0
    except FileNotFoundError:
        print("[Resume] No progress file found. Starting fresh.")
        return 0

def save_progress(page_num):
    with open(PROGRESS_FILENAME, 'w') as f:
        f.write(str(page_num))
    print(f"[Progress Saved] Page {page_num} completed.\n")

# --- Login ---
def login(page):
    print("Logging in...")
    page.goto("https://nannews.com.ng/login/?redirect_to=https%3A%2F%2Fnannews.com.ng%2F")
    page.get_by_role("textbox", name="Username or Email Address").fill(os.getenv("NAN_USERNAME"))
    page.get_by_role("textbox", name="Password").fill(os.getenv("NAN_PASSWORD"))
    page.get_by_role("button", name="Log In").click()
    page.wait_for_url("https://nannews.com.ng/", timeout=60000)
    print("Login successful.\n")

# --- Scrape Article ---
# *** MODIFIED: Added 'category_from_listing' parameter ***
def scrape_article(article_page, url, date_from_listing, category_from_listing):
    try:
        article_page.goto(url, timeout=90000, wait_until="domcontentloaded")
        article_page.wait_for_selector(".inner, .entry-content, article", timeout=60000)
        soup = BeautifulSoup(article_page.content(), 'html.parser')

        container = soup.select_one('div.inner, div.entry-content, article')
        if not container:
            raise Exception("Missing article container")

        title_tag = container.select_one('h4.title, h1.entry-title, h1')
        author_tag = container.select_one('div.author span, span.author, a[rel="author"]')
        content_tag = container.select_one('div.content, div.entry-content')

        title = title_tag.get_text(strip=True) if title_tag else "No Title"
        author = author_tag.get_text(strip=True) if author_tag else "No Author"
        # Use the date and category passed from the listing page
        date = date_from_listing
        category = category_from_listing
        content = content_tag.get_text(strip=True, separator="\n") if content_tag else "No Content"

        # *** MODIFIED: Added category to the returned list ***
        return [title, author, date, category, content, url]
    except Exception as e:
        print(f"    ✗ Error scraping article: {e}")
        # *** MODIFIED: Added category to the error return list ***
        return ["SCRAPE FAILED", "N/A", date_from_listing, category_from_listing, str(e), url]

# --- Main Loop ---
def scrape_all_pages(playwright):
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    login(page)

    last_completed_page = read_progress()

    # Fast-forward to saved progress
    for i in range(last_completed_page):
        try:
            print(f"Skipping page {i + 1}...")
            current_page = page.locator(".dataTables_paginate .paginate_button.current")
            next_button = current_page.locator("xpath=following-sibling::a[1]")
            if next_button.is_visible():
                with page.expect_response(lambda r: "admin-ajax.php" in r.url, timeout=30000):
                    next_button.click()
                time.sleep(1.5)
            else:
                raise Exception("Next button not visible during fast-forward.")
        except Exception as e:
            print(f"Error during fast-forward: {e}")
            context.close()
            browser.close()
            return

    page_num = last_completed_page + 1

    while True:
        print(f"\n--- Scraping Listing Page {page_num} ---")
        try:
            page.wait_for_selector("table.posts-data-table tbody tr", timeout=30000)
            
            rows = page.locator("table.posts-data-table tbody tr").all()
            
            articles_to_scrape = []
            for row in rows:
                link_element = row.locator("td.col-title a")
                date_element = row.locator("td.col-date")
                # *** ADDED: Locator for the category cell ***
                category_element = row.locator("td.col-categories")

                if link_element.count() > 0 and date_element.count() > 0 and category_element.count() > 0:
                    article_url = link_element.get_attribute("href")
                    date_text = date_element.inner_text()
                    # *** ADDED: Get the category text from the listing table ***
                    category_text = category_element.inner_text()
                    
                    if article_url:
                        # *** MODIFIED: Add category to the dictionary ***
                        articles_to_scrape.append({"url": article_url, "date": date_text, "category": category_text})

            print(f"Found {len(articles_to_scrape)} articles on page {page_num}.")

            # Save articles for this page
            csv_exists = os.path.exists(CSV_FILENAME)
            with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # *** MODIFIED: Added 'Category' to the CSV header ***
                if not csv_exists and page_num == 1:
                    writer.writerow(['Title', 'Author', 'Date', 'Category', 'Content', 'URL'])

                for idx, article_info in enumerate(articles_to_scrape, start=1):
                    article_url = article_info["url"]
                    date_from_listing = article_info["date"]
                    # *** ADDED: Get category from the dictionary ***
                    category_from_listing = article_info["category"]
                    
                    print(f"  ({idx}/{len(articles_to_scrape)}) Scraping: {article_url}")
                    article_tab = context.new_page()
                    # *** MODIFIED: Pass the category to the scrape_article function ***
                    article_data = scrape_article(article_tab, article_url, date_from_listing, category_from_listing)
                    writer.writerow(article_data)
                    article_tab.close()
                    time.sleep(1)

            save_progress(page_num)

            # Move to next page
            current_page = page.locator(".dataTables_paginate .paginate_button.current")
            next_button = current_page.locator("xpath=following-sibling::a[1]")

            if next_button.is_visible():
                with page.expect_response(lambda r: "admin-ajax.php" in r.url, timeout=30000):
                    next_button.click()
                time.sleep(2)
                page_num += 1
            else:
                print("✅ All pages completed.")
                break

        except Exception as e:
            print(f"✗ Error on page {page_num}: {e}")
            break

    context.close()
    browser.close()

# --- Run ---
if __name__ == "__main__":
    os.environ["NAN_USERNAME"] = "Bayo.olawunmi@gmail.com"
    os.environ["NAN_PASSWORD"] = "Speedy123@"  # Replace with env or prompt in real use

    with sync_playwright() as p:
        scrape_all_pages(p)