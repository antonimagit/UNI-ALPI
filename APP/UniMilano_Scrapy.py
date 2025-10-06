import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import csv
from pathlib import Path
import datetime
import re

def get_timestamp_str():
    now = datetime.datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")

# ================== CONFIGURAZIONE ==================
SAVE_TEXT = True
timestamp = get_timestamp_str()
URL_TO_SCRAP = "https://web.archive.org/web/20250401085050/https://www.unimi.it/it/studiare"
SITE_NAME = "UNIMILANO"
MAX_DEPTH = 2        # 0 = solo la root, 1 = root + link, 2 = anche i link dei link
RECURSIVE = True     # una sola volta qui
output_dir = Path("scraped_texts") / SITE_NAME / timestamp
output_file = output_dir / "scraped_pages.csv"

if SAVE_TEXT:
    output_dir.mkdir(parents=True, exist_ok=True)
# =====================================================

def clean_url_for_filename(url):
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', url)
    return cleaned[:100]

def get_page_content(url, attr_name=None, attr_value=None):
    try:
        print(f"[INFO] Processing: {url}")
        response = requests.get(url, timeout=5)
        content_type_header = response.headers.get("Content-Type", "").lower()
        content_type_main = content_type_header.split(";")[0].strip()

        if "text/html" in content_type_main:
            content_type = "html"
        elif "application/pdf" in content_type_main:
            content_type = "pdf"
        elif content_type_main:
            content_type = content_type_main
        else:
            content_type = "unknown"

        if content_type != "html":
            print(f"[SKIP] Not HTML: {url} ({content_type})")
            return "", [], content_type

        soup = BeautifulSoup(response.text, "html.parser")
        target_element = soup.find(attrs={attr_name: attr_value}) if attr_name and attr_value else None
        text_content = target_element.get_text(separator=" ", strip=True) if target_element else soup.get_text(separator=" ", strip=True)
        links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
        return text_content, links, content_type

    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch: {url} ({e})")
        return "", [], "error"

def save_text_content(url, content):
    cleaned_url = clean_url_for_filename(url).replace("https___", "")
    filename = f"{cleaned_url}_{timestamp}.txt"
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filename

def scrape(url, current_depth=0, visited=None,
           attr_name=None, attr_value=None, same_domain=True,
           writer=None, base_domain=None, base_url=None, original_domain=None):

    if visited is None:
        visited = set()
    if base_domain is None:
        base_domain = urlparse(url).netloc
    if base_url is None:
        base_url = url if url.endswith('/') else url + '/'

    if original_domain is None:
        parsed = urlparse(url)
        if "web.archive.org" in base_domain:
            parts = parsed.path.split('/', 3)
            if len(parts) > 3:
                original_url = parts[3]
                if original_url.startswith("http"):
                    original_domain = urlparse(original_url).netloc
        else:
            original_domain = base_domain

    # --- limite profondità (MAX_DEPTH è globale e incluso) ---
    if current_depth > MAX_DEPTH:
        return

    if url in visited:
        return
    visited.add(url)

    text_content, links, content_type = get_page_content(url, attr_name, attr_value)
    status = "OK" if content_type != "error" else "ERROR"

    filename = ""
    if SAVE_TEXT and text_content:
        filename = save_text_content(url, text_content)

    writer.writerow([url, content_type, status, current_depth, filename])

    # --- Ricorsione controllata dal flag globale ---
    if RECURSIVE and content_type == "html" and current_depth < MAX_DEPTH:
        next_depth = current_depth + 1
        for link in links:
            link_domain = urlparse(link).netloc

            if same_domain:
                if link_domain != base_domain:
                    print(f"[SKIP] External domain: {link}")
                    continue
                if original_domain and original_domain not in link:
                    print(f"[SKIP] Link non del dominio originale: {link}")
                    continue

            scrape(
                link,
                current_depth=next_depth,
                visited=visited,
                attr_name=attr_name,
                attr_value=attr_value,
                same_domain=same_domain,
                writer=writer,
                base_domain=base_domain,
                base_url=base_url,
                original_domain=original_domain
            )

if __name__ == "__main__":
    with open(output_file, mode="w", newline='', encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["URL", "Content-Type", "Status", "Depth", "Text File"])

        visited = set()

        scrape(
            url=URL_TO_SCRAP,
            current_depth=0,
            visited=visited,
            attr_name="itemprop",
            attr_value="articleBody",
            same_domain=True,
            writer=csv_writer
        )

    print(f"\n[INFO] Output written to: {output_file}")
    print(f"[INFO] Text files saved to: {output_dir.resolve()}")
