import os
import requests
from bs4 import BeautifulSoup
from data import urls


# List of URLs to scrape
def clean_html(html):
    """Remove HTML tags and extra whitespace."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    return " ".join(text.split())


def save_to_file(content, filename):
    """Save cleaned content to a file."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)


# Create a folder to save the files
if not os.path.exists("scraped_content"):
    os.makedirs("scraped_content")

# Scrape each URL and save the cleaned content
# Headers to mimic a request from a web browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Scrape each URL and save the cleaned content
for url in urls:
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        cleaned_content = clean_html(response.text)
        file_name = os.path.join("scraped_content", url.split("/")[-2] + ".txt")
        save_to_file(cleaned_content, file_name)
        print(f"Saved cleaned content from {url} to {file_name}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to scrape {url}: {e}")

print("Scraping completed.")
