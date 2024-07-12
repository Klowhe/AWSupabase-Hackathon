import os
import requests
from bs4 import BeautifulSoup
import re


def clean_text(text):

    clean = re.sub("<[^<]+?>", "", text)
    clean = re.sub("\s+", " ", clean).strip()
    return clean


def download_content(url, base_folder):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}")
        return

    soup = BeautifulSoup(response.text, "html.parser")

    # Create a filename from the URL
    filename = url.split("/")[-1]
    if not filename.endswith(".txt"):
        filename += ".txt"

    # Create the full path
    file_path = os.path.join(base_folder, filename)

    # Clean and save the content
    clean_content = clean_text(soup.get_text())
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(clean_content)

    print(f"Saved {url} to {file_path}")

    # Find all links and recursively download their content
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.startswith("http"):
            download_content(href, base_folder)
        elif href.startswith("/"):
            # Handle relative URLs
            base_url = "/".join(url.split("/")[:3])
            download_content(base_url + href, base_folder)


def main():
    base_url = "https://singaporelegaladvice.com/law-articles/buying-selling-property/"
    base_folder = "downloaded_content"

    os.makedirs(base_folder, exist_ok=True)

    download_content(base_url, base_folder)


if __name__ == "__main__":
    main()
