import requests
from bs4 import BeautifulSoup

def extract_website_text(url):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()
