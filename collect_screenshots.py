import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# -----------------------
# CONFIG
# -----------------------
PHISHING_URLS_FILE = "phishing_urls.csv"   # CSV with a column "url"
LEGIT_URLS_FILE = "legit_urls.csv"         # CSV with a column "url"

OUTPUT_DIR = "dataset_images"
PHISHING_DIR = os.path.join(OUTPUT_DIR, "phishing")
LEGIT_DIR = os.path.join(OUTPUT_DIR, "legit")

# Make folders if not exist
os.makedirs(PHISHING_DIR, exist_ok=True)
os.makedirs(LEGIT_DIR, exist_ok=True)

# -----------------------
# SELENIUM SETUP
# -----------------------
chrome_options = Options()
chrome_options.add_argument("--headless")  # run browser invisibly
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1280,1024")

driver = webdriver.Chrome(options=chrome_options)

def capture_screenshots(urls, save_dir, label):
    for i, url in enumerate(urls):
        try:
            print(f"[{label}] {i+1}/{len(urls)} → {url}")
            driver.get(url)
            time.sleep(3)  # wait for page load
            filename = os.path.join(save_dir, f"{label}_{i}.png")
            driver.save_screenshot(filename)
        except Exception as e:
            print(f"❌ Failed: {url} → {e}")

# -----------------------
# LOAD URL LISTS
# -----------------------
if os.path.exists(PHISHING_URLS_FILE):
    phishing_urls = pd.read_csv(PHISHING_URLS_FILE)["url"].dropna().tolist()
else:
    phishing_urls = ["http://examplephishingsite.com"]  # dummy

if os.path.exists(LEGIT_URLS_FILE):
    legit_urls = pd.read_csv(LEGIT_URLS_FILE)["url"].dropna().tolist()
else:
    legit_urls = ["https://google.com", "https://amazon.com"]

# -----------------------
# RUN
# -----------------------
capture_screenshots(phishing_urls, PHISHING_DIR, "phishing")
capture_screenshots(legit_urls, LEGIT_DIR, "legit")

driver.quit()
print("\n✅ Screenshot collection complete. Check dataset_images/ folder.")
