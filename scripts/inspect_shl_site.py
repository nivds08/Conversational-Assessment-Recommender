import re
from collections import Counter

import httpx

r = httpx.get("https://www.shl.com/products/product-catalog/", timeout=30, follow_redirects=True)
text = r.text
lower = text.lower()

print("Status:", r.status_code, "| Length:", len(text))
for term in [
    "solution",
    "simulation",
    "precise fit",
    "product type",
    "filter",
    "category",
    "bundle",
    "individual test",
    "pre-packaged",
    "job solution",
]:
    print(f"  {term!r}: {lower.count(term)}")

links = re.findall(r'href="([^"]*solution[^"]*)"', text, re.I)
print("\nSolution-related links (first 12):")
for link in links[:12]:
    print(" ", link)

# Check if Entry Level Cashier Solution page exists on site
for slug in [
    "entry-level-cashier-solution",
    "core-java-entry-level-new",
    "customer-service-phone-simulation",
]:
    url = f"https://www.shl.com/products/product-catalog/view/{slug}/"
    try:
        pr = httpx.head(url, timeout=15, follow_redirects=True)
        print(f"{slug}: {pr.status_code}")
    except Exception as e:
        print(f"{slug}: error {e}")
