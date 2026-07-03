import json
import re
from pathlib import Path

def load(p):
    text = Path(p).read_text(encoding="utf-8")
    text = re.sub(
        r'("name":\s*")([^"]*?)\s*\n\s*([^"]*?)(")',
        r"\1\2 \3\4",
        text,
        flags=re.M,
    )
    return json.loads(text)

uploaded = load(
    r"C:\Users\nived\.cursor\projects\c-Users-nived-OneDrive-Desktop-SHL-Assessment\uploads\shl_product_catalog-0.json"
)
local = load(r"c:\Users\nived\OneDrive\Desktop\SHL_Assessment\data\catalog_raw.json")
print("Uploaded count:", len(uploaded))
print("Local count:", len(local))
u_ids = {e["entity_id"] for e in uploaded}
l_ids = {e["entity_id"] for e in local}
print("Only in uploaded:", len(u_ids - l_ids))
print("Only in local:", len(l_ids - u_ids))
if u_ids != l_ids:
    for eid in sorted(u_ids ^ l_ids)[:5]:
        print("  diff id:", eid)
