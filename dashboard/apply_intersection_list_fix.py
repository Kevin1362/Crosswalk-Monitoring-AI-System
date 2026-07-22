from pathlib import Path
import shutil
import sys

dashboard = Path(__file__).resolve().parent / "dashboard.py"

if not dashboard.exists():
    print("ERROR: Put this file in the same folder as dashboard.py.")
    sys.exit(1)

text = dashboard.read_text(encoding="utf-8")
backup = dashboard.with_name("dashboard_before_intersection_list_fix.py")
shutil.copy2(dashboard, backup)

catalog_source = 'INTERSECTION_CATALOG = [\n    "GuideKaro Winter Presentation Intersection",\n    "GuideKaro Presentation Crosswalk",\n    "King St & Victoria St",\n    "Homer Watson Blvd & Block Line Rd",\n    "Fairway Rd & Wilson Ave",\n    "Ottawa St & Fischer-Hallman Rd",\n    "University Ave & King St",\n    "King St & Frederick St",\n    "Weber St & Victoria St",\n    "Courtland Ave & Block Line Rd",\n    "Highland Rd & Westmount Rd",\n    "Hespeler Rd & Pinebush Rd",\n]\n'

if "INTERSECTION_CATALOG = [" not in text:
    marker = 'BASE_DIR = Path(__file__).resolve().parent\n'
    if marker not in text:
        print("ERROR: Could not find BASE_DIR in dashboard.py.")
        sys.exit(1)
    text = text.replace(marker, marker + "\n" + catalog_source, 1)

old = '            options=sorted(df["intersection"].dropna().astype(str).unique()),\n'
new = (
    '            options=sorted(\n'
    '                set(INTERSECTION_CATALOG)\n'
    '                | set(df["intersection"].dropna().astype(str).unique())\n'
    '            ),\n'
)

if old in text:
    text = text.replace(old, new, 1)
elif "set(INTERSECTION_CATALOG)" not in text:
    print("ERROR: Could not find the intersection dropdown options line.")
    sys.exit(1)

if "The full location catalogue stays visible" not in text:
    old_block = (
        '            ),\n'
        '        )\n'
        '        selected_statuses = st.multiselect(\n'
    )
    new_block = (
        '            ),\n'
        '        )\n'
        '        st.caption(\n'
        '            "The full location catalogue stays visible. Locations without current "\n'
        '            "database events will return no records when selected."\n'
        '        )\n'
        '        selected_statuses = st.multiselect(\n'
    )
    if old_block in text:
        text = text.replace(old_block, new_block, 1)

dashboard.write_text(text, encoding="utf-8")

updated = dashboard.read_text(encoding="utf-8")
checks = {
    "Intersection catalogue": "INTERSECTION_CATALOG = [" in updated,
    "Catalogue merged with database": "set(INTERSECTION_CATALOG)" in updated,
}

print("\nIntersection list fix")
print("---------------------")
for label, result in checks.items():
    print(f"{label}: {result}")

print(f"\nBackup created: {backup.name}")

if not all(checks.values()):
    print("\nERROR: The dashboard was not updated completely.")
    sys.exit(1)

print("\nSUCCESS: The complete intersection list is now available.")
print("Restart Streamlit with: python -m streamlit run dashboard.py")
