import os

files_to_check = [
    ('analytics/titanic.csv', 'Titanic offline fallback'),
    ('analytics/titanic_pipeline.joblib', 'Saved ML pipeline'),
    ('analytics/01_eda_executed.ipynb', 'Executed EDA notebook'),
    ('analytics/02_modeling_executed.ipynb', 'Executed modeling notebook'),
]

print("=== Analytics Output Check ===")
all_ok = True
for path, desc in files_to_check:
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = '[OK]' if exists and size > 100 else '[MISSING]'
    print(f"  {status} {desc}: {path} ({size:,} bytes)")
    if not exists: all_ok = False

print()
charts_dir = 'analytics/charts'
charts = sorted(os.listdir(charts_dir)) if os.path.exists(charts_dir) else []
print(f"=== Charts Generated: {len(charts)} ===")
for c in charts:
    cpath = os.path.join(charts_dir, c)
    print(f"  - {c} ({os.path.getsize(cpath):,} bytes)")

print()
print(f"All analytics outputs OK: {all_ok}")
