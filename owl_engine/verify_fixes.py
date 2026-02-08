"""Verify deprecation fixes"""
import re

files = ['palantir_dashboard.py', 'abbey_road_dashboard.py']

print("="*50)
print("Streamlit Deprecation Check")
print("="*50)

for filename in files:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    container_count = len(re.findall(r'use_container_width', content))
    column_count = len(re.findall(r'use_column_width', content))
    
    status = "✓" if (container_count == 0 and column_count == 0) else "⚠"
    
    print(f"\n{status} {filename}:")
    print(f"   use_container_width: {container_count}")
    print(f"   use_column_width: {column_count}")
    
    # Check for image keys (for update fix)
    if 'abbey_road' in filename:
        has_keys = bool(re.search(r'st\.image\([^)]+key=', content))
        print(f"   Image update keys: {'✓ Present' if has_keys else '❌ Missing'}")

print("\n" + "="*50)
print("Summary:")
total_issues = sum([
    len(re.findall(r'use_container_width|use_column_width', open(f, 'r', encoding='utf-8').read())) 
    for f in files
])
print(f"Total deprecated parameters: {total_issues}")
print("Status: " + ("✓ All fixed!" if total_issues == 0 else f"⚠ {total_issues} remaining"))
print("="*50)
