"""Fix Streamlit deprecation warnings"""
import re

def fix_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace use_container_width=True with width="stretch"
    content = re.sub(r', use_container_width=True', r', width="stretch"', content)
    content = re.sub(r'use_container_width=True,', r'width="stretch",', content)
    content = re.sub(r'\(use_container_width=True\)', r'(width="stretch")', content)
    content = re.sub(r'use_container_width=True\)', r'width="stretch")', content)
    
    # Remove use_column_width=True from buttons (no direct replacement)
    content = re.sub(r', use_column_width=True', '', content)
    content = re.sub(r'use_column_width=True,', '', content)
    content = re.sub(r'\(use_column_width=True\)', '()', content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Fixed {filename}")

# Fix the files
fix_file('abbey_road_dashboard.py')
print("Done!")
