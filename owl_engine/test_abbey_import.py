"""
Quick verification that Abbey Road dashboard module can be imported
"""

print("Testing Abbey Road dashboard integration...")
print()

# Test 1: Import cv2
try:
    import cv2
    print(f"✓ OpenCV (cv2) version: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")
    exit(1)

# Test 2: Import abbey_road_dashboard
try:
    from abbey_road_dashboard import display_abbey_road_analytics
    print("✓ abbey_road_dashboard module imported")
except ImportError as e:
    print(f"✗ abbey_road_dashboard import failed: {e}")
    exit(1)

# Test 3: Import DataManager
try:
    from data_manager import DataManager
    print("✓ DataManager imported")
except ImportError as e:
    print(f"✗ DataManager import failed: {e}")
    exit(1)

# Test 4: Create instances
try:
    dm = DataManager()
    print("✓ DataManager instance created")
except Exception as e:
    print(f"✗ DataManager creation failed: {e}")
    exit(1)

print()
print("✅ All Abbey Road dashboard components ready!")
print()
print("To run the Palantir dashboard with Abbey Road analytics:")
print("  cd owl_engine")
print("  streamlit run palantir_dashboard.py")
