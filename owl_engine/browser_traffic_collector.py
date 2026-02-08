"""
Enhanced Traffic Browser Automation
Uses browser to collect real traffic data from Google Maps
"""

import logging
from datetime import datetime
import time
import json
from pathlib import Path

# Will attempt to use Selenium if installed
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠ Selenium not installed. Install with: pip install selenium webdriver-manager")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrafficBrowser')


class TrafficBrowserCollector:
    """Browser automation for Google Maps traffic collection"""
    
    # London routes to monitor
    LONDON_ROUTES = [
        {"name": "Central to North", "origin": "Trafalgar Square, London", "destination": "Camden Town, London"},
        {"name": "Central to South", "origin": "Trafalgar Square, London", "destination": "Clapham Common, London"},
        {"name": "Central to East", "origin": "Trafalgar Square, London", "destination": "Stratford, London"},
        {"name": "Central to West", "origin": "Trafalgar Square, London", "destination": "Hammersmith, London"},
        {"name": "North Circular East", "origin": "Finchley, London", "destination": "Ilford, London"},
        {"name": "North Circular West", "origin": "Chiswick, London", "destination": "Wembley, London"},
        {"name": "South Circular East", "origin": "Woolwich, London", "destination": "Clapham, London"},
        {"name": "South Circular West", "origin": "Kew, London", "destination": "Dulwich, London"},
        {"name": "A1 North", "origin": "City of London", "destination": "Barnet, London"},
        {"name": "A2 Southeast", "origin": "City of London", "destination": "Greenwich, London"},
        {"name": "A4 West", "origin": "Kensington, London", "destination": "Heathrow Airport"},
        {"name": "A23 South", "origin": "Westminster, London", "destination": "Croydon, London"},
    ]
    
    def __init__(self, output_dir="traffic_data_collected"):
        """Initialize browser collector"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.driver = None
    
    def setup_browser(self):
        """Setup Chrome browser with automatic ChromeDriver installation"""
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium not available - cannot setup browser")
            return False
        
        try:
            chrome_options = Options()
            # Uncomment to run headless (without browser window)
            # chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Use webdriver-manager to automatically download ChromeDriver
            logger.info("🔧 Setting up ChromeDriver (auto-download if needed)...")
            service = Service(ChromeDriverManager().install())
            
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("✓ Browser initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to setup browser: {e}")
            logger.info("  Tip: Make sure Chrome browser is installed")
            return False
    
    def collect_route_traffic(self, route):
        """Collect traffic data for a single route"""
        if not self.driver:
            logger.error("Browser not initialized")
            return None
        
        try:
            # Create Google Maps URL
            import urllib.parse
            origin_encoded = urllib.parse.quote(route['origin'])
            dest_encoded = urllib.parse.quote(route['destination'])
            
            url = f"https://www.google.com/maps/dir/{origin_encoded}/{dest_encoded}/"
            
            logger.info(f"  → Loading: {route['name']}")
            self.driver.get(url)
            
            # Wait for page to load
            time.sleep(5)
            
            # Try to extract traffic information
            traffic_data = {
                'route_name': route['name'],
                'origin': route['origin'],
                'destination': route['destination'],
                'url': url,
                'timestamp': datetime.now().isoformat(),
                'status': 'collected'
            }
            
            # Try to find duration
            try:
                # Multiple selectors to try for duration
                selectors = [
                    "div.Fk3sm.fontHeadlineSmall",
                    "div[jstcache]",
                    "span.delay",
                ]
                
                for selector in selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for elem in elements:
                            text = elem.text
                            if 'min' in text.lower() or 'hour' in text.lower():
                                traffic_data['duration'] = text.strip()
                                logger.debug(f"    Duration: {text.strip()}")
                                break
                        if 'duration' in traffic_data:
                            break
                    except:
                        continue
            except Exception as e:
                logger.debug(f"    Could not extract duration: {e}")
            
            # Try to find distance
            try:
                distance_elem = self.driver.find_element(By.CSS_SELECTOR, "div.Fk3sm.fontBodyMedium")
                traffic_data['distance'] = distance_elem.text.strip()
                logger.debug(f"    Distance: {distance_elem.text.strip()}")
            except:
                pass
            
            # Get page title (often contains route summary)
            traffic_data['page_title'] = self.driver.title
            
            return traffic_data
            
        except Exception as e:
            logger.error(f"  ✗ Failed to collect {route['name']}: {e}")
            return None
    
    def collect_all_routes(self):
        """Collect traffic data for all London routes"""
        if not self.setup_browser():
            return []
        
        all_data = []
        
        try:
            for route in self.LONDON_ROUTES:
                data = self.collect_route_traffic(route)
                if data:
                    all_data.append(data)
                
                # Small delay between routes
                time.sleep(2)
            
            # Save collected data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"traffic_collection_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            logger.info(f"✓ Collected {len(all_data)} routes - saved to {output_file}")
            
        finally:
            if self.driver:
                self.driver.quit()
                logger.info("✓ Browser closed")
        
        return all_data
    
    def close(self):
        """Close browser"""
        if self.driver:
            self.driver.quit()


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║       Traffic Browser Automation - Google Maps            ║
    ╚═══════════════════════════════════════════════════════════╝
    
    This script will:
    1. Open Chrome browser
    2. Visit 12 major London routes on Google Maps
    3. Extract current traffic conditions
    4. Save data to traffic_data_collected/
    
    """)
    
    if not SELENIUM_AVAILABLE:
        print("""
    ⚠ SELENIUM NOT INSTALLED
    
    To use browser automation, install:
        pip install selenium
    
    Then download ChromeDriver:
        https://chromedriver.chromium.org/
    
    Or use the manual instructions:
        python traffic_automation.py instructions
        """)
        return
    
    collector = TrafficBrowserCollector()
    
    # Ask if continuous mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'continuous':
        print("Running in continuous mode (every 5 minutes)...")
        print("Press Ctrl+C to stop.")
        
        try:
            while True:
                collector.collect_all_routes()
                print(f"\n⏰ Waiting 5 minutes until next collection...")
                time.sleep(300)  # 5 minutes
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
    else:
        # Single run
        collector.collect_all_routes()


if __name__ == "__main__":
    main()
