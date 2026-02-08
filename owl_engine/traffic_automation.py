"""
Traffic Data Collection via Browser Automation
Automated Google Maps traffic monitoring for London routes

This script opens Chrome and collects traffic data every 5-10 minutes
"""

import time
import logging
from datetime import datetime
from pathlib import Path

# Import the traffic collector
from data_collection.traffic_monitor import TrafficMonitorCollector
from database.json_store import JSONDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('TrafficAutomation')


def extract_traffic_from_page():
    """
    This function will be called by browser automation
    to extract traffic data from the current Google Maps page
    
    Returns JavaScript to execute in the page
    """
    js_extractor = """
    // Extract traffic data from Google Maps
    function extractTrafficData() {
        const data = {
            timestamp: new Date().toISOString(),
            url: window.location.href
        };
        
        // Try to find duration element
        const durationElements = document.querySelectorAll('[jstcache]');
        for (let elem of durationElements) {
            const text = elem.innerText;
            if (text.includes('min') || text.includes('hour')) {
                data.duration = text;
                break;
            }
        }
        
        // Try to find distance
        const distanceElements = document.querySelectorAll('.Fk3sm.fontBodyMedium');
        if (distanceElements.length > 0) {
            data.distance = distanceElements[0].innerText;
        }
        
        // Check for traffic incidents/alerts
        const alertElements = document.querySelectorAll('[data-tooltip]');
        data.alerts = Array.from(alertElements).map(el => el.getAttribute('data-tooltip')).filter(Boolean);
        
        // Get route summary
        const summaryElements = document.querySelectorAll('.fontBodyLarge');
        if (summaryElements.length > 0) {
            data.route_summary = summaryElements[0].innerText;
        }
        
        return data;
    }
    
    return extractTrafficData();
    """
    
    return js_extractor


def create_browser_automation_instructions():
    """
    Create instructions for manual browser automation
    (Can be enhanced with Selenium/Playwright later)
    """
    db = JSONDatabase(base_path="owl_data")
    collector = TrafficMonitorCollector(db)
    routes = collector.LONDON_ROUTES
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║       Traffic Data Collection - Browser Automation        ║
    ╚═══════════════════════════════════════════════════════════╝
    
    INSTRUCTIONS:
    1. Open Chrome browser
    2. For each route below, visit the URL
    3. Wait for the map to load (5-10 seconds)
    4. The page will show current traffic conditions
    5. Repeat every 5-10 minutes for continuous monitoring
    
    ROUTES TO MONITOR:
    """)
    
    for i, route in enumerate(routes, 1):
        url = collector._create_maps_url(route['origin'], route['destination'])
        print(f"\n{i}. {route['name']}")
        print(f"   Origin: {route['origin']}")
        print(f"   Destination: {route['destination']}")
        print(f"   URL: {url}")
    
    print("""
    
    AUTOMATION TIP:
    Use the browser automation script below with Selenium or Playwright
    to automatically visit these URLs and extract traffic data.
    """)


def run_browser_automation_cycle():
    """
    Run one cycle of browser automation
    This is a placeholder for full Selenium/Playwright integration
    """
    logger.info("🚗 Starting traffic monitoring cycle...")
    
    db = JSONDatabase(base_path="owl_data")
    collector = TrafficMonitorCollector(db)
    
    # Get all routes
    routes = collector.LONDON_ROUTES
    
    logger.info(f"📍 Monitoring {len(routes)} routes across London")
    
    # Collect data for all routes
    collector.collect()
    
    logger.info("✓ Traffic monitoring cycle complete")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'instructions':
        create_browser_automation_instructions()
    else:
        # Run continuous monitoring
        print("""
        ╔═══════════════════════════════════════════════════════════╗
        ║          Traffic Monitoring - Continuous Mode             ║
        ╚═══════════════════════════════════════════════════════════╝
        
        Running traffic monitoring every 5 minutes...
        Press Ctrl+C to stop.
        """)
        
        try:
            while True:
                run_browser_automation_cycle()
                
                # Wait 5 minutes
                logger.info("⏰ Waiting 5 minutes until next cycle...")
                time.sleep(300)  # 5 minutes
                
        except KeyboardInterrupt:
            logger.info("🛑 Traffic monitoring stopped by user")
