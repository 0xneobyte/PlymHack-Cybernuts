"""
Social Media Threat Intelligence
Monitor Twitter/X, Reddit, local forums for citizen-reported incidents
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List
import re
from collections import Counter

logger = logging.getLogger('SOCIAL_MONITOR')


class SocialThreatMonitor:
    """
    Monitor social media for emerging threats
    Uses keyword detection, geo-tagging, and velocity analysis
    """
    
    def __init__(self, db=None):
        self.db = db
        
        # Threat keyword patterns
        self.threat_patterns = {
            'fire': {
                'keywords': ['fire', 'smoke', 'burning', 'flames', 'blaze', 'inferno'],
                'severity': 'high',
                'category': 'emergency'
            },
            'flood': {
                'keywords': ['flood', 'flooding', 'water rising', 'submerged', 'waterlogged'],
                'severity': 'high',
                'category': 'environmental'
            },
            'accident': {
                'keywords': ['crash', 'accident', 'collision', 'hit', 'injured', 'ambulance'],
                'severity': 'medium',
                'category': 'public_safety'
            },
            'chemical': {
                'keywords': ['gas leak', 'chemical', 'fumes', 'toxic', 'smell', 'odor'],
                'severity': 'high',
                'category': 'hazmat'
            },
            'infrastructure': {
                'keywords': ['power out', 'blackout', 'water main', 'gas main', 'outage'],
                'severity': 'medium',
                'category': 'infrastructure'
            },
            'public_safety': {
                'keywords': ['riot', 'protest', 'violence', 'fight', 'stabbing', 'shooting'],
                'severity': 'high',
                'category': 'public_safety'
            },
            'traffic': {
                'keywords': ['traffic jam', 'gridlock', 'standstill', 'road closed', 'blocked'],
                'severity': 'low',
                'category': 'traffic'
            }
        }
        
        # London areas for geo-filtering
        self.london_areas = [
            'Westminster', 'Camden', 'Islington', 'Hackney', 'Tower Hamlets',
            'Greenwich', 'Lewisham', 'Southwark', 'Lambeth', 'Wandsworth',
            'Hammersmith', 'Kensington', 'Chelsea', 'Richmond', 'Kingston',
            'Merton', 'Sutton', 'Croydon', 'Bromley', 'Bexley', 'Havering',
            'Barking', 'Redbridge', 'Newham', 'Waltham Forest', 'Haringey',
            'Enfield', 'Barnet', 'Harrow', 'Brent', 'Ealing', 'Hounslow',
            'Hillingdon', 'London', 'Central London'
        ]
    
    def collect(self):
        """Main collection method"""
        logger.info("📱 Collecting social media threat intelligence...")
        
        try:
            # In production, this would call Twitter API, Reddit API, etc.
            # For now, we'll create a simulated collector that demonstrates structure
            
            simulated_posts = self._simulate_social_posts()
            threats = self._analyze_posts(simulated_posts)
            
            if self.db:
                for threat in threats:
                    self.db.store(
                        source_type='social_threat',
                        data=threat,
                        initial_confidence=threat.get('confidence', 0.4)
                    )
            
            logger.info(f"✓ Social: Analyzed {len(simulated_posts)} posts, detected {len(threats)} potential threats")
            
        except Exception as e:
            logger.error(f"Social monitoring error: {e}")
    
    def _simulate_social_posts(self) -> List[Dict]:
        """
        Simulate social media posts for demonstration
        In production: Replace with actual Twitter API v2 / Reddit API calls
        """
        # Example posts that would come from API
        simulated = [
            {
                'text': 'Massive traffic jam on A4 near Hammersmith, water everywhere! Looks like flooding',
                'location': 'Hammersmith',
                'coordinates': {'lat': 51.4927, 'lon': -0.2339},
                'timestamp': datetime.now().isoformat(),
                'user_verified': False,
                'has_media': True,
                'retweets': 12
            },
            {
                'text': 'Fire truck racing down Camden High Street, smoke visible',
                'location': 'Camden',
                'coordinates': {'lat': 51.5390, 'lon': -0.1426},
                'timestamp': datetime.now().isoformat(),
                'user_verified': True,
                'has_media': True,
                'retweets': 45
            },
            {
                'text': 'Power outage in Greenwich, all traffic lights out. Be careful everyone!',
                'location': 'Greenwich',
                'coordinates': {'lat': 51.4825, 'lon': -0.0077},
                'timestamp': datetime.now().isoformat(),
                'user_verified': False,
                'has_media': False,
                'retweets': 3
            }
        ]
        
        return simulated
    
    def _analyze_posts(self, posts: List[Dict]) -> List[Dict]:
        """
        Analyze posts for threat detection
        Returns high-confidence threats only
        """
        threats = []
        
        for post in posts:
            text_lower = post['text'].lower()
            
            # Check each threat pattern
            for threat_type, pattern in self.threat_patterns.items():
                # Check if any keywords match
                matches = [kw for kw in pattern['keywords'] if kw in text_lower]
                
                if matches:
                    # Calculate confidence score
                    confidence = self._calculate_confidence(post, matches, pattern)
                    
                    if confidence >= 0.3:  # Threshold for storage
                        threat = {
                            'timestamp': post['timestamp'],
                            'threat_type': threat_type,
                            'category': pattern['category'],
                            'severity': pattern['severity'],
                            'text': post['text'],
                            'location': post.get('location', 'Unknown'),
                            'coordinates': post.get('coordinates'),
                            'lat': post.get('coordinates', {}).get('lat'),
                            'lon': post.get('coordinates', {}).get('lon'),
                            'matched_keywords': matches,
                            'confidence': confidence,
                            'domain': 'social',
                            'source': 'social_media',
                            'source_type': 'social',
                            'engagement': post.get('retweets', 0),
                            'has_media': post.get('has_media', False),
                            'user_verified': post.get('user_verified', False)
                        }
                        threats.append(threat)
        
        return threats
    
    def _calculate_confidence(self, post: Dict, matches: List[str], pattern: Dict) -> float:
        """
        Calculate confidence score for a social threat
        
        Confidence factors:
        - Verified account: +0.3
        - Has photo/video: +0.4
        - Multiple keywords: +0.2
        - High engagement: +0.2
        - Geo-tagged: +0.2
        
        Base score: 0.2 (unverified text-only post)
        Max score: 1.0
        """
        confidence = 0.2  # Base
        
        # Verified user boost
        if post.get('user_verified'):
            confidence += 0.3
        
        # Media evidence
        if post.get('has_media'):
            confidence += 0.4
        
        # Multiple keyword matches
        if len(matches) > 1:
            confidence += 0.2
        
        # Engagement (virality indicator)
        if post.get('retweets', 0) > 10:
            confidence += 0.2
        
        # Geo-tagged location
        if post.get('coordinates'):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def detect_threat_clusters(self, threats: List[Dict], time_window_minutes: int = 30) -> List[Dict]:
        """
        Find clusters of similar reports (confirmation through multiple sources)
        """
        # Group by location + threat_type + time window
        clusters = []
        
        # This would implement clustering logic
        # If 3+ reports of same threat type in same area within 30 min → HIGH confidence
        
        return clusters


# Twitter API Integration (placeholder for actual implementation)
class TwitterThreatCollector:
    """
    Production Twitter/X API integration
    Requires Twitter API v2 credentials
    """
    
    def __init__(self, bearer_token: str = None):
        self.bearer_token = bearer_token
        # import tweepy
        # self.client = tweepy.Client(bearer_token=bearer_token)
    
    def stream_geo_threats(self, bounding_box: tuple, keywords: List[str]):
        """
        Stream tweets with geo-tagging in London bounding box
        
        bounding_box: (sw_lon, sw_lat, ne_lon, ne_lat)
        Example London: (-0.5103, 51.2868, 0.3340, 51.6918)
        """
        # Implementation with tweepy StreamingClient
        pass
    
    def search_recent_threats(self, query: str, max_results: int = 100):
        """
        Search recent tweets matching threat keywords
        """
        # Use Twitter API v2 search_recent_tweets
        pass


if __name__ == '__main__':
    # Test the monitor
    monitor = SocialThreatMonitor()
    monitor.collect()
