"""
OWL ENGINE - Layer 5: VECTORIZE (Embedding & Similarity Layer)

Purpose: Convert entities and events into vector embeddings for semantic similarity.
Enable fuzzy matching of locations and events that traditional string matching misses.

Philosophy: Meaning lives in vector space. Similar events cluster together.

Math:
- TF-IDF vectors for text (locations, descriptions)
- Cosine similarity for comparing vectors
- Dimensionality reduction via SVD/PCA for visualization
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from layer3_link import EntityGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('VECTORIZE_LAYER')


class VectorEngine:
    """Convert entities to vector embeddings"""
    
    def __init__(self, graph: EntityGraph):
        self.graph = graph
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.entity_vectors = {}
        self.entity_ids = []
        self.vector_matrix = None
        
    def vectorize_all_entities(self):
        """Create vector embeddings for all entities"""
        logger.info("🔢 Vectorizing entities...")
        
        # Prepare text corpus
        corpus = []
        entity_ids = []
        
        for entity_id, entity in self.graph.entities.items():
            text = self._entity_to_text(entity)
            corpus.append(text)
            entity_ids.append(entity_id)
        
        if not corpus:
            logger.warning("No entities to vectorize")
            return
        
        # Generate TF-IDF vectors
        self.vector_matrix = self.vectorizer.fit_transform(corpus)
        self.entity_ids = entity_ids
        
        # Store individual vectors
        for i, entity_id in enumerate(entity_ids):
            self.entity_vectors[entity_id] = self.vector_matrix[i]
        
        logger.info(f"✓ Vectorized {len(entity_ids)} entities")
        logger.info(f"✓ Vector dimensions: {self.vector_matrix.shape[1]}")
    
    def _entity_to_text(self, entity: Dict) -> str:
        """Convert entity to text representation for vectorization"""
        parts = []
        
        # Add type
        parts.append(entity.get('type', ''))
        
        # Add locations
        locations = entity.get('locations', [])
        parts.extend(locations)
        
        # Add data fields
        data = entity.get('data', {})
        for key, value in data.items():
            if isinstance(value, str):
                parts.append(value)
        
        return ' '.join(parts)
    
    def find_similar_entities(self, entity_id: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar entities using cosine similarity.
        
        Cosine Similarity:
        cos(θ) = (A · B) / (||A|| × ||B||)
        
        Returns: List of (entity_id, similarity_score) tuples
        """
        if entity_id not in self.entity_vectors:
            return []
        
        query_vector = self.entity_vectors[entity_id]
        
        # Compute cosine similarity with all entities
        similarities = cosine_similarity(query_vector, self.vector_matrix)[0]
        
        # Get top N (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
        
        results = []
        for idx in similar_indices:
            similar_id = self.entity_ids[idx]
            score = similarities[idx]
            results.append((similar_id, float(score)))
        
        return results
    
    def cluster_by_similarity(self, threshold: float = 0.3) -> Dict[str, List[str]]:
        """Group similar entities into clusters"""
        logger.info(f"🎯 Clustering entities (threshold={threshold})...")
        
        clusters = {}
        processed = set()
        
        for entity_id in self.entity_ids:
            if entity_id in processed:
                continue
            
            # Find similar entities
            similar = self.find_similar_entities(entity_id, top_n=20)
            
            # Create cluster of entities above threshold
            cluster_members = [entity_id]
            for sim_id, score in similar:
                if score >= threshold and sim_id not in processed:
                    cluster_members.append(sim_id)
                    processed.add(sim_id)
            
            processed.add(entity_id)
            
            if len(cluster_members) > 1:
                clusters[entity_id] = cluster_members
        
        logger.info(f"✓ Found {len(clusters)} clusters")
        return clusters
    
    def get_vector(self, entity_id: str) -> np.ndarray:
        """Get vector for an entity"""
        return self.entity_vectors.get(entity_id)


class SemanticMatcher:
    """Match entities using semantic similarity"""
    
    def __init__(self, vector_engine: VectorEngine):
        self.vector_engine = vector_engine
    
    def match_flood_to_traffic_routes(self, flood_id: str, threshold: float = 0.2) -> List[Tuple[str, float]]:
        """Find traffic routes semantically similar to a flood location"""
        similar = self.vector_engine.find_similar_entities(flood_id, top_n=50)
        
        # Filter for traffic routes only
        graph = self.vector_engine.graph
        traffic_matches = []
        
        for entity_id, score in similar:
            entity = graph.get_entity(entity_id)
            if entity and entity.get('type') == 'traffic_route' and score >= threshold:
                traffic_matches.append((entity_id, score))
        
        return traffic_matches


if __name__ == '__main__':
    from layer3_link import build_entity_graph
    
    logger.info("=" * 60)
    logger.info("OWL ENGINE - Layer 5: VECTORIZE")
    logger.info("=" * 60)
    
    # Build graph
    graph = build_entity_graph()
    
    # Create vector engine
    vector_engine = VectorEngine(graph)
    vector_engine.vectorize_all_entities()
    
    # Test similarity
    logger.info("\n🔍 Testing semantic similarity...")
    
    # Get first flood entity
    floods = [eid for eid, e in graph.entities.items() if e['type'] == 'flood_warning']
    if floods:
        test_flood = floods[0]
        logger.info(f"\nTest flood: {test_flood}")
        
        similar = vector_engine.find_similar_entities(test_flood, top_n=5)
        logger.info(f"\nMost similar entities:")
        for sim_id, score in similar:
            entity = graph.get_entity(sim_id)
            logger.info(f"  {score:.3f} - {entity['type']}: {entity.get('data', {}).get('description', sim_id)}")
    
    logger.info("\n✓ Vectorization complete")
