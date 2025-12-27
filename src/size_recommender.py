"""
Size Recommendation Engine
Production-ready module for Sustainable Basics sizing optimization.

Author: Esra Eslem Savaş
Date: December 2024
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional


class SizeRecommender:
    """
    Production-ready size recommendation system.
    
    Uses K-Nearest Neighbors to recommend optimal clothing sizes based on
    body measurements (chest, shoulder, torso).
    
    Attributes:
        model_path (Path): Path to trained model pickle file
        engine: Loaded SizeRecommendationEngine from training
        
    Example:
        >>> recommender = SizeRecommender()
        >>> result = recommender.recommend(chest_cm=98, shoulder_cm=44, torso_cm=68)
        >>> print(f"Size: {result['size']}, Confidence: {result['confidence']:.0%}")
        Size: M, Confidence: 85%
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the recommender with a trained model.
        
        Args:
            model_path: Path to .pkl model file. If None, uses default path.
        """
        if model_path is None:
            # Default path relative to this file
            base_path = Path(__file__).parent.parent
            model_path = base_path / 'models' / 'size_recommendation_model.pkl'
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                f"Please run notebook 04 to generate the model."
            )
        
        with open(self.model_path, 'rb') as f:
            self.engine = pickle.load(f)
    
    def recommend(
        self, 
        chest_cm: float, 
        shoulder_cm: float, 
        torso_cm: float,
        return_alternatives: bool = True
    ) -> Dict:
        """
        Recommend a size based on body measurements.
        
        Args:
            chest_cm: Chest circumference in centimeters
            shoulder_cm: Shoulder width in centimeters
            torso_cm: Torso length in centimeters
            return_alternatives: Whether to suggest alternative sizes
            
        Returns:
            Dictionary containing:
                - size: Recommended size (e.g., 'M', 'L')
                - confidence: Confidence score (0-1)
                - chest_range: Acceptable chest range for this size
                - shoulder_range: Acceptable shoulder range
                - torso_range: Acceptable torso range
                - alternatives: List of alternative sizes (if confidence < 0.7)
                
        Example:
            >>> result = recommender.recommend(98, 44, 68)
            >>> print(result)
            {
                'size': 'M',
                'confidence': 0.85,
                'chest_range': '94-102',
                'shoulder_range': '42-46',
                'torso_range': '65-71'
            }
        """
        # Validate inputs
        self._validate_measurements(chest_cm, shoulder_cm, torso_cm)
        
        # Get recommendation from engine
        size, confidence = self.engine.recommend_size(
            chest_cm, shoulder_cm, torso_cm
        )
        
        # Get size details
        details = self.engine.get_size_details(size)
        
        # Build response
        response = {
            'size': size,
            'confidence': round(confidence, 2),
            'chest_range': details['chest_range'],
            'shoulder_range': details['shoulder_range'],
            'torso_range': details['torso_range']
        }
        
        # Add alternatives if confidence is low
        if return_alternatives and confidence < 0.7:
            response['alternatives'] = self._get_alternative_sizes(size)
            response['note'] = (
                'Medium confidence - consider trying alternative sizes'
            )
        
        return response
    
    def recommend_batch(
        self, 
        measurements: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Recommend sizes for multiple customers at once.
        
        Args:
            measurements: DataFrame with columns [chest_cm, shoulder_cm, torso_cm]
            
        Returns:
            DataFrame with original measurements plus recommended_size and confidence
            
        Example:
            >>> df = pd.DataFrame({
            ...     'chest_cm': [88, 98, 108],
            ...     'shoulder_cm': [40, 44, 48],
            ...     'torso_cm': [64, 68, 72]
            ... })
            >>> results = recommender.recommend_batch(df)
        """
        required_cols = ['chest_cm', 'shoulder_cm', 'torso_cm']
        if not all(col in measurements.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        results = measurements.copy()
        recommendations = []
        confidences = []
        
        for _, row in measurements.iterrows():
            result = self.recommend(
                row['chest_cm'], 
                row['shoulder_cm'], 
                row['torso_cm'],
                return_alternatives=False
            )
            recommendations.append(result['size'])
            confidences.append(result['confidence'])
        
        results['recommended_size'] = recommendations
        results['confidence'] = confidences
        
        return results
    
    def _validate_measurements(
        self, 
        chest: float, 
        shoulder: float, 
        torso: float
    ) -> None:
        """Validate that measurements are within reasonable ranges."""
        if not (60 <= chest <= 150):
            raise ValueError(
                f"Chest measurement {chest}cm is outside valid range (60-150cm)"
            )
        if not (30 <= shoulder <= 70):
            raise ValueError(
                f"Shoulder measurement {shoulder}cm is outside valid range (30-70cm)"
            )
        if not (50 <= torso <= 90):
            raise ValueError(
                f"Torso measurement {torso}cm is outside valid range (50-90cm)"
            )
    
    def _get_alternative_sizes(self, recommended_size: str) -> list:
        """Get adjacent sizes as alternatives."""
        size_order = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL']
        
        try:
            idx = size_order.index(recommended_size)
        except ValueError:
            return []
        
        alternatives = []
        if idx > 0:
            alternatives.append(size_order[idx - 1])
        if idx < len(size_order) - 1:
            alternatives.append(size_order[idx + 1])
        
        return alternatives
    
    def get_all_sizes(self) -> pd.DataFrame:
        """
        Get the complete size guide.
        
        Returns:
            DataFrame with all available sizes and their measurements
        """
        return self.engine.size_guide.copy()


# API-ready function for simple use cases
def recommend_size_api(
    chest_cm: float, 
    shoulder_cm: float, 
    torso_cm: float,
    model_path: Optional[str] = None
) -> Dict:
    """
    Standalone API function for size recommendation.
    
    This is a convenience wrapper around SizeRecommender for
    simple, stateless API calls.
    
    Args:
        chest_cm: Chest circumference in cm
        shoulder_cm: Shoulder width in cm
        torso_cm: Torso length in cm
        model_path: Optional path to model file
        
    Returns:
        Dictionary with recommendation details
        
    Example:
        >>> result = recommend_size_api(98, 44, 68)
        >>> print(f"Recommended: {result['size']}")
    """
    recommender = SizeRecommender(model_path)
    return recommender.recommend(chest_cm, shoulder_cm, torso_cm)


# Command-line interface
if __name__ == '__main__':
    import sys
    
    print("="*60)
    print("SUSTAINABLE SIZING - Size Recommendation System")
    print("="*60)
    
    if len(sys.argv) == 4:
        # Command-line usage
        try:
            chest = float(sys.argv[1])
            shoulder = float(sys.argv[2])
            torso = float(sys.argv[3])
            
            recommender = SizeRecommender()
            result = recommender.recommend(chest, shoulder, torso)
            
            print(f"\n📏 Input Measurements:")
            print(f"   Chest: {chest}cm")
            print(f"   Shoulder: {shoulder}cm")
            print(f"   Torso: {torso}cm")
            print(f"\n✅ Recommendation:")
            print(f"   Size: {result['size']}")
            print(f"   Confidence: {result['confidence']:.0%}")
            print(f"\n📐 Size Range:")
            print(f"   Chest: {result['chest_range']}cm")
            print(f"   Shoulder: {result['shoulder_range']}cm")
            print(f"   Torso: {result['torso_range']}cm")
            
            if 'alternatives' in result:
                print(f"\n💡 Consider also trying: {', '.join(result['alternatives'])}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        print("\nUsage:")
        print("  python size_recommender.py <chest_cm> <shoulder_cm> <torso_cm>")
        print("\nExample:")
        print("  python size_recommender.py 98 44 68")
        print("\nOR use in Python:")
        print("  from size_recommender import SizeRecommender")
        print("  recommender = SizeRecommender()")
        print("  result = recommender.recommend(98, 44, 68)")
        print("  print(result)")
        
        # Quick test
        print("\n" + "="*60)
        print("RUNNING TEST...")
        print("="*60)
        
        recommender = SizeRecommender()
        test_cases = [
            (88, 40, 64, "Small"),
            (98, 44, 68, "Medium"),
            (108, 48, 72, "Large"),
        ]
        
        for chest, shoulder, torso, label in test_cases:
            result = recommender.recommend(chest, shoulder, torso)
            print(f"\n{label} Customer: {chest}/{shoulder}/{torso}cm")
            print(f"  → Size {result['size']} ({result['confidence']:.0%} confidence)")
        
        print("\n" + "="*60)
