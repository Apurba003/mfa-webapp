# feature_optimization.py - Feature Engineering & Optimization

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureOptimization:
    """Feature engineering and optimization"""
    
    def __init__(self, n_components=10):
        """
        Initialize feature optimization
        
        Args:
            n_components: Number of PCA components
        """
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()
        print(f"✓ Feature Optimization Initialized (PCA components: {n_components})")
    
    def normalize_features(self, features):
        """
        Normalize features to 0-1 range
        
        Args:
            features: Feature vector or array
            
        Returns:
            np.ndarray: Normalized features
        """
        try:
            features = np.array(features).reshape(1, -1)
            normalized = self.normalizer.fit_transform(features)
            return normalized[0]
        
        except Exception as e:
            print(f"Error normalizing features: {e}")
            return np.array(features[0]) if len(features.shape) > 1 else features
    
    def standardize_features(self, features):
        """
        Standardize features (zero mean, unit variance)
        
        Args:
            features: Feature vector or array
            
        Returns:
            np.ndarray: Standardized features
        """
        try:
            features = np.array(features).reshape(1, -1)
            standardized = self.scaler.fit_transform(features)
            return standardized[0]
        
        except Exception as e:
            print(f"Error standardizing features: {e}")
            return np.array(features[0]) if len(features.shape) > 1 else features
    
    def apply_pca(self, features, fit=False):
        """
        Apply PCA dimensionality reduction
        
        Args:
            features: Feature array
            fit: Whether to fit PCA first
            
        Returns:
            np.ndarray: Reduced features
        """
        try:
            features = np.array(features)
            
            if fit:
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)
                self.pca.fit(features)
            
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            reduced = self.pca.transform(features)
            return reduced[0]
        
        except Exception as e:
            print(f"Error applying PCA: {e}")
            return np.array(features[0]) if len(features.shape) > 1 else features
    
    def calculate_statistics(self, features_list):
        """
        Calculate statistical features
        
        Args:
            features_list: List of feature vectors
            
        Returns:
            dict: Statistical features
        """
        try:
            features_array = np.array(features_list)
            
            stats = {
                'mean': np.mean(features_array, axis=0).tolist(),
                'std': np.std(features_array, axis=0).tolist(),
                'median': np.median(features_array, axis=0).tolist(),
                'min': np.min(features_array, axis=0).tolist(),
                'max': np.max(features_array, axis=0).tolist(),
                'variance': np.var(features_array, axis=0).tolist(),
                'skewness': self._calculate_skewness(features_array),
                'kurtosis': self._calculate_kurtosis(features_array)
            }
            
            return stats
        
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return {}
    
    def _calculate_skewness(self, features_array):
        """Calculate skewness"""
        try:
            from scipy.stats import skew
            return skew(features_array, axis=0).tolist()
        except:
            return [0] * features_array.shape[1]
    
    def _calculate_kurtosis(self, features_array):
        """Calculate kurtosis"""
        try:
            from scipy.stats import kurtosis
            return kurtosis(features_array, axis=0).tolist()
        except:
            return [0] * features_array.shape[1]
    
    def select_best_features(self, features_array, importances, n_features=10):
        """
        Select best features based on importance scores
        
        Args:
            features_array: Feature array
            importances: Importance scores for each feature
            n_features: Number of features to select
            
        Returns:
            np.ndarray: Selected features
        """
        try:
            importances = np.array(importances)
            indices = np.argsort(importances)[-n_features:]
            
            selected = features_array[:, indices]
            return selected, indices
        
        except Exception as e:
            print(f"Error selecting best features: {e}")
            return features_array, np.arange(features_array.shape[1])
    
    def apply_scaling(self, features, method='standard'):
        """
        Apply feature scaling
        
        Args:
            features: Feature vector
            method: 'standard' or 'minmax'
            
        Returns:
            np.ndarray: Scaled features
        """
        try:
            if method == 'standard':
                return self.standardize_features(features)
            elif method == 'minmax':
                return self.normalize_features(features)
            else:
                return np.array(features)
        
        except Exception as e:
            print(f"Error applying scaling: {e}")
            return np.array(features)