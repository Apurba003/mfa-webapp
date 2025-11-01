# distance_metrics.py - All Distance & Similarity Metrics

import numpy as np
from scipy.spatial.distance import euclidean, mahalanobis
from sklearn.covariance import LedoitWolf


class DistanceMetrics:
    """Calculate various distance and similarity metrics"""
    
    @staticmethod
    def euclidean_distance(x, y):
        """
        Euclidean distance: sqrt(sum((x_i - y_i)^2))
        
        Args:
            x: First vector
            y: Second vector
            
        Returns:
            float: Euclidean distance
        """
        try:
            x = np.array(x)
            y = np.array(y)
            return float(euclidean(x, y))
        except Exception as e:
            print(f"Error calculating Euclidean distance: {e}")
            return float('inf')
    
    @staticmethod
    def mahalanobis_distance(x, y, cov_matrix=None):
        """
        Mahalanobis distance: sqrt((x-y)^T * Cov^-1 * (x-y))
        Accounts for correlations between features
        
        Args:
            x: First vector
            y: Second vector
            cov_matrix: Covariance matrix
            
        Returns:
            float: Mahalanobis distance
        """
        try:
            x = np.array(x)
            y = np.array(y)
            
            if cov_matrix is None:
                cov_matrix = np.eye(len(x))
            else:
                cov_matrix = np.array(cov_matrix)
            
            # Use Ledoit-Wolf for robust covariance estimation
            try:
                lw = LedoitWolf()
                cov_matrix = lw.fit(np.vstack([x, y])).covariance_
            except:
                pass
            
            # Calculate inverse
            try:
                cov_inv = np.linalg.inv(cov_matrix)
            except:
                cov_inv = np.linalg.pinv(cov_matrix)
            
            diff = x - y
            distance = np.sqrt(diff @ cov_inv @ diff)
            return float(distance)
        except Exception as e:
            print(f"Error calculating Mahalanobis distance: {e}")
            return float('inf')
    
    @staticmethod
    def cosine_distance(x, y):
        """
        Cosine distance: 1 - (x·y) / (||x|| * ||y||)
        
        Args:
            x: First vector
            y: Second vector
            
        Returns:
            float: Cosine distance
        """
        try:
            x = np.array(x)
            y = np.array(y)
            
            dot_product = np.dot(x, y)
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            
            if norm_x == 0 or norm_y == 0:
                return 1.0
            
            similarity = dot_product / (norm_x * norm_y)
            return float(1 - similarity)
        except Exception as e:
            print(f"Error calculating Cosine distance: {e}")
            return 1.0
    
    @staticmethod
    def manhattan_distance(x, y):
        """
        Manhattan distance: sum(|x_i - y_i|)
        
        Args:
            x: First vector
            y: Second vector
            
        Returns:
            float: Manhattan distance
        """
        try:
            x = np.array(x)
            y = np.array(y)
            return float(np.sum(np.abs(x - y)))
        except Exception as e:
            print(f"Error calculating Manhattan distance: {e}")
            return float('inf')
    
    @staticmethod
    def similarity_score(distance, max_distance=1.0):
        """
        Convert distance to similarity score (0-1)
        
        Args:
            distance: Distance value
            max_distance: Maximum distance
            
        Returns:
            float: Similarity score (0-1)
        """
        try:
            similarity = 1 / (1 + distance)
            return float(np.clip(similarity, 0, 1))
        except Exception as e:
            print(f"Error calculating similarity score: {e}")
            return 0.0
    
    @staticmethod
    def statistical_distance(profile, test_sample):
        """
        Statistical distance using mean and standard deviation
        
        Args:
            profile: Profile with mean and std
            test_sample: Test sample
            
        Returns:
            float: Statistical distance score
        """
        try:
            profile_mean = np.array(profile.get('mean', []))
            profile_std = np.array(profile.get('std', []))
            test_sample = np.array(test_sample)
            
            if len(profile_mean) == 0:
                return 1.0
            
            # Z-score: how many standard deviations away
            z_scores = np.abs((test_sample - profile_mean) / (profile_std + 1e-8))
            distance = np.mean(z_scores)
            
            return float(np.clip(distance, 0, 10))
        except Exception as e:
            print(f"Error calculating statistical distance: {e}")
            return 1.0