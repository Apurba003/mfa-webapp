# auth_pipeline.py - Complete Authentication Pipeline

import numpy as np
from distance_metrics import DistanceMetrics


class AuthenticationPipeline:
    """Complete authentication pipeline"""
    
    def __init__(self, keystroke_config=None, face_config=None):
        """
        Initialize authentication pipeline
        
        Args:
            keystroke_config: Keystroke configuration
            face_config: Face configuration
        """
        self.keystroke_threshold = keystroke_config.get('threshold', 0.7) if keystroke_config else 0.7
        self.face_threshold = face_config.get('threshold', 0.6) if face_config else 0.6
        self.combined_threshold = 0.75
        self.distance_metrics = DistanceMetrics()
        print("✓ Authentication Pipeline Initialized")
    
    def authenticate_keystroke(self, test_features, profile):
        """
        Authenticate using keystroke dynamics
        
        Args:
            test_features: Test keystroke features
            profile: User keystroke profile
            
        Returns:
            tuple: (authenticated, confidence, details)
        """
        try:
            if not profile or not test_features:
                return False, 0.0, {'error': 'Invalid input'}
            
            profile_mean = np.array(profile.get('mean', []))
            profile_std = np.array(profile.get('std', []))
            profile_cov = np.array(profile.get('cov', []))
            
            test_features = np.array(test_features)
            
            details = {}
            
            # 1. Euclidean Distance
            euclidean_dist = self.distance_metrics.euclidean_distance(test_features, profile_mean)
            euclidean_score = 1 / (1 + euclidean_dist)
            details['euclidean_distance'] = float(euclidean_dist)
            details['euclidean_score'] = float(euclidean_score)
            
            # 2. Mahalanobis Distance
            mahal_dist = self.distance_metrics.mahalanobis_distance(test_features, profile_mean, profile_cov)
            mahal_score = 1 / (1 + mahal_dist)
            details['mahalanobis_distance'] = float(mahal_dist)
            details['mahalanobis_score'] = float(mahal_score)
            
            # 3. Statistical Distance
            stat_profile = {
                'mean': profile_mean.tolist(),
                'std': profile_std.tolist()
            }
            stat_dist = self.distance_metrics.statistical_distance(stat_profile, test_features)
            stat_score = 1 / (1 + stat_dist)
            details['statistical_distance'] = float(stat_dist)
            details['statistical_score'] = float(stat_score)
            
            # Average confidence
            confidence = (euclidean_score + mahal_score + stat_score) / 3
            
            # Decision
            authenticated = confidence >= self.keystroke_threshold
            
            details['threshold'] = self.keystroke_threshold
            details['method'] = 'keystroke'
            
            return authenticated, float(confidence), details
        
        except Exception as e:
            print(f"Error in keystroke authentication: {e}")
            return False, 0.0, {'error': str(e)}
    
    def authenticate_face(self, test_embedding, profile):
        """
        Authenticate using face recognition
        
        Args:
            test_embedding: Test face embedding
            profile: User face profile
            
        Returns:
            tuple: (authenticated, confidence, details)
        """
        try:
            if not profile or test_embedding is None:
                return False, 0.0, {'error': 'Invalid input'}
            
            profile_mean = np.array(profile.get('mean', []))
            profile_std = np.array(profile.get('std', []))
            profile_cov = np.array(profile.get('cov', []))
            
            test_embedding = np.array(test_embedding)
            
            details = {}
            
            # 1. Euclidean Distance
            euclidean_dist = self.distance_metrics.euclidean_distance(test_embedding, profile_mean)
            euclidean_score = 1 / (1 + euclidean_dist)
            details['euclidean_distance'] = float(euclidean_dist)
            details['euclidean_score'] = float(euclidean_score)
            
            # 2. Mahalanobis Distance
            try:
                mahal_dist = self.distance_metrics.mahalanobis_distance(test_embedding, profile_mean, profile_cov)
                mahal_score = 1 / (1 + mahal_dist)
            except:
                mahal_score = euclidean_score
                mahal_dist = euclidean_dist
            
            details['mahalanobis_distance'] = float(mahal_dist)
            details['mahalanobis_score'] = float(mahal_score)
            
            # 3. Cosine Distance
            cosine_dist = self.distance_metrics.cosine_distance(test_embedding, profile_mean)
            cosine_score = 1 - cosine_dist
            details['cosine_distance'] = float(cosine_dist)
            details['cosine_score'] = float(cosine_score)
            
            # Average confidence
            confidence = (euclidean_score + mahal_score + cosine_score) / 3
            
            # Decision
            authenticated = confidence >= self.face_threshold
            
            details['threshold'] = self.face_threshold
            details['method'] = 'face'
            
            return authenticated, float(confidence), details
        
        except Exception as e:
            print(f"Error in face authentication: {e}")
            return False, 0.0, {'error': str(e)}
    
    def authenticate_combined(self, keystroke_score, keystroke_passed, face_score, face_passed):
        """
        Combined authentication decision
        
        Args:
            keystroke_score: Keystroke confidence score
            keystroke_passed: Keystroke authentication result
            face_score: Face confidence score
            face_passed: Face authentication result
            
        Returns:
            tuple: (authenticated, combined_score, details)
        """
        try:
            # Weighted average
            keystroke_weight = 0.4
            face_weight = 0.6
            
            combined_score = (keystroke_score * keystroke_weight) + (face_score * face_weight)
            
            # Decision logic
            # Both must pass OR combined score is high enough
            if keystroke_passed and face_passed:
                authenticated = combined_score >= self.combined_threshold
            elif keystroke_passed or face_passed:
                authenticated = combined_score >= (self.combined_threshold * 1.1)
            else:
                authenticated = False
            
            details = {
                'keystroke_score': keystroke_score,
                'keystroke_passed': keystroke_passed,
                'face_score': face_score,
                'face_passed': face_passed,
                'combined_score': float(combined_score),
                'threshold': self.combined_threshold,
                'weights': {
                    'keystroke': keystroke_weight,
                    'face': face_weight
                }
            }
            
            return authenticated, float(combined_score), details
        
        except Exception as e:
            print(f"Error in combined authentication: {e}")
            return False, 0.0, {'error': str(e)}