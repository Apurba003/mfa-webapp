# keystroke_dynamics.py - Keystroke Feature Extraction

import numpy as np
import json


class KeystrokeDynamics:
    """Extract and process keystroke dynamics features"""
    
    def __init__(self):
        """Initialize keystroke dynamics extractor"""
        self.reference_text = "the quick brown fox jumps over the lazy dog"
    
    def extract_features(self, keystroke_data):
        """
        Extract features from keystroke data
        
        Args:
            keystroke_data: Dict with timestamps, keys, events
            
        Returns:
            dict: Extracted features
        """
        try:
            timestamps = keystroke_data.get('timestamps', [])
            keys = keystroke_data.get('keys', [])
            events = keystroke_data.get('events', [])
            
            if not timestamps or len(timestamps) < 2:
                return None
            
            features = {}
            
            # 1. Dwell Time (time key is held down)
            dwell_times = self._extract_dwell_times(timestamps, events)
            features['dwell_time_mean'] = np.mean(dwell_times) if dwell_times else 0
            features['dwell_time_std'] = np.std(dwell_times) if dwell_times else 0
            
            # 2. Flight Time (time between key releases and presses)
            flight_times = self._extract_flight_times(timestamps, events)
            features['flight_time_mean'] = np.mean(flight_times) if flight_times else 0
            features['flight_time_std'] = np.std(flight_times) if flight_times else 0
            
            # 3. Overall typing speed
            total_time = max(timestamps) - min(timestamps)
            key_count = len([e for e in events if e == 'keydown'])
            features['typing_speed'] = key_count / (total_time / 1000) if total_time > 0 else 0
            
            # 4. Typing rhythm (inter-key intervals)
            inter_key_intervals = self._extract_inter_key_intervals(timestamps, events)
            features['rhythm_mean'] = np.mean(inter_key_intervals) if inter_key_intervals else 0
            features['rhythm_std'] = np.std(inter_key_intervals) if inter_key_intervals else 0
            
            # 5. Key press variance
            features['variance'] = np.var(dwell_times) if dwell_times else 0
            
            # 6. Total time
            features['total_time'] = total_time
            
            # 7. Entropy (uniqueness of pattern)
            features['entropy'] = self._calculate_entropy(dwell_times)
            
            return features
        
        except Exception as e:
            print(f"Error extracting keystroke features: {e}")
            return None
    
    def _extract_dwell_times(self, timestamps, events):
        """Extract dwell times (key hold duration)"""
        dwell_times = []
        i = 0
        while i < len(events) - 1:
            if events[i] == 'keydown' and events[i + 1] == 'keyup':
                dwell = timestamps[i + 1] - timestamps[i]
                if dwell >= 0:
                    dwell_times.append(dwell)
            i += 1
        return dwell_times
    
    def _extract_flight_times(self, timestamps, events):
        """Extract flight times (gap between keys)"""
        flight_times = []
        i = 0
        while i < len(events) - 1:
            if events[i] == 'keyup' and events[i + 1] == 'keydown':
                flight = timestamps[i + 1] - timestamps[i]
                if flight >= 0:
                    flight_times.append(flight)
            i += 1
        return flight_times
    
    def _extract_inter_key_intervals(self, timestamps, events):
        """Extract inter-key intervals"""
        intervals = []
        keydown_times = [t for i, t in enumerate(timestamps) if events[i] == 'keydown']
        
        for i in range(1, len(keydown_times)):
            interval = keydown_times[i] - keydown_times[i - 1]
            if interval > 0:
                intervals.append(interval)
        
        return intervals
    
    def _calculate_entropy(self, values):
        """Calculate Shannon entropy of values"""
        if not values or len(values) < 2:
            return 0
        
        values = np.array(values)
        # Normalize to 0-1
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val - min_val == 0:
            return 0
        
        normalized = (values - min_val) / (max_val - min_val)
        
        # Bin into 10 intervals
        hist, _ = np.histogram(normalized, bins=10)
        hist = hist / len(values)
        hist = hist[hist > 0]
        
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return float(entropy)
    
    def get_profile_stats(self, keystroke_list):
        """
        Get profile statistics from multiple keystroke samples
        
        Args:
            keystroke_list: List of feature dictionaries
            
        Returns:
            dict: Profile statistics (mean, std, etc.)
        """
        try:
            if not keystroke_list:
                return None
            
            keystroke_array = np.array([list(ks.values()) for ks in keystroke_list])
            
            profile = {
                'mean': np.mean(keystroke_array, axis=0).tolist(),
                'std': np.std(keystroke_array, axis=0).tolist(),
                'min': np.min(keystroke_array, axis=0).tolist(),
                'max': np.max(keystroke_array, axis=0).tolist(),
                'cov': np.cov(keystroke_array.T).tolist(),
                'count': len(keystroke_list),
                'feature_names': list(keystroke_list[0].keys())
            }
            
            return profile
        
        except Exception as e:
            print(f"Error getting keystroke profile: {e}")
            return None