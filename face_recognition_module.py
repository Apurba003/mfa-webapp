# face_recognition_module.py - Face Recognition & Feature Extraction

import numpy as np
import cv2
import base64
from deepface import DeepFace
import os
from datetime import datetime


class FaceRecognitionModule:
    """Face recognition using DeepFace"""
    
    def __init__(self):
        """Initialize face recognition"""
        self.model_name = "VGGFace2"
        self.detector_backend = "opencv"
        print("✓ Face Recognition Module Initialized")
    
    def process_base64_image(self, image_base64, save_path=None):
        """
        Process base64 encoded image
        
        Args:
            image_base64: Base64 encoded image string
            save_path: Optional path to save image
            
        Returns:
            np.ndarray: Image array
        """
        try:
            # Remove header if present
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
            
            # Save if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, img)
            
            return img
        
        except Exception as e:
            print(f"Error processing base64 image: {e}")
            return None
    
    def extract_embedding(self, image):
        """
        Extract face embedding using DeepFace
        
        Args:
            image: Image array (BGR format from OpenCV)
            
        Returns:
            np.ndarray: Face embedding vector
        """
        try:
            # Verify image is valid
            if image is None or len(image.shape) != 3:
                return None
            
            # Extract embedding
            embedding_objs = DeepFace.represent(
                img_path=image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
            
            if not embedding_objs:
                return None
            
            # Return embedding vector
            embedding = np.array(embedding_objs[0]['embedding'])
            return embedding
        
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            return None
    
    def get_face_quality(self, image):
        """
        Assess face image quality
        
        Args:
            image: Image array
            
        Returns:
            float: Quality score (0-1)
        """
        try:
            if image is None:
                return 0.0
            
            # Convert BGR to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (blur detection)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1
            quality = min(laplacian_var / 100, 1.0)
            
            return float(quality)
        
        except Exception as e:
            print(f"Error calculating face quality: {e}")
            return 0.0
    
    def get_profile_embeddings(self, embeddings):
        """
        Get profile statistics from multiple embeddings
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            dict: Profile statistics
        """
        try:
            if not embeddings:
                return None
            
            embeddings_array = np.array(embeddings)
            
            profile = {
                'mean': embeddings_array.mean(axis=0).tolist(),
                'std': embeddings_array.std(axis=0).tolist(),
                'cov': np.cov(embeddings_array.T).tolist(),
                'count': len(embeddings),
                'dimension': len(embeddings_array[0])
            }
            
            return profile
        
        except Exception as e:
            print(f"Error getting face profile: {e}")
            return None
    
    def detect_faces(self, image):
        """
        Detect faces in image
        
        Args:
            image: Image array
            
        Returns:
            list: Face detections
        """
        try:
            detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            
            return faces
        
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def compare_faces(self, embedding1, embedding2):
        """
        Compare two face embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            float: Similarity score (0-1)
        """
        try:
            embedding1 = np.array(embedding1)
            embedding2 = np.array(embedding2)
            
            # Euclidean distance
            distance = np.linalg.norm(embedding1 - embedding2)
            
            # Convert to similarity (0-1)
            similarity = 1 / (1 + distance)
            
            return float(similarity)
        
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return 0.0