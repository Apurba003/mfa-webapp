# classification_engine.py - Machine Learning Classification Algorithms

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os


class ClassificationEngine:
    """Machine learning classification for biometric authentication"""
    
    def __init__(self):
        """Initialize classification algorithms"""
        self.scaler = StandardScaler()
        self.models = {
            'neural_network': None,
            'random_forest': None,
            'svm': None,
            'knn': None
        }
        print("✓ Classification Engine Initialized")
    
    def train_neural_network(self, X_train, y_train):
        """
        Train Neural Network classifier
        
        MLP: Multi-Layer Perceptron
        Architecture: Input -> 128 neurons -> 64 neurons -> 32 neurons -> Output
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            MLPClassifier: Trained model
        """
        try:
            model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                learning_rate='adaptive',
                early_stopping=True,
                validation_fraction=0.1
            )
            
            X_scaled = self.scaler.fit_transform(X_train)
            model.fit(X_scaled, y_train)
            
            self.models['neural_network'] = model
            print("✓ Neural Network trained")
            return model
        
        except Exception as e:
            print(f"Error training Neural Network: {e}")
            return None
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest classifier
        
        Ensemble of 100 decision trees
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            RandomForestClassifier: Trained model
        """
        try:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            self.models['random_forest'] = model
            print("✓ Random Forest trained")
            return model
        
        except Exception as e:
            print(f"Error training Random Forest: {e}")
            return None
    
    def train_svm(self, X_train, y_train):
        """
        Train Support Vector Machine classifier
        
        Uses RBF (Radial Basis Function) kernel
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            SVC: Trained model
        """
        try:
            model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            
            X_scaled = self.scaler.fit_transform(X_train)
            model.fit(X_scaled, y_train)
            
            self.models['svm'] = model
            print("✓ SVM trained")
            return model
        
        except Exception as e:
            print(f"Error training SVM: {e}")
            return None
    
    def train_knn(self, X_train, y_train):
        """
        Train k-Nearest Neighbors classifier
        
        Uses k=5, distance-weighted voting
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            KNeighborsClassifier: Trained model
        """
        try:
            model = KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean',
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            self.models['knn'] = model
            print("✓ k-NN trained")
            return model
        
        except Exception as e:
            print(f"Error training k-NN: {e}")
            return None
    
    def predict_neural_network(self, X_test):
        """Predict using Neural Network"""
        try:
            if self.models['neural_network'] is None:
                return None
            
            X_scaled = self.scaler.transform(X_test)
            prediction = self.models['neural_network'].predict(X_scaled)
            probability = self.models['neural_network'].predict_proba(X_scaled)
            
            return prediction, probability
        
        except Exception as e:
            print(f"Error predicting with Neural Network: {e}")
            return None
    
    def predict_random_forest(self, X_test):
        """Predict using Random Forest"""
        try:
            if self.models['random_forest'] is None:
                return None
            
            prediction = self.models['random_forest'].predict(X_test)
            probability = self.models['random_forest'].predict_proba(X_test)
            
            return prediction, probability
        
        except Exception as e:
            print(f"Error predicting with Random Forest: {e}")
            return None
    
    def predict_svm(self, X_test):
        """Predict using SVM"""
        try:
            if self.models['svm'] is None:
                return None
            
            X_scaled = self.scaler.transform(X_test)
            prediction = self.models['svm'].predict(X_scaled)
            probability = self.models['svm'].predict_proba(X_scaled)
            
            return prediction, probability
        
        except Exception as e:
            print(f"Error predicting with SVM: {e}")
            return None
    
    def predict_knn(self, X_test):
        """Predict using k-NN"""
        try:
            if self.models['knn'] is None:
                return None
            
            prediction = self.models['knn'].predict(X_test)
            probability = self.models['knn'].predict_proba(X_test)
            
            return prediction, probability
        
        except Exception as e:
            print(f"Error predicting with k-NN: {e}")
            return None
    
    def ensemble_predict(self, X_test):
        """
        Ensemble prediction from all models
        
        Returns:
            dict: Predictions from all models with confidence
        """
        try:
            results = {}
            
            # Neural Network
            if self.models['neural_network']:
                nn_pred, nn_prob = self.predict_neural_network(X_test)
                results['neural_network'] = {
                    'prediction': nn_pred[0] if nn_pred is not None else None,
                    'confidence': float(max(nn_prob[0])) if nn_prob is not None else 0.0
                }
            
            # Random Forest
            if self.models['random_forest']:
                rf_pred, rf_prob = self.predict_random_forest(X_test)
                results['random_forest'] = {
                    'prediction': rf_pred[0] if rf_pred is not None else None,
                    'confidence': float(max(rf_prob[0])) if rf_prob is not None else 0.0
                }
            
            # SVM
            if self.models['svm']:
                svm_pred, svm_prob = self.predict_svm(X_test)
                results['svm'] = {
                    'prediction': svm_pred[0] if svm_pred is not None else None,
                    'confidence': float(max(svm_prob[0])) if svm_prob is not None else 0.0
                }
            
            # k-NN
            if self.models['knn']:
                knn_pred, knn_prob = self.predict_knn(X_test)
                results['knn'] = {
                    'prediction': knn_pred[0] if knn_pred is not None else None,
                    'confidence': float(max(knn_prob[0])) if knn_prob is not None else 0.0
                }
            
            return results
        
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            return {}
    
    def save_model(self, model_name, path):
        """Save model to disk"""
        try:
            if self.models[model_name]:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    pickle.dump(self.models[model_name], f)
                print(f"✓ Model saved: {path}")
        
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, model_name, path):
        """Load model from disk"""
        try:
            with open(path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
                print(f"✓ Model loaded: {path}")
        
        except Exception as e:
            print(f"Error loading model: {e}")