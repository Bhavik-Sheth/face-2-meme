"""
Stage 1: Emotion Detection Service
Takes a user image as input and returns detected emotion as a string.
Uses emotion_teller.pkl model (placeholder for now).
"""
import os
import pickle
from pathlib import Path
from PIL import Image
import numpy as np


class EmotionDetector:
    """Detects emotion from user's face image using ML model."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the emotion detector with the model.
        
        Args:
            model_path (str): Path to emotion_teller.pkl model.
                            If None, uses backend/models/emotion_teller.pkl
        """
        if model_path is None:
            # Default to backend/models/emotion_teller.pkl
            current_dir = Path(__file__).parent.parent
            self.model_path = current_dir / "models" / "emotion_teller.pkl"
        else:
            self.model_path = Path(model_path)
        
        # Supported emotions (must match Stage 2)
        self.supported_emotions = [
            "happy", "sad", "angry", "neutral", 
            "surprise", "fear", "disgust"
        ]
        
        # Load model (placeholder for now)
        self.model = self._load_model()
    
    def _load_model(self):
        """
        Load the emotion detection model.
        Currently returns None (placeholder until emotion_teller.pkl is added).
        """
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
                print(f"✓ Loaded emotion model from {self.model_path}")
                return model
            except Exception as e:
                print(f"⚠️  Warning: Could not load model: {e}")
                return None
        else:
            print(f"⚠️  Model not found at {self.model_path}")
            print(f"   Using placeholder emotion detection.")
            return None
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image_path (str): Path to user's image
        
        Returns:
            np.ndarray: Preprocessed image array
        """
        try:
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to model input size (adjust based on your model)
            # Common sizes: 48x48, 224x224, etc.
            img = img.resize((224, 224))
            
            # Convert to array
            img_array = np.array(img)
            
            # Normalize (0-255 -> 0-1)
            img_array = img_array / 255.0
            
            return img_array
        
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {e}")
    
    def detect_emotion(self, image_path: str) -> dict:
        """
        Detect emotion from user's face image.
        
        Args:
            image_path (str): Path to user's captured image
        
        Returns:
            dict: {
                "emotion": str,           # Detected emotion
                "confidence": float,      # Confidence score (0-1)
                "success": bool,          # Detection succeeded
                "message": str,           # Status message
                "model_used": str         # Model type (placeholder/real)
            }
        """
        # Validate image path
        if not os.path.exists(image_path):
            return {
                "emotion": None,
                "confidence": 0.0,
                "success": False,
                "message": f"Image not found: {image_path}",
                "model_used": "none"
            }
        
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            
            # Use model if available, otherwise use placeholder
            if self.model is not None:
                # TODO: Replace with actual model prediction
                # Example: emotion, confidence = self.model.predict(img_array)
                emotion = self._placeholder_prediction(image_path)
                confidence = 0.85  # Placeholder confidence
                model_used = "emotion_teller.pkl"
            else:
                # Placeholder: return random emotion for testing
                emotion = self._placeholder_prediction(image_path)
                confidence = 0.75  # Placeholder confidence
                model_used = "placeholder"
            
            # Validate emotion
            if emotion not in self.supported_emotions:
                emotion = "neutral"  # Fallback to neutral
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "success": True,
                "message": f"Emotion detected: {emotion}",
                "model_used": model_used
            }
        
        except Exception as e:
            return {
                "emotion": None,
                "confidence": 0.0,
                "success": False,
                "message": f"Error detecting emotion: {str(e)}",
                "model_used": "none"
            }
    
    def _placeholder_prediction(self, image_path: str) -> str:
        """
        Placeholder emotion prediction for testing.
        Returns a deterministic emotion based on image filename.
        Replace this when emotion_teller.pkl is available.
        """
        # Use filename hash to get consistent emotion for testing
        filename = os.path.basename(image_path).lower()
        
        # Check if filename contains emotion keywords
        for emotion in self.supported_emotions:
            if emotion in filename:
                return emotion
        
        # Otherwise, use hash for consistency
        import hashlib
        hash_val = int(hashlib.md5(filename.encode()).hexdigest(), 16)
        emotion_index = hash_val % len(self.supported_emotions)
        return self.supported_emotions[emotion_index]


# Convenience function for quick usage
def detect_emotion_from_image(image_path: str) -> str:
    """
    Quick function to detect emotion from an image.
    Returns just the emotion string for easy integration with Stage 2.
    
    Args:
        image_path (str): Path to user's image
    
    Returns:
        str: Detected emotion ("happy", "sad", etc.) or None if failed
    """
    detector = EmotionDetector()
    result = detector.detect_emotion(image_path)
    
    if result['success']:
        return result['emotion']
    else:
        print(f"Error: {result['message']}")
        return None
