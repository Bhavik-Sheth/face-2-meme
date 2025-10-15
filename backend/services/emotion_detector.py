"""
Stage 1: Emotion Detection Service
Takes a user image as input and returns detected emotion as a string.
Uses MobileNetV2 model trained on FER2013 dataset.
"""
import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.amp as amp


class EmotionDetector:
    """Detects emotion from user's face image using ML model."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the emotion detector with the MobileNetV2 model.
        
        Args:
            model_path (str): Path to emotion_teller.pth model.
                            If None, uses backend/models/emotion_teller.pth
        """
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path is None:
            # Default to backend/models/emotion_teller.pth
            current_dir = Path(__file__).parent.parent
            self.model_path = current_dir / "models" / "emotion_teller.pth"
        else:
            self.model_path = Path(model_path)
        
        # Class names in the same order as training data
        # Based on FER2013 dataset
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Supported emotions (must match Stage 2)
        self.supported_emotions = [
            "happy", "sad", "angry", "neutral", 
            "surprise", "fear", "disgust"
        ]
        
        # Preprocessing transformation (matches validation transforms from training)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # FER2013 is grayscale
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        self.model = self._load_model()
        self.model_loaded = self.model is not None
    
    def _load_model(self):
        """
        Load the MobileNetV2 emotion detection model with custom classifier.
        Returns None if model file not found (uses placeholder mode).
        """
        if not self.model_path.exists():
            print(f"⚠️  Model not found at {self.model_path}")
            print(f"   Using placeholder emotion detection.")
            return None
        
        try:
            # Create the model architecture (must match training script)
            model = models.mobilenet_v2(weights=None)
            
            # Number of classes based on FER2013 dataset
            num_classes = 7
            
            # Replace classifier with the same custom architecture used during training
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.last_channel, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            
            # Load the saved state dictionary
            model.load_state_dict(torch.load(str(self.model_path), map_location=self.device))
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            print(f"✓ Loaded MobileNetV2 emotion model from {self.model_path}")
            return model
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load model: {e}")
            print(f"   Using placeholder emotion detection.")
            return None
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for MobileNetV2 model input.
        Applies grayscale conversion, resizing, normalization.
        
        Args:
            image_path (str): Path to user's image
        
        Returns:
            torch.Tensor: Preprocessed image tensor ready for model
        """
        try:
            # Load and convert image to RGB
            image = Image.open(image_path).convert('RGB')
            
            # Apply transformations (grayscale, resize, normalize)
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            return input_tensor
        
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {e}")
    
    def detect_emotion(self, image_path: str) -> dict:
        """
        Detect emotion from user's face image using MobileNetV2 model.
        
        Args:
            image_path (str): Path to user's captured image
        
        Returns:
            dict: {
                "emotion": str,           # Detected emotion
                "confidence": float,      # Confidence score (0-1)
                "success": bool,          # Detection succeeded
                "message": str,           # Status message
                "model_used": str         # Model type (MobileNetV2/placeholder)
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
            # Use real model if available
            if self.model is not None:
                emotion, confidence = self._predict_with_model(image_path)
                model_used = "MobileNetV2"
            else:
                # Placeholder mode for testing without model
                emotion = self._placeholder_prediction(image_path)
                confidence = 0.75
                model_used = "placeholder"
            
            # Validate emotion is in supported list
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
    
    def _predict_with_model(self, image_path: str) -> tuple:
        """
        Perform emotion prediction using the loaded MobileNetV2 model.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            tuple: (emotion_str, confidence_float)
        """
        try:
            # Preprocess the image
            input_tensor = self.preprocess_image(image_path)
            
            # Perform inference
            with torch.no_grad():
                # Use autocast for consistency with training/validation
                with amp.autocast(device_type=self.device.type):
                    outputs = self.model(input_tensor)
                    # Apply Softmax to get probabilities
                    probs = nn.Softmax(dim=1)(outputs)
                    # Get the top prediction
                    confidence, pred_class_idx = torch.max(probs, dim=1)
            
            # Get emotion string from class names
            emotion = self.class_names[pred_class_idx.item()]
            confidence_value = confidence.item()
            
            return emotion, confidence_value
            
        except Exception as e:
            print(f"Error during model prediction: {e}")
            # Fallback to placeholder
            return self._placeholder_prediction(image_path), 0.50
    
    def _placeholder_prediction(self, image_path: str) -> str:
        """
        Placeholder emotion prediction for testing.
        Returns a deterministic emotion based on image filename.
        Replace this when emotion_teller.pth is available.
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
