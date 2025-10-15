"""
Test script for MobileNetV2 emotion detection integration.
Tests the emotion detector with the new MobileNetV2 model architecture.
"""
from pathlib import Path
from services.emotion_detector import EmotionDetector, detect_emotion_from_image


def test_mobilenetv2_integration():
    """Test the MobileNetV2 emotion detector integration."""
    print("=" * 70)
    print("TESTING MOBILENETV2 EMOTION DETECTION")
    print("=" * 70)
    
    # Initialize detector
    print("\n1. Initializing EmotionDetector...")
    detector = EmotionDetector()
    
    print(f"   Device: {detector.device}")
    print(f"   Model Path: {detector.model_path}")
    print(f"   Model Loaded: {detector.model_loaded}")
    print(f"   Supported Emotions: {detector.supported_emotions}")
    print(f"   Class Names (FER2013): {detector.class_names}")
    
    # Test with sample images
    backend_path = Path(__file__).parent
    test_images = [
        backend_path / "assets/memes/happy/politician_a.png",
        backend_path / "assets/memes/sad/politician_c.png",
        backend_path / "assets/memes/angry/politician_e.png",
    ]
    
    print("\n2. Testing emotion detection on sample images:")
    print("-" * 70)
    
    for img_path in test_images:
        if img_path.exists():
            print(f"\n📸 Image: {img_path.name}")
            
            # Test full detection method
            result = detector.detect_emotion(str(img_path))
            
            print(f"   Success: {result['success']}")
            print(f"   Emotion: {result['emotion']}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"   Model Used: {result['model_used']}")
            print(f"   Message: {result['message']}")
        else:
            print(f"\n⚠️  Image not found: {img_path}")
    
    # Test convenience function
    print("\n" + "=" * 70)
    print("3. Testing convenience function:")
    print("-" * 70)
    
    if test_images[0].exists():
        emotion = detect_emotion_from_image(str(test_images[0]))
        print(f"   detect_emotion_from_image() → {emotion}")
    
    print("\n" + "=" * 70)
    print("✅ MobileNetV2 Integration Test Complete!")
    print("=" * 70)
    
    # Print model architecture info if model is loaded
    if detector.model is not None:
        print("\n📊 Model Architecture:")
        print(f"   Type: MobileNetV2")
        print(f"   Parameters: {sum(p.numel() for p in detector.model.parameters()):,}")
        print(f"   Trainable: {sum(p.numel() for p in detector.model.parameters() if p.requires_grad):,}")
    
    print("\n💡 Next Steps:")
    if not detector.model_loaded:
        print("   1. Place your trained emotion_teller.pth in models/ directory")
        print("   2. Ensure the model architecture matches the training script")
        print("   3. Run this test again to verify the model loads correctly")
    else:
        print("   ✓ Model loaded successfully!")
        print("   ✓ Ready for production use")
        print("   ✓ Test with your own face images")


if __name__ == "__main__":
    test_mobilenetv2_integration()
