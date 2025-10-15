"""
Complete Pipeline Test: Stage 1 + Stage 2
Tests emotion detection → meme selection integration.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from services.emotion_detector import EmotionDetector, detect_emotion_from_image
from services.politician_selector import select_and_output_meme
from services.meme_retrieval import get_available_emotions


def test_stage1_emotion_detection():
    """Test Stage 1: Emotion Detection"""
    print("=" * 70)
    print("STAGE 1: EMOTION DETECTION TEST")
    print("=" * 70)
    
    detector = EmotionDetector()
    
    # Test with sample meme images (they exist in assets/memes/)
    test_images = [
        "assets/memes/happy/politician_a.jpg",
        "assets/memes/sad/politician_c.jpg",
        "assets/memes/angry/politician_e.jpg",
    ]
    
    print("\nTesting emotion detection on sample images:")
    for img_path in test_images:
        full_path = backend_path / img_path
        if full_path.exists():
            result = detector.detect_emotion(str(full_path))
            print(f"\n  Image: {img_path}")
            print(f"  Success: {result['success']}")
            print(f"  Detected Emotion: {result['emotion']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Model Used: {result['model_used']}")
        else:
            print(f"\n  ⚠️  Image not found: {img_path}")
    
    print("\n" + "=" * 70)


def test_complete_pipeline():
    """Test Complete Pipeline: Stage 1 → Stage 2"""
    print("\n" + "=" * 70)
    print("COMPLETE PIPELINE TEST: STAGE 1 → STAGE 2")
    print("=" * 70)
    
    # Use a sample image
    test_image = backend_path / "assets/memes/happy/politician_a.jpg"
    
    if not test_image.exists():
        print("⚠️  No test images available. Run create_sample_memes.py first.")
        return
    
    print(f"\nInput: User image at {test_image}")
    print("\n--- STAGE 1: EMOTION DETECTION ---")
    
    # Stage 1: Detect emotion
    detector = EmotionDetector()
    emotion_result = detector.detect_emotion(str(test_image))
    
    print(f"  Detected Emotion: {emotion_result['emotion']}")
    print(f"  Confidence: {emotion_result['confidence']:.2f}")
    print(f"  Model: {emotion_result['model_used']}")
    
    if not emotion_result['success']:
        print(f"  ❌ Failed: {emotion_result['message']}")
        return
    
    detected_emotion = emotion_result['emotion']
    
    print(f"\n--- STAGE 2: MEME SELECTION ---")
    
    # Stage 2: Select and output meme
    meme_result = select_and_output_meme(detected_emotion)
    
    print(f"  Input Emotion: {detected_emotion}")
    print(f"  Selection Success: {meme_result['success']}")
    
    if meme_result['success']:
        print(f"  Selected Meme: {meme_result['meme_filename']}")
        print(f"  Original Path: {meme_result['meme_path']}")
        print(f"  Output Path: {meme_result.get('output_path', 'N/A')}")
        print(f"  Output Success: {meme_result.get('output_success', False)}")
    else:
        print(f"  ❌ Failed: {meme_result['message']}")
        return
    
    print("\n--- PIPELINE OUTPUT ---")
    print(f"  ✅ Complete pipeline executed successfully!")
    print(f"  📸 Final meme ready at: {meme_result.get('output_path')}")
    print("=" * 70)


def test_multiple_emotions():
    """Test pipeline with multiple emotions"""
    print("\n" + "=" * 70)
    print("TESTING PIPELINE WITH MULTIPLE EMOTIONS")
    print("=" * 70)
    
    available_emotions = get_available_emotions()
    print(f"\nAvailable emotions with memes: {available_emotions}")
    
    # Test each emotion
    for emotion in available_emotions[:3]:  # Test first 3 to keep output manageable
        print(f"\n--- Testing emotion: {emotion.upper()} ---")
        
        # Find a sample image for this emotion
        emotion_folder = backend_path / "assets" / "memes" / emotion
        images = list(emotion_folder.glob("*.jpg"))
        
        if not images:
            print(f"  ⚠️  No images found for {emotion}")
            continue
        
        test_image = images[0]
        
        # Stage 1: Detect emotion (will use placeholder)
        result = detect_emotion_from_image(str(test_image))
        print(f"  Stage 1 Output: {result}")
        
        # Stage 2: Select meme
        meme_result = select_and_output_meme(result if result else emotion)
        
        if meme_result['success']:
            print(f"  Stage 2 Output: {meme_result.get('output_path')}")
            print(f"  ✓ Pipeline completed for {emotion}")
        else:
            print(f"  ✗ Pipeline failed for {emotion}")
    
    print("\n" + "=" * 70)


def test_convenience_function():
    """Test the quick convenience function"""
    print("\n" + "=" * 70)
    print("TESTING CONVENIENCE FUNCTION")
    print("=" * 70)
    
    test_image = backend_path / "assets/memes/happy/politician_a.jpg"
    
    if test_image.exists():
        print(f"\nUsing detect_emotion_from_image():")
        emotion = detect_emotion_from_image(str(test_image))
        print(f"  Result: {emotion}")
        
        if emotion:
            print(f"\nPassing to Stage 2:")
            result = select_and_output_meme(emotion)
            print(f"  Meme selected: {result.get('meme_filename')}")
            print(f"  Output: {result.get('output_path')}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run all tests
    test_stage1_emotion_detection()
    test_complete_pipeline()
    test_multiple_emotions()
    test_convenience_function()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Add your trained emotion_teller.pkl model to backend/models/")
    print("2. Update preprocessing in emotion_detector.py if needed")
    print("3. Test with real user images")
    print("4. Integrate with FastAPI endpoints")
