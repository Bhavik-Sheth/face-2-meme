"""
Example: Pipeline Integration - Stage 1 + Stage 2
Demonstrates how emotion detection connects to meme selection.
"""

# STAGE 1: Emotion Detection (now using actual code!)
from services.emotion_detector import EmotionDetector

def stage1_emotion_detection(image_path: str) -> str:
    """
    Stage 1: Detect emotion from user's face image using EmotionDetector.
    Uses emotion_teller.pkl model (or placeholder if not available).
    """
    detector = EmotionDetector()
    result = detector.detect_emotion(image_path)
    
    print(f"[Stage 1] Emotion Detection:")
    print(f"  - Success: {result['success']}")
    print(f"  - Detected Emotion: {result['emotion']}")
    print(f"  - Confidence: {result['confidence']:.2f}")
    print(f"  - Model Used: {result['model_used']}")
    
    if result['success']:
        return result['emotion']
    else:
        print(f"  - Error: {result['message']}")
        return "neutral"  # Fallback to neutral


# STAGE 2: Meme Selection with Output
from services.politician_selector import select_politician_meme, select_and_output_meme

def stage2_meme_selection(emotion: str, output: bool = True) -> dict:
    """
    Stage 2: Select a politician meme based on detected emotion.
    
    Args:
        emotion (str): Detected emotion
        output (bool): If True, copy meme to outputs folder
    """
    if output:
        result = select_and_output_meme(emotion)
        print(f"[Stage 2] Meme selection and output:")
        print(f"  - Selection Success: {result['success']}")
        print(f"  - Message: {result['message']}")
        if result['success']:
            print(f"  - Selected file: {result['meme_filename']}")
            print(f"  - Original path: {result['meme_path']}")
            if result.get('output_success'):
                print(f"  - Output path: {result['output_path']}")
                print(f"  - Output status: {result['output_message']}")
    else:
        result = select_politician_meme(emotion)
        print(f"[Stage 2] Meme selection result:")
        print(f"  - Success: {result['success']}")
        print(f"  - Message: {result['message']}")
        if result['success']:
            print(f"  - Selected file: {result['meme_filename']}")
            print(f"  - Full path: {result['meme_path']}")
    return result


# COMPLETE PIPELINE
def complete_pipeline(user_image_path: str):
    """
    Complete pipeline: Image → Emotion → Meme
    """
    print("=" * 70)
    print("COMPLETE PIPELINE EXECUTION")
    print("=" * 70)
    print(f"Input: User image at {user_image_path}\n")
    
    # Stage 1: Detect emotion
    detected_emotion = stage1_emotion_detection(user_image_path)
    print()
    
    # Stage 2: Select meme
    meme_result = stage2_meme_selection(detected_emotion)
    print()
    
    # Output
    print("=" * 70)
    print("PIPELINE OUTPUT")
    print("=" * 70)
    if meme_result['success']:
        print(f"✅ Successfully matched emotion '{detected_emotion}' to meme:")
        print(f"   Original: {meme_result['meme_path']}")
        if meme_result.get('output_path'):
            print(f"   Output: {meme_result['output_path']}")
            print(f"\n📸 Meme ready for display at: {meme_result['output_path']}")
        else:
            print(f"\n📸 Next step: Display this meme to the user!")
    else:
        print(f"❌ Failed to find meme for emotion '{detected_emotion}'")
        print(f"   Reason: {meme_result['message']}")
    print("=" * 70)
    
    return meme_result


if __name__ == "__main__":
    # Example usage with actual image
    from pathlib import Path
    backend_path = Path(__file__).parent
    
    # Use a sample image for testing
    user_image = str(backend_path / "assets/memes/happy/politician_a.jpg")
    
    if Path(user_image).exists():
        result = complete_pipeline(user_image)
    else:
        print("⚠️  No test image available. Run create_sample_memes.py first.")
    
    print("\n\n" + "=" * 70)
    print("Testing with different sample images:")
    print("=" * 70)
    
    # Test with various sample images
    test_images = [
        "assets/memes/happy/politician_b.jpg",
        "assets/memes/sad/politician_c.jpg",
        "assets/memes/angry/politician_e.jpg",
    ]
    
    for img_path in test_images:
        full_path = backend_path / img_path
        if full_path.exists():
            print(f"\n{'='*70}")
            print(f"Testing with: {img_path}")
            print('='*70)
            result = complete_pipeline(str(full_path))
        else:
            print(f"\n⚠️  Image not found: {img_path}")
