"""
Simple script to run the Face-to-Meme pipeline on your image.
"""
from services.emotion_detector import detect_emotion_from_image
from services.politician_selector import select_and_output_meme
from pathlib import Path


def run_face_to_meme(image_path: str):
    """
    Run the complete Face-to-Meme pipeline.
    
    Args:
        image_path: Path to your face image
    """
    print("=" * 70)
    print("FACE-TO-MEME PIPELINE")
    print("=" * 70)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found at {image_path}")
        print("\nPlease provide a valid image path.")
        return
    
    print(f"\n📸 Input Image: {image_path}")
    print("\n" + "-" * 70)
    print("STAGE 1: Detecting Emotion...")
    print("-" * 70)
    
    # Stage 1: Detect emotion
    emotion = detect_emotion_from_image(image_path)
    
    if emotion is None:
        print("❌ Failed to detect emotion")
        return
    
    print(f"✅ Detected Emotion: {emotion.upper()}")
    
    print("\n" + "-" * 70)
    print("STAGE 2: Selecting Matching Meme...")
    print("-" * 70)
    
    # Stage 2: Get matching meme
    result = select_and_output_meme(emotion)
    
    if result['success']:
        print(f"✅ Meme Selected: {result['meme_filename']}")
        print(f"📂 Original Location: {result['meme_path']}")
        
        if result.get('output_success'):
            print(f"\n🎉 SUCCESS! Final meme saved at:")
            print(f"   {result['output_path']}")
            print(f"\n💡 You can now open this file to see your meme!")
        else:
            print(f"⚠️  Meme selected but not saved: {result.get('output_message')}")
    else:
        print(f"❌ Failed to select meme: {result['message']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 70)
    print("WELCOME TO FACE-TO-MEME GENERATOR")
    print("=" * 70)
    
    # Check if image path provided
    if len(sys.argv) > 1:
        # User provided image path as command line argument
        image_path = sys.argv[1]
        run_face_to_meme(image_path)
    else:
        # Interactive mode - ask for image path
        print("\nNo image path provided.")
        print("\nUsage:")
        print("  python run_model.py path/to/your/image.png")
        print("\nOr enter the path now:")
        
        image_path = input("\nImage path: ").strip().strip('"').strip("'")
        
        if image_path:
            run_face_to_meme(image_path)
        else:
            print("\n❌ No image path provided. Exiting.")
            
            # Show example with test image
            print("\n" + "-" * 70)
            print("RUNNING WITH SAMPLE IMAGE...")
            print("-" * 70)
            backend_path = Path(__file__).parent
            sample_image = backend_path / "assets/memes/happy/politician_a.png"
            
            if sample_image.exists():
                run_face_to_meme(str(sample_image))
            else:
                print("No sample images found. Please run: python create_sample_memes.py")
