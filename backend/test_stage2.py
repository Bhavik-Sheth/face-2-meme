"""
Test script for Stage 2: Politician/Meme Selector
Tests the emotion-to-meme selection functionality including output.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from services.politician_selector import (
    PoliticianSelector, 
    select_politician_meme,
    select_and_output_meme
)
from services.meme_retrieval import get_meme_for_emotion, get_available_emotions


def test_politician_selector():
    """Test the PoliticianSelector class."""
    print("=" * 60)
    print("Testing Stage 2: Politician/Meme Selector")
    print("=" * 60)
    
    selector = PoliticianSelector()
    
    # Test 1: Check available emotions
    print("\n1. Checking available emotions:")
    available = get_available_emotions()
    print(f"   Available emotions: {available}")
    if not available:
        print("   ⚠️  Warning: No memes found in any emotion folder!")
        print("   Please add meme images to emotion subfolders.")
    
    # Test 2: Test each supported emotion
    print("\n2. Testing meme selection for each emotion:")
    test_emotions = ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"]
    
    for emotion in test_emotions:
        result = select_politician_meme(emotion)
        print(f"\n   Emotion: {emotion}")
        print(f"   Success: {result['success']}")
        print(f"   Message: {result['message']}")
        if result['success']:
            print(f"   Meme File: {result['meme_filename']}")
            print(f"   Meme Path: {result['meme_path']}")
    
    # Test 3: Test invalid emotion
    print("\n3. Testing invalid emotion:")
    result = select_politician_meme("invalid_emotion")
    print(f"   Success: {result['success']}")
    print(f"   Message: {result['message']}")
    
    # Test 4: Test wrapper function
    print("\n4. Testing meme_retrieval wrapper:")
    result = get_meme_for_emotion("happy")
    print(f"   Success: {result['success']}")
    print(f"   Message: {result['message']}")
    
    # Test 5: Test output functionality (if memes exist)
    print("\n5. Testing meme output to folder:")
    if available:
        test_emotion = available[0]
        print(f"   Testing with emotion: {test_emotion}")
        output_result = select_and_output_meme(test_emotion)
        print(f"   Selection Success: {output_result['success']}")
        if output_result.get('output_success'):
            print(f"   Output Success: {output_result['output_success']}")
            print(f"   Output Path: {output_result.get('output_path')}")
            print(f"   Output Message: {output_result.get('output_message')}")
        else:
            print(f"   Output Failed: {output_result.get('output_message', 'Unknown error')}")
    else:
        print("   ⚠️  Skipping - no memes available to test output")
    
    # Test 6: Test output with wrapper
    print("\n6. Testing output via meme_retrieval wrapper:")
    if available:
        result = get_meme_for_emotion(available[0], output=True)
        print(f"   Success: {result.get('output_success', result['success'])}")
        if result.get('output_path'):
            print(f"   Output Path: {result['output_path']}")
    else:
        print("   ⚠️  Skipping - no memes available")
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_politician_selector()
