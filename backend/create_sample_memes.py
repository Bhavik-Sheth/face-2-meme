"""
Create sample test meme images for demonstration.
This creates simple colored placeholder images for testing.
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def create_sample_meme(emotion: str, politician_name: str, color: tuple):
    """Create a simple colored test image with text."""
    # Create image
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color=color)
    draw = ImageDraw.Draw(img)
    
    # Add text
    text = f"{emotion.upper()}\n{politician_name}"
    
    # Try to use a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position (center)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((width - text_width) / 2, (height - text_height) / 2)
    
    # Draw text with shadow for visibility
    shadow_offset = 3
    draw.text((position[0] + shadow_offset, position[1] + shadow_offset), 
              text, fill=(0, 0, 0), font=font, align="center")
    draw.text(position, text, fill=(255, 255, 255), font=font, align="center")
    
    return img


def generate_sample_memes():
    """Generate sample memes for testing."""
    base_path = Path(__file__).parent / "assets" / "memes"
    
    samples = [
        ("happy", "Politician_A", (255, 215, 0)),    # Gold
        ("happy", "Politician_B", (255, 228, 181)),  # Light yellow
        ("sad", "Politician_C", (70, 130, 180)),     # Steel blue
        ("sad", "Politician_D", (100, 149, 237)),    # Cornflower blue
        ("angry", "Politician_E", (220, 20, 60)),    # Crimson
        ("angry", "Politician_F", (178, 34, 34)),    # Firebrick
        ("neutral", "Politician_G", (169, 169, 169)), # Gray
        ("surprise", "Politician_H", (255, 140, 0)),  # Orange
        ("fear", "Politician_I", (138, 43, 226)),     # Purple
        ("disgust", "Politician_J", (107, 142, 35)),  # Olive green
    ]
    
    print("Generating sample meme images...")
    for emotion, politician, color in samples:
        emotion_folder = base_path / emotion
        emotion_folder.mkdir(parents=True, exist_ok=True)
        
        img = create_sample_meme(emotion, politician, color)
        output_path = emotion_folder / f"{politician.lower()}.jpg"
        img.save(output_path, "JPEG", quality=95)
        print(f"  ✓ Created: {output_path}")
    
    print(f"\n✅ Generated {len(samples)} sample memes!")
    print("These are placeholder images for testing.")
    print("Replace them with actual politician meme images.")


if __name__ == "__main__":
    generate_sample_memes()
