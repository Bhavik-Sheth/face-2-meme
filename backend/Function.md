# Function Documentation

## How the Backend Works

This document explains the technical implementation of the emotion-to-meme pipeline.

---

## Stage 1: Emotion Detection

### **Location:** `services/emotion_detector.py`

### **Purpose**
Detects emotion from a user's face image using the `emotion_teller.pkl` ML model.

### **Input**
- Image file path (string): `"path/to/user/image.jpg"`
- Supported formats: JPG, PNG, GIF, WEBP

### **Processing Steps**

#### 1. Load Model
```python
from services.emotion_detector import EmotionDetector

detector = EmotionDetector()
# Automatically loads models/emotion_teller.pkl
```

#### 2. Preprocess Image
- Opens image file
- Converts to RGB (if needed)
- Resizes to 224×224 pixels
- Normalizes pixel values (0-255 → 0-1)
- Converts to NumPy array

#### 3. Model Prediction
```python
result = detector.detect_emotion("image.jpg")
# Model predicts: "happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"
```

#### 4. Validation
- Ensures predicted emotion is in supported list
- Falls back to "neutral" if invalid

### **Output**
```python
{
    "emotion": "happy",           # Detected emotion string
    "confidence": 0.85,           # Model confidence (0-1)
    "success": True,              # Detection succeeded
    "message": "Emotion detected: happy",
    "model_used": "emotion_teller.pkl"  # or "placeholder"
}
```

### **Usage Examples**

#### Method 1: Full Result
```python
from services.emotion_detector import EmotionDetector

detector = EmotionDetector()
result = detector.detect_emotion("user_image.jpg")

if result['success']:
    emotion = result['emotion']
    confidence = result['confidence']
    print(f"Detected: {emotion} ({confidence:.2%})")
```

#### Method 2: Simple String
```python
from services.emotion_detector import detect_emotion_from_image

emotion = detect_emotion_from_image("user_image.jpg")
print(emotion)  # "happy"
```

### **Model Integration**

#### Adding Your Model
1. Save your trained model:
   ```python
   import pickle
   with open('emotion_teller.pkl', 'wb') as f:
       pickle.dump(your_model, f)
   ```

2. Place in `models/emotion_teller.pkl`

3. Update preprocessing if needed (in `emotion_detector.py`):
   ```python
   # Change resize dimensions
   img = img.resize((48, 48))  # Default: (224, 224)
   
   # Change normalization
   img_array = img_array / 255.0  # Default: 0-1 range
   ```

#### Model Requirements
- **Input:** NumPy array (H, W, C)
- **Output:** Emotion string from supported list
- **Format:** Pickle (.pkl) file

### **Placeholder Mode**
If `emotion_teller.pkl` not found:
- Uses deterministic placeholder for testing
- Returns consistent emotions based on filename hash
- Allows pipeline testing without trained model

---

## Stage 2: Meme Selection

### **Location:** `services/politician_selector.py`

### **Purpose**
Selects a matching politician meme based on detected emotion and outputs it for display.

### **Input**
- Emotion string from Stage 1: `"happy"`, `"sad"`, etc.

### **Processing Steps**

#### 1. Initialize Selector
```python
from services.politician_selector import PoliticianSelector

selector = PoliticianSelector()
# Sets base path to assets/memes/
```

#### 2. Validate Emotion
- Checks if emotion is in supported list
- Returns error if invalid

#### 3. Locate Emotion Folder
```python
emotion_folder = assets/memes/{emotion}/
# Example: assets/memes/happy/
```

#### 4. Find Available Memes
- Scans folder for image files (.jpg, .jpeg, .png, .gif, .webp)
- Returns error if folder empty

#### 5. Random Selection
```python
import random
selected_meme = random.choice(available_memes)
# Example: politician_a.jpg
```

#### 6. Output Meme
- Copies selected meme to `outputs/final_meme.jpg`
- Preserves original file extension

### **Output**
```python
{
    "emotion": "happy",
    "meme_path": "assets/memes/happy/politician_a.jpg",  # Original
    "meme_filename": "politician_a.jpg",
    "success": True,
    "message": "Meme selected successfully",
    "output_path": "outputs/final_meme.jpg",  # Copy for display
    "output_success": True,
    "output_message": "Meme output to outputs/final_meme.jpg"
}
```

### **Usage Examples**

#### Method 1: Select + Output (Recommended)
```python
from services.politician_selector import select_and_output_meme

result = select_and_output_meme("happy")
print(result['output_path'])  # "outputs/final_meme.jpg"
```

#### Method 2: Manual Control
```python
from services.politician_selector import PoliticianSelector

selector = PoliticianSelector()

# Step 1: Select meme
selection = selector.select_meme("happy")

# Step 2: Output meme
if selection['success']:
    output = selector.output_meme(selection)
    print(output['output_path'])
```

#### Method 3: Display in Viewer
```python
from services.politician_selector import select_and_display_meme

result = select_and_display_meme("surprise")
# Opens meme in default image viewer
```

### **Folder Structure**
```
assets/memes/
├── happy/
│   ├── politician_a.jpg
│   ├── politician_b.jpg
│   └── ...
├── sad/
│   ├── politician_c.jpg
│   └── ...
├── angry/
├── neutral/
├── surprise/
├── fear/
└── disgust/
```

---

## Complete Pipeline Integration

### **Location:** `test_complete_pipeline.py` / `example_pipeline.py`

### **Full Workflow**

```python
from services.emotion_detector import EmotionDetector
from services.politician_selector import select_and_output_meme

# Initialize detector
detector = EmotionDetector()

# Stage 1: Detect emotion
user_image = "captured_frame.jpg"
emotion_result = detector.detect_emotion(user_image)

if emotion_result['success']:
    detected_emotion = emotion_result['emotion']
    confidence = emotion_result['confidence']
    
    print(f"Stage 1: Detected {detected_emotion} ({confidence:.2%})")
    
    # Stage 2: Select and output meme
    meme_result = select_and_output_meme(detected_emotion)
    
    if meme_result['success']:
        print(f"Stage 2: Meme ready at {meme_result['output_path']}")
        
        # Frontend can now display outputs/final_meme.jpg
```

### **Pipeline Flow Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER INPUT                                               │
│    - Captured image from video feed                         │
│    - Saved as temporary file                                │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. STAGE 1: EMOTION DETECTION                               │
│    File: services/emotion_detector.py                       │
│                                                              │
│    ┌──────────────────────────────────────────────┐        │
│    │ Load emotion_teller.pkl model                │        │
│    └──────────────┬───────────────────────────────┘        │
│                   ↓                                          │
│    ┌──────────────────────────────────────────────┐        │
│    │ Preprocess: Resize → Normalize → Array       │        │
│    └──────────────┬───────────────────────────────┘        │
│                   ↓                                          │
│    ┌──────────────────────────────────────────────┐        │
│    │ Model Prediction → Emotion String            │        │
│    └──────────────┬───────────────────────────────┘        │
│                   ↓                                          │
│    Output: {"emotion": "happy", "confidence": 0.85}         │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. STAGE 2: MEME SELECTION                                  │
│    File: services/politician_selector.py                    │
│                                                              │
│    ┌──────────────────────────────────────────────┐        │
│    │ Receive emotion string: "happy"              │        │
│    └──────────────┬───────────────────────────────┘        │
│                   ↓                                          │
│    ┌──────────────────────────────────────────────┐        │
│    │ Scan assets/memes/happy/ folder              │        │
│    └──────────────┬───────────────────────────────┘        │
│                   ↓                                          │
│    ┌──────────────────────────────────────────────┐        │
│    │ Random selection from available memes        │        │
│    └──────────────┬───────────────────────────────┘        │
│                   ↓                                          │
│    ┌──────────────────────────────────────────────┐        │
│    │ Copy to outputs/final_meme.jpg               │        │
│    └──────────────┬───────────────────────────────┘        │
│                   ↓                                          │
│    Output: {"output_path": "outputs/final_meme.jpg"}        │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. OUTPUT                                                   │
│    - Meme image ready at outputs/final_meme.jpg            │
│    - Frontend displays the meme to user                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Wrapper Services

### **Location:** `services/meme_retrieval.py`

Provides convenience functions for easier integration:

```python
from services.meme_retrieval import (
    get_meme_for_emotion,      # Select meme with optional output
    get_available_emotions,    # List emotions with memes
    output_meme_to_folder      # Custom output location
)

# Simple usage
result = get_meme_for_emotion("happy", output=True)

# Check available emotions
emotions = get_available_emotions()
print(emotions)  # ["happy", "sad", "angry", ...]
```

---

## Error Handling

### **Stage 1 Errors**

| Error | Cause | Solution |
|-------|-------|----------|
| Image not found | Invalid file path | Check file exists |
| Failed to preprocess | Corrupted image | Verify image format |
| Model not found | Missing .pkl file | Add emotion_teller.pkl to models/ |

### **Stage 2 Errors**

| Error | Cause | Solution |
|-------|-------|----------|
| Unsupported emotion | Invalid emotion string | Use supported emotions |
| No meme images found | Empty folder | Add meme images to emotion folder |
| Emotion folder not found | Missing subfolder | Create emotion subfolders |

---

## Testing

### **Test Individual Stages**

```bash
# Test Stage 1
python -c "from services.emotion_detector import detect_emotion_from_image; print(detect_emotion_from_image('assets/memes/happy/politician_a.jpg'))"

# Test Stage 2
python -c "from services.politician_selector import select_and_output_meme; print(select_and_output_meme('happy'))"
```

### **Test Complete Pipeline**

```bash
python test_complete_pipeline.py
```

### **Test with Examples**

```bash
python example_pipeline.py
```

---

## Configuration

### **Preprocessing Settings**
Edit `services/emotion_detector.py`:
```python
# Line ~55: Change image dimensions
img = img.resize((224, 224))  # Adjust for your model

# Line ~63: Change normalization
img_array = img_array / 255.0  # 0-1 range
```

### **Output Location**
Edit `services/politician_selector.py`:
```python
# Line ~120: Change default output path
output_path = outputs_dir / "final_meme.jpg"  # Customize filename
```

### **Supported Emotions**
Edit both files to match:
- `services/emotion_detector.py` → Line ~25
- `services/politician_selector.py` → Line ~18

```python
self.supported_emotions = [
    "happy", "sad", "angry", "neutral",
    "surprise", "fear", "disgust"
]
```

---

## Performance

- **Stage 1**: ~50-200ms (depends on model complexity)
- **Stage 2**: ~5-10ms (file system operations)
- **Total Pipeline**: ~100-300ms (acceptable for real-time)

### **Optimization Tips**
1. Load `EmotionDetector()` once and reuse
2. Cache preprocessed images if processing multiple times
3. Use GPU for model inference (if available)
4. Optimize model size (quantization, pruning)

---

## Dependencies

See `requirements.txt`:
```
fastapi          # Web framework
uvicorn          # ASGI server
pillow           # Image processing
opencv-python    # Computer vision
numpy            # Array operations
scikit-learn     # ML utilities
tensorflow       # Deep learning (optional)
```

---

## Next Steps

1. **Add your model**: Place `emotion_teller.pkl` in `models/`
2. **Add meme images**: Populate `assets/memes/{emotion}/` folders
3. **Create API endpoints**: Implement `routes/capture_routes.py`
4. **Connect frontend**: Integrate video capture with backend API
5. **Deploy**: Host on server for production use
