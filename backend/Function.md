# Function Documentation

## How the Backend Works

This document explains the technical implementation of the emotion-to-meme pi#### Model Requirements:
- **Input:** PyTorch Tensor (1, 3, 224, 224) - Batch of 1, RGB, 224x224
- **Output:** Tensor (1, 7) - Logits for 7 emotion classes
- **Format:** PyTorch state dictionary (.pth) file
- **Device:** Auto-detects CUDA/CPU
- **Precision:** Uses torch.amp.autocast for mixed precision

### **Placeholder Mode**
If `emotion_teller.pth` not found:

---

## Stage 1: Emotion Detection

### **Location:** `services/emotion_detector.py`

### **Purpose**
Detects emotion from a user's face image using the **MobileNetV2** model trained on FER2013 dataset.

### **Model Architecture**
- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Custom Classifier**:
  - Dropout(0.5) → Linear(1280→512) → ReLU → BatchNorm1d
  - Dropout(0.4) → Linear(512→256) → ReLU → BatchNorm1d  
  - Dropout(0.3) → Linear(256→7) [7 emotion classes]
- **Training Dataset**: FER2013 (Facial Expression Recognition)
- **Classes**: angry, disgust, fear, happy, neutral, sad, surprise

### **Input**
- Image file path (string): `"path/to/user/image.png"`
- Supported formats: PNG (any RGB/Grayscale image)
- Image is automatically converted to grayscale with 3 channels

### **Processing Steps**

#### 1. Load Model
```python
from services.emotion_detector import EmotionDetector

detector = EmotionDetector()
# Automatically loads models/emotion_teller.pth
```

#### 2. Preprocess Image
- Opens image file and converts to RGB
- Converts to grayscale (3 channels for MobileNetV2 compatibility)
- Resizes to 224×224 pixels
- Converts to PyTorch tensor
- Normalizes using ImageNet statistics:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

#### 3. Model Prediction
```python
result = detector.detect_emotion("image.png")
# MobileNetV2 predicts: "happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"
```

**Inference Process:**
- Forward pass through MobileNetV2
- Apply Softmax to get probabilities
- Select class with highest probability
- Return emotion label and confidence score

#### 4. Validation
- Ensures predicted emotion is in supported list
- Falls back to "neutral" if invalid
- Handles errors gracefully with placeholder mode

### **Output**
```python
{
    "emotion": "happy",           # Detected emotion string
    "confidence": 0.85,           # Model confidence (0-1)
    "success": True,              # Detection succeeded
    "message": "Emotion detected: happy",
    "model_used": "emotion_teller.pth"  # or "placeholder"
}
```

### **Usage Examples**

#### Method 1: Full Result
```python
from services.emotion_detector import EmotionDetector

detector = EmotionDetector()
result = detector.detect_emotion("user_image.png")

if result['success']:
    emotion = result['emotion']
    confidence = result['confidence']
    print(f"Detected: {emotion} ({confidence:.2%})")
```

#### Method 2: Simple String
```python
from services.emotion_detector import detect_emotion_from_image

emotion = detect_emotion_from_image("user_image.png")
print(emotion)  # "happy"
```

### **Model Integration**

#### Model File Format
The model file `emotion_teller.pth` should contain the **state dictionary** of the trained MobileNetV2 model.

**Training Script (save model):**
```python
import torch
import torch.nn as nn
from torchvision import models

# After training your model
torch.save(model.state_dict(), 'emotion_teller.pth')
```

**Model Architecture (must match exactly):**
```python
model = models.mobilenet_v2(weights=None)
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
    nn.Linear(256, 7)  # 7 emotion classes
)
```

**Place model file at:** `backend/models/emotion_teller.pth`

#### Model Requirements
- **Input:** NumPy array (H, W, C)
- **Output:** Emotion string from supported list
- **Format:** PyTorch (.pth) file

### **Placeholder Mode**
If `emotion_teller.pth` not found:
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
- Scans folder for PNG image files
- Returns error if folder empty

#### 5. Random Selection
```python
import random
selected_meme = random.choice(available_memes)
# Example: politician_a.png
```

#### 6. Output Meme
- Copies selected meme to `outputs/final_meme.png`

### **Output**
```python
{
    "emotion": "happy",
    "meme_path": "assets/memes/happy/politician_a.png",  # Original
    "meme_filename": "politician_a.png",
    "success": True,
    "message": "Meme selected successfully",
    "output_path": "outputs/final_meme.png",  # Copy for display
    "output_success": True,
    "output_message": "Meme output to outputs/final_meme.png"
}
```

### **Usage Examples**

#### Method 1: Select + Output (Recommended)
```python
from services.politician_selector import select_and_output_meme

result = select_and_output_meme("happy")
print(result['output_path'])  # "outputs/final_meme.png"
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
│   ├── politician_a.png
│   ├── politician_b.png
│   └── ...
├── sad/
│   ├── politician_c.png
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
user_image = "captured_frame.png"
emotion_result = detector.detect_emotion(user_image)

if emotion_result['success']:
    detected_emotion = emotion_result['emotion']
    confidence = emotion_result['confidence']
    
    print(f"Stage 1: Detected {detected_emotion} ({confidence:.2%})")
    
    # Stage 2: Select and output meme
    meme_result = select_and_output_meme(detected_emotion)
    
    if meme_result['success']:
        print(f"Stage 2: Meme ready at {meme_result['output_path']}")
        
        # Frontend can now display outputs/final_meme.png
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
│    │ Load emotion_teller.pth model                │        │
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
│    │ Copy to outputs/final_meme.png               │        │
│    └──────────────┬───────────────────────────────┘        │
│                   ↓                                          │
│    Output: {"output_path": "outputs/final_meme.png"}        │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. OUTPUT                                                   │
│    - Meme image ready at outputs/final_meme.png            │
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
| Model not found | Missing .pth file | Add emotion_teller.pth to models/ |

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
python -c "from services.emotion_detector import detect_emotion_from_image; print(detect_emotion_from_image('assets/memes/happy/politician_a.png'))"

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
output_path = outputs_dir / "final_meme.png"  # Customize filename
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

1. **Add your model**: Place `emotion_teller.pth` in `models/`
2. **Add meme images**: Populate `assets/memes/{emotion}/` folders
3. **Create API endpoints**: Implement `routes/capture_routes.py`
4. **Connect frontend**: Integrate video capture with backend API
5. **Deploy**: Host on server for production use
