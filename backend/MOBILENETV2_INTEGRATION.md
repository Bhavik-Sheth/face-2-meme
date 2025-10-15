# MobileNetV2 Model Integration Guide

## Overview
This project now uses **MobileNetV2** architecture for emotion detection, trained on the FER2013 dataset. The model has been fully integrated into the pipeline.

---

## Model Architecture

### Base Model
- **Architecture**: MobileNetV2
- **Pretrained**: ImageNet (during training)
- **Input Size**: 224×224×3 (Grayscale converted to 3 channels)

### Custom Classifier
```python
nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(1280, 512),        # MobileNetV2 last_channel = 1280
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.3),
    nn.Linear(256, 7)            # 7 emotion classes
)
```

### Emotion Classes (FER2013 order)
1. angry
2. disgust
3. fear
4. happy
5. neutral
6. sad
7. surprise

---

## File Structure
```
backend/
├── models/
│   └── emotion_teller.pth       ← Your trained model goes here
├── services/
│   └── emotion_detector.py      ← MobileNetV2 integration
├── test_mobilenetv2.py          ← Test script for model
└── requirements.txt             ← Updated with torch & torchvision
```

---

## Model File Format

### Option 1: State Dictionary (Recommended)
```python
# Save only the model weights (smaller file size)
torch.save(model.state_dict(), 'emotion_teller.pth')
```

### Option 2: Full Model
```python
# Save entire model (larger file size)
torch.save(model, 'emotion_teller.pth')
```

**Note**: The current code expects **state_dict format** and will reconstruct the architecture automatically.

---

## Preprocessing Pipeline

### Input Image
- **Format**: PNG, JPG, any image format
- **Color**: RGB or Grayscale (auto-converted)

### Transformations
```python
transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
    transforms.Resize((224, 224)),                 # Resize to 224×224
    transforms.ToTensor(),                         # Convert to tensor
    transforms.Normalize(                          # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## Inference Process

### 1. Load Model
```python
from services.emotion_detector import EmotionDetector

detector = EmotionDetector()  # Auto-loads from models/emotion_teller.pth
```

### 2. Detect Emotion
```python
result = detector.detect_emotion("path/to/image.png")

print(result)
# {
#     'emotion': 'happy',
#     'confidence': 0.9234,
#     'success': True,
#     'message': 'Emotion detected: happy',
#     'model_used': 'MobileNetV2'
# }
```

### 3. Get Emotion String (Quick Method)
```python
from services.emotion_detector import detect_emotion_from_image

emotion = detect_emotion_from_image("image.png")
print(emotion)  # 'happy'
```

---

## Device Support

### Automatic Detection
The model automatically detects and uses the best available device:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### GPU (CUDA)
- Faster inference
- Requires CUDA-compatible GPU
- Automatically used if available

### CPU
- Slower inference
- Works on any machine
- Fallback option

---

## Testing

### Run Model Integration Test
```bash
python test_mobilenetv2.py
```

### Run Complete Pipeline Test
```bash
python test_complete_pipeline.py
```

### Expected Output
```
✓ Loaded MobileNetV2 emotion model from models/emotion_teller.pth
Device: cuda (or cpu)
Model Loaded: True

Testing emotion detection:
  Emotion: happy
  Confidence: 0.9234
  Model Used: MobileNetV2
```

---

## Placeholder Mode

If `emotion_teller.pth` is **not found**, the system automatically enters **placeholder mode**:
- Uses deterministic hash-based predictions
- Allows testing the pipeline without the model
- Returns consistent emotions for the same filename

**To exit placeholder mode**: Place your trained model at `models/emotion_teller.pth`

---

## Troubleshooting

### Issue: "Model not found"
**Solution**: Ensure `emotion_teller.pth` exists at `backend/models/emotion_teller.pth`

### Issue: "Could not load model: CUDA error"
**Solution**: Add `map_location='cpu'` in your training script when saving:
```python
torch.save(model.state_dict(), 'emotion_teller.pth')
```

### Issue: "Import torchvision could not be resolved"
**Solution**: Install dependencies:
```bash
pip install torch torchvision
```

### Issue: "Size mismatch for classifier"
**Solution**: Ensure your model architecture **exactly matches** the custom classifier defined in `emotion_detector.py`

### Issue: Wrong class order
**Solution**: Verify class names match FER2013 order:
```python
['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
```

---

## Performance Optimization

### Mixed Precision (Automatic)
The code uses `torch.amp.autocast` for mixed precision inference:
- Faster on GPU
- Lower memory usage
- Minimal accuracy impact

### Batch Processing (Future)
To process multiple images:
```python
# Modify preprocess_image to handle batches
images = [img1, img2, img3]
batch_tensor = torch.stack([transform(img) for img in images])
```

---

## Model Training Tips

### If you're training your own model:

1. **Match the architecture exactly**:
   - Use the same classifier structure
   - Save using `state_dict()`

2. **Use FER2013 class order**:
   - angry, disgust, fear, happy, neutral, sad, surprise

3. **Save the model**:
   ```python
   torch.save(model.state_dict(), 'emotion_teller.pth')
   ```

4. **Test immediately**:
   ```bash
   python test_mobilenetv2.py
   ```

---

## Integration with Stage 2

The emotion detector seamlessly integrates with Stage 2 (meme selection):

```python
from services.emotion_detector import detect_emotion_from_image
from services.politician_selector import select_and_output_meme

# Stage 1: Detect emotion
emotion = detect_emotion_from_image("user_image.png")

# Stage 2: Get matching meme
if emotion:
    result = select_and_output_meme(emotion)
    print(f"Meme ready at: {result['output_path']}")
```

---

## Next Steps

1. ✅ Place `emotion_teller.pth` in `models/` directory
2. ✅ Run `python test_mobilenetv2.py` to verify
3. ✅ Test with real face images
4. ✅ Integrate with FastAPI endpoints (coming soon)
5. ✅ Connect to frontend video capture

---

## References

- **MobileNetV2**: https://arxiv.org/abs/1801.04381
- **FER2013 Dataset**: https://www.kaggle.com/datasets/msambare/fer2013
- **PyTorch**: https://pytorch.org/docs/stable/index.html
- **TorchVision**: https://pytorch.org/vision/stable/index.html
