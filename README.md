# DEEPGUARD AI
## Multimodal AI-Based Deepfake Detection Using Audio-Visual and Temporal Features

DeepGuard AI is an end-to-end deepfake detection system consisting of:
1. **PyTorch AI Engine & FastAPI Backend**: 4 distinct neural network branches + multimodal fusion layer.
2. **Flutter Android Application**: Premium mobile UI for selecting videos, streaming progress, inspecting branch probabilities, and reviewing detection history.

---

## 🏛️ System Architecture

```
                      ┌──────────────────────────────────────────────┐
                      │                 Input Video                  │
                      └──────────────────────┬───────────────────────┘
                                             │
             ┌────────────────┬──────────────┴───────────────┬────────────────┐
             ▼                ▼                              ▼                ▼
     ┌───────────────┐ ┌───────────────┐             ┌───────────────┐ ┌───────────────┐
     │ Visual Branch │ │ Audio Branch  │             │Lip-Sync Branch│ │Temporal Branch│
     │ MobileNetV3   │ │ 128-d MFCC    │             │SyncNet BiLSTM │ │Frame ResNet   │
     │ Fine-tuned    │ │ Spectral MLP  │             │with Attention │ │+ Bidirectional│
     │ Face Cropper  │ │ Classifier    │             │               │ │LSTM           │
     └───────┬───────┘ └───────┬───────┘             └───────┬───────┘ └───────┬───────┘
             │                 │                             │                 │
             └────────────────►├─────────────────────────────┴─────────────────┘
                               │ (4 Branch Scores)
                               ▼
                   ┌───────────────────────┐
                   │ Multimodal Fusion MLP │
                   │  (4 → 64 → 32 → 1)    │
                   └───────────┬───────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       Real Probability            AI-Generated Probability
               │                               │
               └───────────────┬───────────────┘
                               ▼
                     [ FINAL PREDICTION ]
```

---

## 🚀 1. Setup & Environment

### Requirements
- Python 3.10+
- PyTorch & Torchvision
- OpenCV, Librosa, Scikit-Learn, FastAPI, Uvicorn

```powershell
cd backend
pip install -r requirements.txt
```

---

## 🧠 2. Training the 4 AI Branches

The pipeline includes full dataset preparation and training scripts in `backend/train/`:

### Step 1: Prepare Dataset
```powershell
python train/prepare_dataset.py --source /path/to/dataset --out_dir ./data
```

### Step 2: Train Visual Model (MobileNetV3)
```powershell
python train/train_visual.py --data_dir ./data --epochs 10 --save_path ./models/visual_model.pth
```

### Step 3: Train Audio Model (MFCC Spectral MLP)
```powershell
python train/train_audio.py --data_dir ./data --epochs 25 --save_path ./models/audio_model.pth
```

### Step 4: Train Lip-Sync Model (Attention BiLSTM)
```powershell
python train/train_lip_sync.py --data_dir ./data --epochs 15 --save_path ./models/lip_sync_model.pth
```

### Step 5: Train Temporal Model (Sequential Consistency LSTM)
```powershell
python train/train_temporal.py --data_dir ./data --epochs 15 --save_path ./models/temporal_model.pth
```

### Step 6: Train Fusion Model
```powershell
python train/train_fusion.py --data_dir ./data --models_dir ./models --epochs 30 --save_path ./models/fusion_model.pth
```

### Step 7: Comprehensive Evaluation
```powershell
python train/evaluate.py --data_dir ./data --models_dir ./models
```

---

## 🧪 3. CLI End-to-End Pipeline Test

Run end-to-end detection directly from the command line:

```powershell
cd backend
python test_pipeline.py --video sample_test.mp4 --models_dir ./models
```

---

## 🌐 4. Running the FastAPI Backend

```powershell
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
`http://localhost:8000/docs`

Key Endpoints:
- `POST /analyze`: Upload MP4 video, run 4 branches + fusion, persist to SQLite, and return report.
- `GET /history`: List recent analysis reports with pagination.
- `GET /analysis/{id}`: Detailed report by ID.
- `DELETE /analysis/{id}`: Delete an analysis report.
- `GET /health`: Server & model status.

---

## 📱 5. Flutter Android App

### Configuration
Update `mobile/lib/config/app_config.dart` with your PC's IP address:
```dart
static const String apiBaseUrl = 'http://192.168.1.100:8000'; // or 10.0.2.2 for Android emulator
```

### Run on Device or Emulator
```powershell
cd mobile
flutter pub get
flutter run
```

### Build APK
```powershell
flutter build apk --release
```
The output APK is generated at `mobile/build/app/outputs/flutter-apk/app-release.apk`.
