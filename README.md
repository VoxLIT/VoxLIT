# VoxLIT — Learning Interpretability Tool for Voice Models

<p align="center">
  <a href="https://github.com/VoxLIT/VoxLIT">
    <img src="https://img.shields.io/badge/version-v1.0-blue" alt="Version"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"/>
  </a>
</p>

Interpreting how deep learning models make decisions is crucial, especially in high-stakes applications like speech recognition, emotion detection, and speaker identification. While the Learning Interpretability Tool (LIT) enables exploration of text and tabular models, there's a lack of equivalent tools for voice-based models. Voice data poses additional challenges due to its temporal nature and multi-modal representations (e.g., waveform, spectrogram).

VoxLIT extends the interpretability paradigm to audio models, providing researchers and developers with tools to analyze and debug speech models with greater transparency. Through interactive visualizations, attention mechanisms, and perturbation analyses, you can gain deeper insights into how your audio models make decisions.

VoxLIT is organized as a homepage plus **five task workbenches**, all active. Tasks, their models, and their datasets are configured centrally — see [STRUCTURE.md](STRUCTURE.md) for the repository layout, the task registry, and how to add a new task, model, or dataset.

| Task | Models | Dataset |
| --- | --- | --- |
| Speech Transcription | Whisper base, Whisper large-v3 | Common Voice |
| Emotion Recognition | wav2vec2 | RAVDESS |
| Speaker Verification | ECAPA-TDNN, ResNet34-LM | VoxCeleb1 demo subset |
| Speaker Diarization | pyannote 3.1, Reverb v1, Reverb v2 | AMI subset |
| Audio Deepfake Detection | wav2vec2 XLS-R, AST, XLSR-Mamba, Wav2Vec2-AASIST, XLSR-SLS, Nes2Net-X | ASVspoof 2019 LA subset |

## Features

- **Audio Data Management**: Upload and manage audio datasets with metadata
- **Waveform Visualization**: Interactive waveform viewer with playback controls
- **Model Prediction Analysis**: Examine model predictions and confidence scores
- **Attention Visualization**: Explore attention patterns in transformer-based audio models
- **Embedding Analysis**: Visualize high-dimensional audio embeddings in 2D/3D space
- **Saliency Mapping**: Identify important regions in audio input using gradient-based methods
- **Perturbation Tools**: Apply various audio perturbations to test model robustness
- **Interactive Dashboard**: Comprehensive interface for exploring model behavior
- **Faithful Emotion Recognition**: Loads the wav2vec2 emotion model with its trained classifier head via a custom model class (the stock `Wav2Vec2ForSequenceClassification` loader silently re-initializes the head, producing random predictions)

## Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **UI Framework**: Tailwind CSS + shadcn/ui components
- **State Management**: TanStack Query
- **Routing**: React Router
- **Data Visualization**: Recharts, Plotly.js, and Three.js (via react-three-fiber) for 3D embedding views
- **Animation**: Motion
- **Audio Processing**: Web Audio API + wavesurfer.js
- **Backend**: FastAPI + Python 3.11
- **ML**: PyTorch, Hugging Face Transformers, pyannote.audio, Captum, scikit-learn, UMAP, librosa
- **Models**: Whisper, wav2vec2 / XLS-R, ECAPA-TDNN, pyannote, Audio Spectrogram Transformer, and XLS-R-based deepfake detectors
- **Storage**: Redis for caching predictions and results
- **Testing**: Vitest (frontend) and pytest (backend)
- **Deployment**: Modal (frontend + backend + Redis in one container, behind nginx)

## Prerequisites

- **Frontend**:
  - Node.js (v18 or higher)
  - npm or bun package manager

- **Backend**:
  - Python 3.11
  - Docker Desktop (for Redis)
  - A Hugging Face access token, for the gated pyannote diarization models

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/VoxLIT/VoxLIT.git
cd VoxLIT
```

### 2. Set up the Frontend

```bash
cd Frontend
npm install
npm run dev
```

The frontend dev server runs at http://localhost:8080.

### 3. Start Redis in Docker

```bash
# In a new terminal (Docker Desktop must be running)
cd Backend
docker compose up -d
```

The backend defaults to `redis://localhost:6379/0`, so no `.env` is needed for local development. To override settings (e.g., a remote Redis), create `Backend/.env` with `REDIS_URL=...`. Speaker Diarization uses gated pyannote models: accept their terms on Hugging Face and add `HF_TOKEN=...` to `Backend/.env`.

### 4. Set up the Backend

> **Important:** always start uvicorn **from the `Backend/` directory** — the app resolves `data/` and `uploads/` relative to the working directory.

```bash
cd Backend
py -3.11 -m venv .venv          # or: python3.11 -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Unix / macOS
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API runs at http://localhost:8000.

Model weights are downloaded from Hugging Face and cached under `~/.cache/huggingface`. Whisper and wav2vec2 load on first startup, which takes a few minutes; the other tasks' models download the first time they are used, and `whisper-large-v3` (~3 GB) only if selected in the UI. The deepfake detectors are about 1.3 GB each.

### 5. Access the Application

Open your browser and navigate to [http://localhost:8080](http://localhost:8080)

## Project Structure

```
VoxLIT/
├── Frontend/                    # React frontend application
│   └── src/
│       ├── components/          # Shared React components
│       │   ├── analysis/        # Analysis and perturbation tools
│       │   ├── audio/           # Audio visualization components
│       │   ├── dataset/         # Dataset management
│       │   ├── panels/          # Dashboard panels
│       │   ├── site/            # Homepage and site chrome
│       │   ├── ui/              # Reusable UI components (shadcn/ui)
│       │   ├── visualization/   # Data visualization components
│       │   └── workbench/       # Shared task-workbench layout
│       ├── features/            # One folder per task workbench
│       │   ├── transcription/
│       │   ├── emotion/
│       │   ├── verification/
│       │   ├── task-b/          # Speaker diarization
│       │   └── deepfake/
│       ├── tasks/               # Frontend task registry
│       ├── contexts/            # React contexts
│       ├── hooks/               # Custom React hooks
│       ├── lib/                 # API client and utilities
│       └── pages/               # Home, task page, help portal
│
├── Backend/                     # FastAPI backend application
│   ├── app/                     # Application code
│   │   ├── api/                 # Shared API routes and endpoints
│   │   ├── core/                # Settings and core functionality
│   │   ├── services/            # Shared business logic services
│   │   └── tasks/               # Per-task routers, models and services + task registry
│   ├── data/                    # Sample datasets (git-ignored; add your own)
│   ├── scripts/                 # Dataset download and preparation scripts
│   ├── tests/                   # Backend tests
│   └── uploads/                 # User-uploaded audio files
│
├── deploy/                      # Modal deployment (app, nginx config, model checks)
├── STRUCTURE.md                 # Repository layout and task registry guide
├── CHANGELOG.md                 # Release notes
├── CODE_OF_CONDUCT.md           # Community guidelines
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # MIT License
├── README.md                    # Project documentation
└── SECURITY.md                  # Security policy
```

## Available Scripts

### Frontend

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run preview` - Preview production build
- `npm test` - Run frontend tests (Vitest)

### Backend

- `pytest` - Run backend tests
- `uvicorn app.main:app --reload` - Start the API server in development mode

## Usage

1. **Pick a Task**: Choose a workbench from the homepage
2. **Load Audio**: Use the task's built-in dataset or upload your own audio files
3. **Select Models**: Choose from the task's available models
4. **Explore Visualizations**:
   - Examine waveforms and spectrograms
   - View model predictions and confidence scores
   - Explore attention patterns and embedding spaces
   - Generate saliency maps to highlight important audio regions
5. **Apply Perturbations**: Test model robustness with various audio perturbations
6. **Analyze Results**: Use the interactive dashboard to gain insights

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## Security

For security-related issues, please refer to our [Security Policy](SECURITY.md).

## Acknowledgments

- Inspired by Google's [Learning Interpretability Tool (LIT)](https://github.com/PAIR-code/lit)
- Built with modern React ecosystem and TypeScript
- Special thanks to the open-source community for the amazing tools and libraries

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built for audio model interpretability</sub>
</p>



<!-- .venv/bin/uvicorn app.main:app --reload --reload-dir app
.venv/bin/python -m pytest tests/test_diarization_*.py -q -->