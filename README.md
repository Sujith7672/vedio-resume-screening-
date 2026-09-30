# AI-Powered Video Resume Screening System

A multi-modal AI system that automatically screens candidate video resumes by analyzing **speech content**, **spoken confidence**, and **visual engagement** — combining NLP, Speech Recognition, and Computer Vision to generate a candidate ranking score.

## Overview

Traditional resume screening only looks at text. This project goes a step further by analyzing **video resumes / interview recordings** to evaluate candidates holistically:

- What they say (skills, relevance to job description)
- How they say it (confidence, fluency, filler words)
- How they present themselves (eye contact, facial engagement)

This is a **multi-modal AI pipeline** combining Audio, Text, and Video processing.

## Features

- Automatic audio extraction from video resumes
- Speech-to-text transcription using OpenAI Whisper
- NLP-based skill extraction & job description matching
- Confidence/fluency scoring from speech patterns
- Facial engagement analysis (eye contact, expression variety) using MediaPipe
- Weighted final score & candidate ranking
- CSV report generation for recruiters

## Tech Stack

| Component | Technology |
|---|---|
| Speech-to-Text | OpenAI Whisper |
| NLP / Text Analysis | spaCy, NLTK, Sentence-BERT |
| Video/Facial Analysis | OpenCV, MediaPipe |
| Audio Extraction | moviepy |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib |

## Project Workflow

1. Load candidate video resume(s)
2. Extract audio track from video
3. Transcribe audio to text using Whisper
4. Analyze transcript: extract skills, compute JD-similarity, confidence score
5. Analyze video frames: eye contact %, expression variety, engagement score
6. Compute weighted final score
7. Rank candidates and export report

## Project Structure

```
video-resume-screening/
│
├── app.py                          (Streamlit web frontend — upload & analyze)
├── Video_Resume_Screening.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── sample_data/
│   └── sample_resumes.csv
├── output/
│   └── ranked_candidates.csv
└── src/
    ├── audio_extractor.py
    ├── speech_to_text.py
    ├── text_analysis.py
    ├── video_analysis.py
    └── scoring.py
```

## Quick Start (Web UI)

No filename editing needed — upload your video directly in the browser:

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens a local web page where you can:
1. Paste/edit the job description
2. Upload a candidate video resume (mp4/mov/avi/mkv)
3. Click **Analyze Video** — transcript, skills, confidence, video engagement, and final score appear automatically

## Installation

```bash
git clone https://github.com/<your-username>/video-resume-screening.git
cd video-resume-screening
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Note: Whisper also requires `ffmpeg` installed on your system:
```bash
sudo apt install ffmpeg   # Linux
brew install ffmpeg       # Mac
```

## How It Works

### 1. Audio Extraction
`moviepy` extracts the audio track (.wav) from each candidate's video file.

### 2. Speech-to-Text (Whisper)
The Whisper `base` model transcribes speech to text, handling accents and mild background noise well.

### 3. Text Analysis
- Skill keywords extracted from the transcript
- Sentence-BERT embeddings compare transcript relevance to the job description
- Filler-word ratio (um, uh, like) used as a simple confidence/fluency signal

### 4. Video/Facial Analysis
OpenCV's Haar Cascade face detector tracks face presence and position per frame to estimate:
- Face-presence percentage (how consistently the candidate is visible/facing the camera)
- Expression variety (variation in face position/movement across frames)
- Overall engagement score

### 5. Final Scoring
```
Final Score = 0.4 * Text_Relevance_Score
            + 0.3 * Confidence_Score
            + 0.3 * Video_Engagement_Score
```

## Future Improvements

- Streamlit web dashboard for recruiters
- Fine-tuned BERT model for domain-specific confidence detection
- Real-time webcam interview analysis
- Bias/fairness auditing of scoring model
- Multi-language support

## Learning Outcomes

- Multi-modal AI pipeline design (audio + text + video)
- Speech recognition with Whisper
- Facial landmark analysis with MediaPipe
- NLP-based semantic similarity (Sentence-BERT)
- Weighted scoring system design

## Use Cases

- Video resume screening automation
- Interview pre-screening for recruiters
- HR analytics & candidate engagement scoring

## Author

Sujith

## License

This project is for educational and learning purposes.
