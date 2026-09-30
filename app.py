"""
app.py
Streamlit frontend for the Video Resume Screening System.

Run with:
    streamlit run app.py

Then open the browser link it shows, upload a video, and get instant results.
"""

import streamlit as st
import tempfile
import os
import sys

sys.path.append("src")

from audio_extractor import extract_audio
from speech_to_text import transcribe_audio, count_filler_words
from text_analysis import extract_skills, jd_similarity_score, confidence_score
from video_analysis import video_engagement_score
from scoring import compute_final_score

st.set_page_config(page_title="Video Resume Screening", page_icon="🎥", layout="centered")

st.title("🎥 AI Video Resume Screening")
st.write("Upload a candidate's self-intro / video resume and get an instant AI-generated analysis.")

job_description = st.text_area(
    "Job Description",
    value="Looking for a candidate skilled in Python, machine learning, NLP, and strong communication.",
    height=100,
)

uploaded_file = st.file_uploader("Upload video resume", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Analyze Video"):
        with st.spinner("Saving uploaded video..."):
            # Save uploaded video to a temp file so our pipeline functions can read it
            suffix = os.path.splitext(uploaded_file.name)[1]
            tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_video.write(uploaded_file.read())
            tmp_video.close()
            video_path = tmp_video.name

        with st.spinner("Extracting audio..."):
            audio_path = extract_audio(video_path, output_dir=tempfile.gettempdir())

        with st.spinner("Transcribing speech (Whisper)... this can take a minute or two"):
            result = transcribe_audio(audio_path, model_size="small")
            transcript = result["text"]
            filler_count = count_filler_words(transcript)

        with st.spinner("Analyzing text..."):
            skills = extract_skills(transcript)
            similarity = jd_similarity_score(transcript, job_description)
            conf_score = confidence_score(transcript, filler_count)

        with st.spinner("Analyzing video engagement..."):
            engagement = video_engagement_score(video_path)

        final_score = compute_final_score(similarity, conf_score, engagement)

        st.success("Analysis complete!")

        st.subheader("📝 Transcript")
        st.write(transcript)

        col1, col2, col3 = st.columns(3)
        col1.metric("JD Similarity", f"{similarity:.2f}")
        col2.metric("Confidence Score", f"{conf_score:.2f}")
        col3.metric("Video Engagement", f"{engagement:.2f}")

        st.subheader("🎯 Final Score")
        st.metric("Overall Candidate Score", f"{final_score:.2f} / 1.00")
        st.progress(final_score)

        st.subheader("🛠️ Skills Detected")
        st.write(", ".join(skills) if skills else "No matching skills detected.")

        # Cleanup temp files
        os.remove(video_path)
        os.remove(audio_path)
