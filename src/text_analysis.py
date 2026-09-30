"""
text_analysis.py
Analyzes transcript text: skill extraction, JD similarity, confidence scoring.
"""

from sentence_transformers import SentenceTransformer, util
import re

SKILL_KEYWORDS = [
    "python", "java", "sql", "machine learning", "deep learning",
    "nlp", "data analysis", "tensorflow", "pytorch", "react",
    "javascript", "aws", "docker", "kubernetes", "power bi",
    "tableau", "excel", "communication", "leadership", "teamwork"
]

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def extract_skills(transcript: str) -> list:
    """Extracts known technical/soft skills mentioned in the transcript."""
    text_lower = transcript.lower()
    found = [skill for skill in SKILL_KEYWORDS if skill in text_lower]
    return found


def jd_similarity_score(transcript: str, job_description: str) -> float:
    """
    Computes semantic similarity between transcript and job description
    using Sentence-BERT embeddings (0 to 1 scale).
    """
    model = _get_model()
    embeddings = model.encode([transcript, job_description], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(max(score, 0), 3)


def confidence_score(transcript: str, filler_count: int) -> float:
    """
    Simple rule-based confidence/fluency score (0 to 1 scale).
    Penalizes high filler-word ratio and very short answers.
    """
    words = re.findall(r"\w+", transcript)
    word_count = max(len(words), 1)
    filler_ratio = filler_count / word_count

    length_score = min(word_count / 150, 1.0)   # longer, substantive answers score higher
    fluency_score = max(1 - (filler_ratio * 10), 0)

    score = 0.5 * length_score + 0.5 * fluency_score
    return round(score, 3)


if __name__ == "__main__":
    sample_transcript = "I have experience in Python, machine learning and data analysis."
    jd = "Looking for a candidate skilled in Python, machine learning, and NLP."
    print("Skills:", extract_skills(sample_transcript))
    print("JD Similarity:", jd_similarity_score(sample_transcript, jd))
    print("Confidence Score:", confidence_score(sample_transcript, filler_count=1))
