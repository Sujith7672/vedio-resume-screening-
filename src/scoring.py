"""
scoring.py
Combines text and video scores into a final weighted candidate score,
and ranks candidates.
"""

import pandas as pd


def compute_final_score(
    jd_similarity: float,
    confidence: float,
    video_engagement: float,
    weights: tuple = (0.4, 0.3, 0.3),
) -> float:
    """
    Computes the weighted final score for a candidate.

    Args:
        jd_similarity: relevance of transcript to job description (0-1)
        confidence: speech confidence/fluency score (0-1)
        video_engagement: facial engagement score (0-1)
        weights: (text_weight, confidence_weight, video_weight)

    Returns:
        final weighted score (0-1)
    """
    w1, w2, w3 = weights
    score = w1 * jd_similarity + w2 * confidence + w3 * video_engagement
    return round(score, 3)


def rank_candidates(candidates: list) -> pd.DataFrame:
    """
    Ranks a list of candidate score dicts by final_score descending.

    Args:
        candidates: list of dicts, each with keys like
            'name', 'jd_similarity', 'confidence', 'video_engagement'

    Returns:
        DataFrame sorted by final_score, with a 'rank' column
    """
    df = pd.DataFrame(candidates)
    df["final_score"] = df.apply(
        lambda row: compute_final_score(
            row["jd_similarity"], row["confidence"], row["video_engagement"]
        ),
        axis=1,
    )
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


if __name__ == "__main__":
    sample = [
        {"name": "Candidate A", "jd_similarity": 0.8, "confidence": 0.7, "video_engagement": 0.75},
        {"name": "Candidate B", "jd_similarity": 0.6, "confidence": 0.9, "video_engagement": 0.5},
    ]
    result = rank_candidates(sample)
    print(result)
