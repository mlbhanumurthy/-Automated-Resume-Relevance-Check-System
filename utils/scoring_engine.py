import numpy as np
import re

class HybridScoringEngine:
    def __init__(self, job_description: str):
        # Save job description (in lowercase for matching)
        self.job_description = job_description.lower()

    def compute_score(self, resume_text: str) -> float:
        """
        Simple hybrid scoring:
        - Extracts words from job description & resume
        - Finds overlap
        - Computes percentage score
        """
        resume_text = resume_text.lower()

        # Split into words (alphanumeric only)
        jd_words = set(re.findall(r"\w+", self.job_description))
        resume_words = set(re.findall(r"\w+", resume_text))

        # Avoid division by zero
        if not jd_words:
            return 0.0

        overlap = jd_words.intersection(resume_words)
        score = len(overlap) / len(jd_words) * 100

        return round(score, 2)
