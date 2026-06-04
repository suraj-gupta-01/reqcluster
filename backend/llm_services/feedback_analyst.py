"""Feedback comment analyst and narrative updater.

Evaluates user correction annotations using keyword overlap analysis and
regenerates cluster descriptions after modifications.
"""

from __future__ import annotations

import re
from typing import List, Optional


class FeedbackAnalyst:
    """Analyzes human-in-the-loop annotations and comments."""

    def __init__(self, provider_name: str = "mock"):
        self.provider_name = provider_name

    def evaluate_comment_confidence(
        self,
        comment: str,
        target_cluster_label: str,
        target_cluster_keywords: List[str],
    ) -> float:
        """Compute a confidence score in [0.5, 1.0] based on comment overlap.

        If a comment is empty, returns 1.0 as a baseline. Otherwise, assesses
        if the comment references target keywords or cluster labels.
        """
        if not comment or not comment.strip():
            return 1.0

        comment_clean = comment.lower()
        keywords_clean = [kw.lower() for kw in target_cluster_keywords]
        label_words = re.findall(r"\w+", target_cluster_label.lower())

        match_count = 0

        # Check for keyword matches
        for kw in keywords_clean:
            if kw in comment_clean:
                match_count += 2
            else:
                for word in re.findall(r"\w+", kw):
                    if len(word) > 3 and word in comment_clean:
                        match_count += 1

        # Check for label word matches
        for word in label_words:
            if len(word) > 3 and word in comment_clean:
                match_count += 2

        # Score is bounded between 0.50 (low semantic match) and 1.00 (high match)
        confidence = 0.5 + min(match_count * 0.1, 0.5)
        return round(float(confidence), 2)

    def generate_updated_narrative(
        self,
        current_summary: str,
        added_text: Optional[str] = None,
        removed_text: Optional[str] = None,
    ) -> str:
        """Produce an updated summary paragraph highlighting user refinement."""
        summary = (current_summary or "").strip()

        if not summary:
            summary = "This cluster contains functionally related requirements."

        suffix = " (Manually refined by expert analyst)."
        if suffix not in summary:
            summary += suffix

        return summary
