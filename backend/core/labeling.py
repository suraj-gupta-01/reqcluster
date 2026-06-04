import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import CountVectorizer
import re


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "must", "can", "need", "to", "of",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "out", "off", "over", "under", "again", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "but", "and", "or",
    "it", "its", "as", "this", "that", "these", "those", "which", "who",
    "system", "shall", "provide", "ensure", "support", "include", "allow",
    "enable", "within", "without", "using", "based", "per", "via", "including",
    "while", "also", "any", "applicable", "defined", "minimum", "maximum",
    "least", "greater", "less", "equal", "type", "value", "number", "set",
    "meet", "maintain", "required", "requirement",
}


def compute_ctfidf(
    texts_by_cluster: Dict[int, List[str]],
    top_n: int = 10,
) -> Dict[int, List[str]]:
    """
    Compute c-TF-IDF scores for each cluster.
    Returns top_n keywords per cluster.
    """
    cluster_ids = list(texts_by_cluster.keys())
    cluster_docs = [" ".join(texts) for texts in texts_by_cluster.values()]

    vectorizer = CountVectorizer(
        stop_words=None,
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
    )

    try:
        tf_matrix = vectorizer.fit_transform(cluster_docs)
    except ValueError:
        return {cid: [] for cid in cluster_ids}

    vocab = vectorizer.get_feature_names_out()
    tf = tf_matrix.toarray().astype(float)

    # Normalize TF by document length
    doc_lengths = tf.sum(axis=1, keepdims=True)
    doc_lengths[doc_lengths == 0] = 1
    tf_normalized = tf / doc_lengths

    # IDF: log(1 + N / df)
    n_clusters = len(cluster_docs)
    df = (tf > 0).sum(axis=0)
    idf = np.log(1 + n_clusters / (df + 1))

    ctfidf = tf_normalized * idf

    # Extract top keywords per cluster, filtering stopwords
    keywords_by_cluster = {}
    for i, cluster_id in enumerate(cluster_ids):
        scores = ctfidf[i]
        top_indices = scores.argsort()[::-1]
        keywords = []
        for idx in top_indices:
            word = vocab[idx]
            word_parts = word.split()
            # Filter stopwords (all parts must be non-stopword)
            if all(w.lower() not in STOPWORDS for w in word_parts):
                keywords.append(word)
            if len(keywords) >= top_n:
                break
        keywords_by_cluster[cluster_id] = keywords

    return keywords_by_cluster


def generate_cluster_label(keywords: List[str]) -> str:
    """
    Generate a human-readable label from top keywords.
    Uses top 3-4 keywords, title-cased.
    """
    if not keywords:
        return "Uncategorized"

    top_keywords = keywords[:4]
    
    # Capitalize each word
    label_parts = []
    for kw in top_keywords:
        label_parts.append(" ".join(w.capitalize() for w in kw.split()))

    return " ".join(label_parts)


def label_clusters(
    texts: List[str],
    labels: np.ndarray,
    top_n_keywords: int = 5,
) -> Dict[int, Dict]:
    """
    Generate labels and keywords for all clusters.
    
    Returns:
        Dict mapping cluster_id -> {label, keywords, size}
    """
    unique_labels = sorted(set(labels))
    cluster_ids = [l for l in unique_labels if l != -1]

    # Group texts by cluster
    texts_by_cluster = {}
    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]
        texts_by_cluster[cluster_id] = cluster_texts

    # Compute c-TF-IDF
    keywords_by_cluster = compute_ctfidf(texts_by_cluster, top_n=top_n_keywords)

    # Generate labels
    result = {}
    for cluster_id in cluster_ids:
        keywords = keywords_by_cluster.get(cluster_id, [])
        label = generate_cluster_label(keywords)
        size = int((labels == cluster_id).sum())
        # Convert numpy integer to Python int for proper SQLite storage
        result[int(cluster_id)] = {
            "label": label,
            "keywords": keywords[:top_n_keywords],
            "size": size,
        }

    return result
