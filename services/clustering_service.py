from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    KMeans = None
    TfidfVectorizer = None

from services.embedding_service import EmbeddingService


def generate_cluster_label(cluster_papers: List[Dict[str, Any]]) -> str:
    """
    Extracts top informative terms from cluster papers to build a grounded thematic title.
    """
    if not cluster_papers:
        return "Miscellaneous Theme"

    texts = [
        f"{p.get('title', '')} {p.get('abstract', '') or ''}"
        for p in cluster_papers
    ]

    if not TfidfVectorizer:
        # Fallback to most frequent title words
        words = []
        for p in cluster_papers:
            words.extend(p.get("title", "").split())
        return "Theme: " + " ".join(words[:4]) if words else "Unassigned Theme"

    try:
        tfidf = TfidfVectorizer(stop_words="english", max_features=5, ngram_range=(1, 2))
        X = tfidf.fit_transform(texts)
        feature_names = tfidf.get_feature_names_out()
        if len(feature_names) > 0:
            top_terms = [t.title() for t in feature_names[:3]]
            return f"Theme: {' / '.join(top_terms)}"
    except Exception:
        pass

    first_title = cluster_papers[0].get("title", "Literature Topic")
    return f"Theme: {first_title[:40]}..."


def cluster_papers(
    papers: List[Dict[str, Any]],
    embedding_service: EmbeddingService,
    num_clusters: int = 3
) -> Dict[str, Any]:
    """
    Groups papers into thematic clusters using dense embeddings and KMeans clustering.
    Handles small collections safely and assigns grounded descriptive labels.
    """
    if not papers:
        return {"num_clusters": 0, "clusters": []}

    n_papers = len(papers)
    if n_papers == 1:
        return {
            "num_clusters": 1,
            "clusters": [
                {
                    "cluster_id": 0,
                    "label": generate_cluster_label(papers),
                    "paper_count": 1,
                    "papers": papers
                }
            ]
        }

    # Clamp k to number of papers
    effective_k = max(1, min(int(num_clusters), n_papers))

    # Generate embeddings for papers based on title and abstract
    paper_texts = [
        f"{p.get('title', '')}. {p.get('abstract', '') or ''}".strip()
        for p in papers
    ]
    embeddings = embedding_service.embed_texts(paper_texts)

    if KMeans is None or effective_k == 1:
        return {
            "num_clusters": 1,
            "clusters": [
                {
                    "cluster_id": 0,
                    "label": generate_cluster_label(papers),
                    "paper_count": n_papers,
                    "papers": papers
                }
            ]
        }

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(embeddings)

    clusters_dict: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(effective_k)}
    for idx, paper in enumerate(papers):
        assigned_cluster = int(cluster_labels[idx])
        clusters_dict[assigned_cluster].append(paper)

    result_clusters = []
    for c_id, c_papers in clusters_dict.items():
        if not c_papers:
            continue
        result_clusters.append({
            "cluster_id": c_id,
            "label": generate_cluster_label(c_papers),
            "paper_count": len(c_papers),
            "papers": c_papers
        })

    return {
        "num_clusters": len(result_clusters),
        "clusters": result_clusters
    }


class ClusteringService:
    """
    Object-oriented wrapper around KMeans paper clustering.
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None, num_clusters: int = 3):
        self.embedding_service = embedding_service
        self.num_clusters = num_clusters

    def cluster_papers(self, papers: List[Dict[str, Any]], num_clusters: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Clusters papers and returns the list of cluster dictionaries.
        """
        k = num_clusters if num_clusters is not None else self.num_clusters
        from services.embedding_service import EmbeddingService
        emb = self.embedding_service or EmbeddingService()
        res = cluster_papers(papers, embedding_service=emb, num_clusters=k)
        return res.get("clusters", [])
