"""Repository recommendations using semantic, graph, and hybrid scoring."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPOSITORIES_PATH = PROCESSED_DATA_DIR / "repositories_clean.csv"
EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "repository_embeddings.npy"
NODE2VEC_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "node2vec_embeddings.pkl"

BASE_RESULT_COLUMNS = [
    "Name",
    "Description",
    "Stars",
    "Forks",
    "Issues",
]
SEMANTIC_RESULT_COLUMNS = [
    *BASE_RESULT_COLUMNS,
    "semantic_similarity",
]
GRAPH_RESULT_COLUMNS = [
    *BASE_RESULT_COLUMNS,
    "graph_similarity",
]
HYBRID_RESULT_COLUMNS = [
    *BASE_RESULT_COLUMNS,
    "semantic_similarity",
    "graph_similarity",
    "popularity_score",
    "final_score",
]


def load_repositories(path: Path = REPOSITORIES_PATH) -> pd.DataFrame:
    """Load cleaned repository metadata."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed repository dataset not found: {path}. "
            "Run python src/data_preprocessing.py first."
        )

    return pd.read_csv(path)


def load_embeddings(path: Path = EMBEDDINGS_PATH) -> np.ndarray:
    """Load repository semantic embeddings."""
    if not path.exists():
        raise FileNotFoundError(
            f"Repository embeddings file not found: {path}. "
            "Run python src/embeddings.py first."
        )

    return np.load(path)


def normalize_node2vec_embeddings(raw_embeddings: Any) -> dict[str, np.ndarray]:
    """Normalize supported node2vec embedding formats to a node-to-vector mapping."""
    if hasattr(raw_embeddings, "wv"):
        raw_embeddings = raw_embeddings.wv

    if hasattr(raw_embeddings, "key_to_index"):
        return {
            str(node_name): np.asarray(raw_embeddings[node_name], dtype=float)
            for node_name in raw_embeddings.key_to_index
        }

    if isinstance(raw_embeddings, dict):
        return {
            str(node_name): np.asarray(vector, dtype=float)
            for node_name, vector in raw_embeddings.items()
        }

    raise TypeError(
        "Unsupported node2vec embeddings format. Expected a dict, gensim KeyedVectors, "
        "or a gensim Word2Vec model."
    )


def load_node2vec_embeddings(path: Path = NODE2VEC_EMBEDDINGS_PATH) -> dict[str, np.ndarray]:
    """Load node2vec repository embeddings from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Node2Vec embeddings file not found: {path}. "
            "Run python src/node2vec_embeddings.py first."
        )

    with path.open("rb") as file:
        return normalize_node2vec_embeddings(pickle.load(file))


def find_repository_index(repositories: pd.DataFrame, repo_name: str) -> int:
    """Find a repository row by exact name, with a case-insensitive fallback."""
    exact_matches = repositories.index[repositories["Name"] == repo_name].tolist()
    if exact_matches:
        return exact_matches[0]

    normalized_repo_name = repo_name.casefold()
    casefolded_names = repositories["Name"].astype(str).str.casefold()
    fallback_matches = repositories.index[casefolded_names == normalized_repo_name].tolist()
    if fallback_matches:
        return fallback_matches[0]

    raise ValueError(f"Repository not found: {repo_name}")


def validate_repository_columns(repositories: pd.DataFrame) -> None:
    """Validate repository metadata columns needed in recommendation results."""
    if "Name" not in repositories.columns:
        raise ValueError("Processed repository dataset is missing the Name column.")

    missing_result_columns = [column for column in BASE_RESULT_COLUMNS if column not in repositories.columns]
    if missing_result_columns:
        missing = ", ".join(missing_result_columns)
        raise ValueError(f"Processed repository dataset is missing required columns: {missing}")


def validate_semantic_inputs(repositories: pd.DataFrame, embeddings: np.ndarray) -> None:
    """Validate that repository metadata and semantic embeddings line up."""
    validate_repository_columns(repositories)

    if len(repositories) != len(embeddings):
        raise ValueError(
            "Repository metadata and embeddings have different row counts: "
            f"{len(repositories)} repositories vs {len(embeddings)} embeddings."
        )


def compute_popularity_scores(repositories: pd.DataFrame) -> np.ndarray:
    """Compute normalized popularity from Stars, Forks, and Issues."""
    validate_repository_columns(repositories)

    popularity_columns = ["Stars", "Forks", "Issues"]
    popularity_values = repositories.loc[:, popularity_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    normalized_values = MinMaxScaler().fit_transform(popularity_values)

    return (
        0.5 * normalized_values[:, 0]
        + 0.3 * normalized_values[:, 1]
        + 0.2 * normalized_values[:, 2]
    )


def find_embedding_key(node_embeddings: dict[str, np.ndarray], repo_name: str) -> str | None:
    """Find the matching embedding key for a repository name."""
    if repo_name in node_embeddings:
        return repo_name

    normalized_repo_name = repo_name.casefold()
    for embedding_key in node_embeddings:
        if embedding_key.casefold() == normalized_repo_name:
            return embedding_key

    return None


def recommend_semantic(repo_name: str, top_k: int = 10) -> pd.DataFrame:
    """Return the top-K semantically similar repositories for a repository name."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    repositories = load_repositories()
    embeddings = load_embeddings()
    validate_semantic_inputs(repositories, embeddings)

    selected_index = find_repository_index(repositories, repo_name)
    selected_embedding = embeddings[selected_index].reshape(1, -1)
    similarities = cosine_similarity(selected_embedding, embeddings)[0]
    similarities[selected_index] = -np.inf

    available_results = min(top_k, len(repositories) - 1)
    top_indices = np.argsort(similarities)[::-1][:available_results]

    recommendations = repositories.iloc[top_indices].copy()
    recommendations["semantic_similarity"] = similarities[top_indices]
    return recommendations.loc[:, SEMANTIC_RESULT_COLUMNS].reset_index(drop=True)


def recommend_graph(repo_name: str, top_k: int = 10) -> pd.DataFrame:
    """Return the top-K graph-similar repositories using Node2Vec embeddings."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    repositories = load_repositories()
    node_embeddings = load_node2vec_embeddings()
    validate_repository_columns(repositories)

    selected_index = find_repository_index(repositories, repo_name)
    selected_repo_name = str(repositories.iloc[selected_index]["Name"])
    selected_embedding_key = find_embedding_key(node_embeddings, selected_repo_name)
    if selected_embedding_key is None:
        raise ValueError(f"Repository missing from graph embeddings: {selected_repo_name}")

    selected_embedding = node_embeddings[selected_embedding_key].reshape(1, -1)
    scored_indices: list[tuple[int, float]] = []

    for repository_index, row in repositories.iterrows():
        if repository_index == selected_index:
            continue

        candidate_name = str(row["Name"])
        candidate_embedding_key = find_embedding_key(node_embeddings, candidate_name)
        if candidate_embedding_key is None:
            continue

        candidate_embedding = node_embeddings[candidate_embedding_key].reshape(1, -1)
        similarity = float(cosine_similarity(selected_embedding, candidate_embedding)[0, 0])
        scored_indices.append((repository_index, similarity))

    scored_indices.sort(key=lambda item: item[1], reverse=True)
    top_scored_indices = scored_indices[:top_k]

    recommendations = repositories.iloc[[index for index, _ in top_scored_indices]].copy()
    recommendations["graph_similarity"] = [similarity for _, similarity in top_scored_indices]
    return recommendations.loc[:, GRAPH_RESULT_COLUMNS].reset_index(drop=True)


def recommend_hybrid(repo_name: str, top_k: int = 10) -> pd.DataFrame:
    """Return the top-K repositories using semantic, graph, and popularity scores."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    repositories = load_repositories()
    embeddings = load_embeddings()
    node_embeddings = load_node2vec_embeddings()
    validate_semantic_inputs(repositories, embeddings)

    selected_index = find_repository_index(repositories, repo_name)
    selected_repo_name = str(repositories.iloc[selected_index]["Name"])
    selected_embedding_key = find_embedding_key(node_embeddings, selected_repo_name)
    if selected_embedding_key is None:
        raise ValueError(f"Repository missing from graph embeddings: {selected_repo_name}")

    semantic_similarities = cosine_similarity(embeddings[selected_index].reshape(1, -1), embeddings)[0]
    selected_graph_embedding = node_embeddings[selected_embedding_key].reshape(1, -1)
    popularity_scores = compute_popularity_scores(repositories)

    scored_indices: list[tuple[int, float, float, float, float]] = []
    for repository_index, row in repositories.iterrows():
        if repository_index == selected_index:
            continue

        candidate_name = str(row["Name"])
        candidate_embedding_key = find_embedding_key(node_embeddings, candidate_name)
        if candidate_embedding_key is None:
            continue

        candidate_graph_embedding = node_embeddings[candidate_embedding_key].reshape(1, -1)
        graph_similarity = float(cosine_similarity(selected_graph_embedding, candidate_graph_embedding)[0, 0])
        semantic_similarity = float(semantic_similarities[repository_index])
        popularity_score = float(popularity_scores[repository_index])
        final_score = (
            0.45 * semantic_similarity
            + 0.35 * graph_similarity
            + 0.20 * popularity_score
        )
        scored_indices.append(
            (
                repository_index,
                semantic_similarity,
                graph_similarity,
                popularity_score,
                final_score,
            )
        )

    scored_indices.sort(key=lambda item: item[4], reverse=True)
    top_scored_indices = scored_indices[:top_k]

    recommendations = repositories.iloc[[index for index, *_ in top_scored_indices]].copy()
    recommendations["semantic_similarity"] = [semantic for _, semantic, _, _, _ in top_scored_indices]
    recommendations["graph_similarity"] = [graph for _, _, graph, _, _ in top_scored_indices]
    recommendations["popularity_score"] = [popularity for _, _, _, popularity, _ in top_scored_indices]
    recommendations["final_score"] = [final for _, _, _, _, final in top_scored_indices]

    return recommendations.loc[:, HYBRID_RESULT_COLUMNS].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Recommend GitHub repositories.")
    parser.add_argument("--repo", required=True, help="Repository name to use as the recommendation seed.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations to return.")
    parser.add_argument(
        "--method",
        choices=["semantic", "graph", "hybrid"],
        default="semantic",
        help="Recommendation method to use.",
    )
    return parser.parse_args()


def print_recommendations(recommendations: pd.DataFrame) -> None:
    """Print recommendations as a readable table."""
    table = recommendations.copy()

    score_columns = [
        "semantic_similarity",
        "graph_similarity",
        "popularity_score",
        "final_score",
    ]
    for column in score_columns:
        if column in table.columns:
            table[column] = table[column].map(lambda value: f"{value:.4f}")

    print(table.to_string(index=False))


def main() -> None:
    """Run the selected recommender from the command line."""
    args = parse_args()

    try:
        if args.method == "semantic":
            recommendations = recommend_semantic(args.repo, args.top_k)
        elif args.method == "graph":
            recommendations = recommend_graph(args.repo, args.top_k)
        else:
            recommendations = recommend_hybrid(args.repo, args.top_k)
    except Exception as exc:
        print(f"Failed to generate {args.method} recommendations.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print_recommendations(recommendations)


if __name__ == "__main__":
    main()
