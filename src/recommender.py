"""Semantic repository recommendations based on description embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPOSITORIES_PATH = PROCESSED_DATA_DIR / "repositories_clean.csv"
EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "repository_embeddings.npy"

RESULT_COLUMNS = [
    "Name",
    "Description",
    "URL",
    "Stars",
    "Forks",
    "Issues",
    "semantic_similarity",
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


def validate_inputs(repositories: pd.DataFrame, embeddings: np.ndarray) -> None:
    """Validate that repository metadata and embeddings line up."""
    if "Name" not in repositories.columns:
        raise ValueError("Processed repository dataset is missing the Name column.")

    missing_result_columns = [
        column for column in RESULT_COLUMNS if column != "semantic_similarity" and column not in repositories.columns
    ]
    if missing_result_columns:
        missing = ", ".join(missing_result_columns)
        raise ValueError(f"Processed repository dataset is missing required columns: {missing}")

    if len(repositories) != len(embeddings):
        raise ValueError(
            "Repository metadata and embeddings have different row counts: "
            f"{len(repositories)} repositories vs {len(embeddings)} embeddings."
        )


def recommend_semantic(repo_name: str, top_k: int = 10) -> pd.DataFrame:
    """Return the top-K semantically similar repositories for a repository name."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    repositories = load_repositories()
    embeddings = load_embeddings()
    validate_inputs(repositories, embeddings)

    selected_index = find_repository_index(repositories, repo_name)
    selected_embedding = embeddings[selected_index].reshape(1, -1)
    similarities = cosine_similarity(selected_embedding, embeddings)[0]
    similarities[selected_index] = -np.inf

    available_results = min(top_k, len(repositories) - 1)
    top_indices = np.argsort(similarities)[::-1][:available_results]

    recommendations = repositories.iloc[top_indices].copy()
    recommendations["semantic_similarity"] = similarities[top_indices]
    return recommendations.loc[:, RESULT_COLUMNS].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Recommend repositories with semantic similarity.")
    parser.add_argument("--repo", required=True, help="Repository name to use as the recommendation seed.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations to return.")
    return parser.parse_args()


def print_recommendations(recommendations: pd.DataFrame) -> None:
    """Print recommendations as a readable table."""
    table = recommendations.copy()
    table["semantic_similarity"] = table["semantic_similarity"].map(lambda value: f"{value:.4f}")
    print(table.to_string(index=False))


def main() -> None:
    """Run the semantic recommender from the command line."""
    args = parse_args()

    try:
        recommendations = recommend_semantic(args.repo, args.top_k)
    except Exception as exc:
        print("Failed to generate semantic recommendations.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print_recommendations(recommendations)


if __name__ == "__main__":
    main()
