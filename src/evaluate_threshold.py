"""Threshold-based approximate evaluation for recommendation methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .recommender import (
        load_repositories,
        recommend_graph,
        recommend_hybrid,
        recommend_semantic,
    )
except ImportError:
    from recommender import (
        load_repositories,
        recommend_graph,
        recommend_hybrid,
        recommend_semantic,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_PATH = OUTPUTS_DIR / "threshold_evaluation.csv"
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.5
TEST_REPOSITORIES = [
    "tensorflow",
    "react",
    "kubernetes",
    "freeCodeCamp",
    "transformers",
]
METHODS = {
    "semantic": (recommend_semantic, "semantic_similarity"),
    "graph": (recommend_graph, "graph_similarity"),
    "hybrid": (recommend_hybrid, "final_score"),
}
RESULT_COLUMNS = [
    "query_repository",
    "method",
    "top_k",
    "threshold",
    "relevant_count",
    "precision_at_k",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run threshold-based recommender evaluation.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of recommendations per method.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum method score needed to count a recommendation as relevant.",
    )
    return parser.parse_args()


def repository_exists(repositories: pd.DataFrame, repo_name: str) -> bool:
    """Return whether a repository exists in the cleaned dataset."""
    repository_names = repositories["Name"].astype(str).str.casefold()
    return repository_names.eq(repo_name.casefold()).any()


def validate_inputs(repositories: pd.DataFrame, test_repositories: list[str], top_k: int) -> None:
    """Validate evaluation inputs before running recommenders."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    missing_repositories = [repo for repo in test_repositories if not repository_exists(repositories, repo)]
    if missing_repositories:
        missing = ", ".join(missing_repositories)
        raise ValueError(f"Test repositories not found in cleaned dataset: {missing}")


def precision_at_k(recommendations: pd.DataFrame, score_column: str, top_k: int, threshold: float) -> tuple[int, float]:
    """Compute Precision@K using a score threshold as the relevance rule."""
    if score_column not in recommendations.columns:
        raise ValueError(f"Recommendation results are missing score column: {score_column}")

    scores = pd.to_numeric(recommendations[score_column], errors="coerce").fillna(float("-inf"))
    relevant_count = int((scores >= threshold).sum())
    return relevant_count, relevant_count / top_k


def evaluate_threshold(
    test_repositories: list[str] = TEST_REPOSITORIES,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Evaluate each method with threshold-defined Precision@K."""
    # This is an approximate evaluation because relevance is defined automatically
    # by a similarity threshold, not by human labels.
    repositories = load_repositories()
    validate_inputs(repositories, test_repositories, top_k)

    rows = []
    for query_repository in test_repositories:
        for method, (recommend, score_column) in METHODS.items():
            recommendations = recommend(query_repository, top_k)
            relevant_count, precision = precision_at_k(recommendations, score_column, top_k, threshold)
            rows.append(
                {
                    "query_repository": query_repository,
                    "method": method,
                    "top_k": top_k,
                    "threshold": threshold,
                    "relevant_count": relevant_count,
                    "precision_at_k": round(precision, 4),
                }
            )

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def save_results(results: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> Path:
    """Save threshold evaluation results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return output_path


def print_summary(results: pd.DataFrame, output_path: Path) -> None:
    """Print per-query results and average Precision@K per method."""
    average_precision = (
        results.groupby("method", as_index=False)["precision_at_k"]
        .mean()
        .sort_values("precision_at_k", ascending=False)
    )
    average_precision["precision_at_k"] = average_precision["precision_at_k"].round(4)

    print("Threshold-Based Recommendation Evaluation")
    print("=========================================")
    print("Note: relevance is approximated by a score threshold, not human labels.")
    print(f"Top-K: {int(results['top_k'].iloc[0])}")
    print(f"Threshold: {results['threshold'].iloc[0]}")
    print(f"Output file: {output_path}")
    print("\nPer-query Precision@K:")
    print(results.to_string(index=False))
    print("\nAverage Precision@K by method:")
    print(average_precision.to_string(index=False))


def main() -> None:
    """Run threshold-based evaluation from the command line."""
    args = parse_args()

    try:
        results = evaluate_threshold(top_k=args.top_k, threshold=args.threshold)
        output_path = save_results(results)
    except Exception as exc:
        print("Failed to run threshold-based evaluation.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print_summary(results, output_path)


if __name__ == "__main__":
    main()

