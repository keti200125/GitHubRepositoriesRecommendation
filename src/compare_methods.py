"""Compare semantic, graph, and hybrid repository recommendations."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from recommender import (
    print_recommendations,
    recommend_graph,
    recommend_hybrid,
    recommend_semantic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
COMPARISON_COLUMNS = [
    "method",
    "rank",
    "Name",
    "Description",
    "Stars",
    "Forks",
    "Issues",
    "semantic_similarity",
    "graph_similarity",
    "popularity_score",
    "final_score",
]
SCORE_COLUMNS = [
    "semantic_similarity",
    "graph_similarity",
    "popularity_score",
    "final_score",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare repository recommendation methods.")
    parser.add_argument("--repo", required=True, help="Repository name to use as the recommendation seed.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations per method.")
    return parser.parse_args()


def safe_filename_part(value: str) -> str:
    """Create a filesystem-safe filename part from a repository name."""
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe_value.strip("_") or "repository"


def prepare_for_csv(method: str, recommendations: pd.DataFrame) -> pd.DataFrame:
    """Add method/rank fields and align recommendation columns for CSV export."""
    table = recommendations.copy()
    table.insert(0, "rank", range(1, len(table) + 1))
    table.insert(0, "method", method)

    for column in COMPARISON_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan

    for column in SCORE_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce").round(4)

    return table.loc[:, COMPARISON_COLUMNS]


def save_comparison(repo_name: str, results: dict[str, pd.DataFrame]) -> Path:
    """Save all recommendation results to one comparison CSV file."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"comparison_{safe_filename_part(repo_name)}.csv"

    comparison = pd.concat(
        [prepare_for_csv(method, recommendations) for method, recommendations in results.items()],
        ignore_index=True,
    )
    comparison.to_csv(output_path, index=False)

    return output_path


def print_section(title: str, recommendations: pd.DataFrame) -> None:
    """Print one recommendation table with a section title."""
    print(f"\n{title}")
    print("=" * len(title))
    print_recommendations(recommendations)


def main() -> None:
    """Compare all recommendation methods for one repository."""
    args = parse_args()

    try:
        results = {
            "semantic": recommend_semantic(args.repo, args.top_k),
            "graph": recommend_graph(args.repo, args.top_k),
            "hybrid": recommend_hybrid(args.repo, args.top_k),
        }
        output_path = save_comparison(args.repo, results)
    except Exception as exc:
        print("Failed to compare recommendation methods.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print_section("Semantic recommendations", results["semantic"])
    print_section("Graph recommendations", results["graph"])
    print_section("Hybrid recommendations", results["hybrid"])
    print(f"\nSaved comparison CSV: {output_path}")


if __name__ == "__main__":
    main()
