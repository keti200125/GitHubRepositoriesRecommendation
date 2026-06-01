"""Build a semantic similarity graph for GitHub repositories."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPOSITORIES_PATH = PROCESSED_DATA_DIR / "repositories_clean.csv"
EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "repository_embeddings.npy"
GRAPH_OUTPUT_PATH = PROCESSED_DATA_DIR / "repository_graph.gpickle"
TOP_K_SIMILAR = 5

NODE_ATTRIBUTES = [
    "Name",
    "Description",
    "URL",
    "Stars",
    "Forks",
    "Issues",
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


def validate_inputs(repositories: pd.DataFrame, embeddings: np.ndarray) -> None:
    """Validate graph construction inputs."""
    missing_columns = [column for column in NODE_ATTRIBUTES if column not in repositories.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Processed repository dataset is missing required columns: {missing}")

    if len(repositories) != len(embeddings):
        raise ValueError(
            "Repository metadata and embeddings have different row counts: "
            f"{len(repositories)} repositories vs {len(embeddings)} embeddings."
        )


def build_semantic_graph(
    repositories: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = TOP_K_SIMILAR,
) -> nx.Graph:
    """Build an undirected graph where edges connect semantically similar repositories."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    validate_inputs(repositories, embeddings)

    graph = nx.Graph()
    repository_names = repositories["Name"].astype(str).tolist()

    for _, row in repositories.iterrows():
        node_name = str(row["Name"])
        graph.add_node(
            node_name,
            **{attribute: row[attribute] for attribute in NODE_ATTRIBUTES},
        )

    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, -np.inf)

    for source_index, source_name in enumerate(repository_names):
        available_neighbors = min(top_k, len(repository_names) - 1)
        top_indices = np.argsort(similarity_matrix[source_index])[::-1][:available_neighbors]

        for target_index in top_indices:
            target_name = repository_names[target_index]
            similarity_score = float(similarity_matrix[source_index, target_index])

            if graph.has_edge(source_name, target_name):
                existing_score = graph[source_name][target_name]["similarity_score"]
                graph[source_name][target_name]["similarity_score"] = max(
                    existing_score,
                    similarity_score,
                )
            else:
                graph.add_edge(
                    source_name,
                    target_name,
                    similarity_score=similarity_score,
                )

    return graph


def save_graph(graph: nx.Graph, output_path: Path = GRAPH_OUTPUT_PATH) -> None:
    """Save the graph as a pickle-backed gpickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(graph, file, protocol=pickle.HIGHEST_PROTOCOL)


def average_degree(graph: nx.Graph) -> float:
    """Return the graph average degree."""
    if graph.number_of_nodes() == 0:
        return 0.0

    return sum(dict(graph.degree()).values()) / graph.number_of_nodes()


def main() -> None:
    """Build and save the repository semantic similarity graph."""
    try:
        repositories = load_repositories()
        embeddings = load_embeddings()
        graph = build_semantic_graph(repositories, embeddings)
        save_graph(graph)
    except Exception as exc:
        print("Failed to build semantic similarity graph.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Average degree: {average_degree(graph):.2f}")
    print(f"Output path: {GRAPH_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
