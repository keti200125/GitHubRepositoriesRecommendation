"""Streamlit demo for GitHub repository recommendations."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.recommender import recommend_graph, recommend_hybrid, recommend_semantic
from src.ollama_baseline import (
    DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL,
    DEFAULT_NUM_CANDIDATES as OLLAMA_NUM_CANDIDATES,
    OllamaUnavailableError,
    run_ollama_baseline,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORIES_PATH = PROJECT_ROOT / "data" / "processed" / "repositories_clean.csv"
SAMPLE_SIZE = 15
MAX_SELECTED_REPOSITORIES = 5
TOP_RECOMMENDATIONS = 5
PROFILE_COLUMNS = ["Name", "Description", "Stars", "Forks", "Issues"]
METHOD_SCORE_COLUMNS = {
    "semantic": ["semantic_similarity"],
    "graph": ["graph_similarity"],
    "hybrid": ["semantic_similarity", "graph_similarity", "popularity_score", "final_score"],
}
METHOD_SORT_COLUMNS = {
    "semantic": "semantic_similarity",
    "graph": "graph_similarity",
    "hybrid": "final_score",
}


@st.cache_data
def load_repositories(path: Path = REPOSITORIES_PATH) -> pd.DataFrame:
    """Load cleaned repository metadata."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned repository dataset not found: {path}. "
            "Run python3 src/data_preprocessing.py first."
        )

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_seed_recommendations(repo_name: str, method: str, top_k: int) -> pd.DataFrame:
    """Generate recommendations for one seed repository."""
    if method == "semantic":
        return recommend_semantic(repo_name, top_k)
    if method == "graph":
        return recommend_graph(repo_name, top_k)
    if method == "hybrid":
        return recommend_hybrid(repo_name, top_k)

    raise ValueError(f"Unsupported recommendation method: {method}")


def sample_repository_names(repositories: pd.DataFrame) -> list[str]:
    """Randomly sample repository names for the selector."""
    repository_names = repositories["Name"].dropna().astype(str)
    sample_count = min(SAMPLE_SIZE, len(repository_names))
    return repository_names.sample(n=sample_count).sort_values().tolist()


def initialize_repository_options(repositories: pd.DataFrame) -> None:
    """Initialize repository selector state."""
    if "repository_options" not in st.session_state:
        st.session_state.repository_options = sample_repository_names(repositories)
    if "selected_repositories" not in st.session_state:
        st.session_state.selected_repositories = []


def reset_repository_options(repositories: pd.DataFrame) -> None:
    """Refresh the random repository selector options."""
    st.session_state.repository_options = sample_repository_names(repositories)
    st.session_state.selected_repositories = []


def build_profile_recommendations(
    selected_repositories: tuple[str, ...],
    method: str,
    repositories: pd.DataFrame,
    top_k: int = TOP_RECOMMENDATIONS,
) -> pd.DataFrame:
    """Aggregate recommendations across selected repositories."""
    candidate_pool_size = max(top_k, len(repositories) - 1)
    selected_names = set(selected_repositories)
    recommendation_frames = []

    for repo_name in selected_repositories:
        recommendations = get_seed_recommendations(repo_name, method, candidate_pool_size)
        recommendations = recommendations[~recommendations["Name"].astype(str).isin(selected_names)].copy()
        recommendation_frames.append(recommendations)

    if not recommendation_frames:
        return pd.DataFrame(columns=[*PROFILE_COLUMNS, *METHOD_SCORE_COLUMNS[method]])

    combined = pd.concat(recommendation_frames, ignore_index=True)
    aggregation = {
        "Description": "first",
        "Stars": "first",
        "Forks": "first",
        "Issues": "first",
        **{column: "mean" for column in METHOD_SCORE_COLUMNS[method]},
    }

    profile_recommendations = combined.groupby("Name", as_index=False).agg(aggregation)
    profile_recommendations = profile_recommendations.sort_values(
        METHOD_SORT_COLUMNS[method],
        ascending=False,
    )

    columns = [*PROFILE_COLUMNS, *METHOD_SCORE_COLUMNS[method]]
    return profile_recommendations.loc[:, columns].head(top_k).reset_index(drop=True)


def format_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Round recommendation score columns for display."""
    display_table = recommendations.copy()
    for column in ["semantic_similarity", "graph_similarity", "popularity_score", "final_score"]:
        if column in display_table.columns:
            display_table[column] = display_table[column].round(4)

    return display_table


def selected_repository_details(repositories: pd.DataFrame, selected_repositories: list[str]) -> pd.DataFrame:
    """Return metadata for selected repositories in selection order."""
    selected = repositories[repositories["Name"].astype(str).isin(selected_repositories)].copy()
    selected["_selection_order"] = selected["Name"].astype(str).map(
        {name: index for index, name in enumerate(selected_repositories)}
    )
    return selected.sort_values("_selection_order").loc[:, PROFILE_COLUMNS]


def main() -> None:
    """Render the Streamlit demo."""
    st.set_page_config(page_title="GitHub Repositories Recommender", layout="wide")
    st.title("GitHub Repositories Recommender")

    try:
        repositories = load_repositories()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    initialize_repository_options(repositories)

    if st.button("Generate New Repository Set", type="secondary"):
        reset_repository_options(repositories)

    st.multiselect(
        "Repositories",
        options=st.session_state.repository_options,
        key="selected_repositories",
        max_selections=MAX_SELECTED_REPOSITORIES,
    )

    selected_repositories = st.session_state.selected_repositories

    st.subheader("Selected User Repositories")
    if selected_repositories:
        st.dataframe(
            selected_repository_details(repositories, selected_repositories),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Select at least one repository.")

    if not 1 <= len(selected_repositories) <= MAX_SELECTED_REPOSITORIES:
        st.stop()

    selected_profile = tuple(selected_repositories)
    tabs = st.tabs(["Semantic", "Graph", "Hybrid"])
    methods = ["semantic", "graph", "hybrid"]

    for tab, method in zip(tabs, methods, strict=True):
        with tab:
            try:
                start_time = time.perf_counter()
                recommendations = build_profile_recommendations(selected_profile, method, repositories)
                elapsed_time = time.perf_counter() - start_time
            except Exception as exc:
                st.error(str(exc))
                continue

            st.dataframe(
                format_recommendations(recommendations),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"Top {TOP_RECOMMENDATIONS} results generated in {elapsed_time:.2f}s")

    st.subheader("Compare with local LLM (Ollama)")
    st.caption(
        f"Uses local Ollama model `{OLLAMA_DEFAULT_MODEL}` with "
        f"{OLLAMA_NUM_CANDIDATES} sampled candidate repositories."
    )

    if st.button("Run Ollama Baseline", type="primary"):
        try:
            with st.spinner("Asking local Ollama model..."):
                answer, prompt_path, answer_path = run_ollama_baseline(
                    list(selected_profile),
                    num_candidates=OLLAMA_NUM_CANDIDATES,
                    model=OLLAMA_DEFAULT_MODEL,
                )
        except OllamaUnavailableError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(str(exc))
        else:
            st.markdown(answer)
            st.caption(f"Prompt saved to: {prompt_path}")
            st.caption(f"Answer saved to: {answer_path}")


if __name__ == "__main__":
    main()
