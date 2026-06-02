"""Generate local Ollama baseline recommendations."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES_PATH = PROJECT_ROOT / "data" / "processed" / "repositories_clean.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"
DEFAULT_NUM_CANDIDATES = 20
PROFILE_COLUMNS = ["Name", "Description", "Stars", "Forks"]


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server is not available."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a local Ollama baseline recommendation.")
    parser.add_argument(
        "--repos",
        nargs="+",
        required=True,
        help="Repository names that represent the user's profile.",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=DEFAULT_NUM_CANDIDATES,
        help="Number of candidate repositories to include.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local Ollama model to use.",
    )
    return parser.parse_args()


def load_repositories(path: Path = REPOSITORIES_PATH) -> pd.DataFrame:
    """Load cleaned repository metadata."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned repository dataset not found: {path}. "
            "Run python3 src/data_preprocessing.py first."
        )

    repositories = pd.read_csv(path)
    missing_columns = [column for column in PROFILE_COLUMNS if column not in repositories.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Cleaned repository dataset is missing required columns: {missing}")

    return repositories


def find_repository(repositories: pd.DataFrame, repo_name: str) -> pd.Series:
    """Find one repository by exact name, with a case-insensitive fallback."""
    exact_matches = repositories[repositories["Name"] == repo_name]
    if not exact_matches.empty:
        return exact_matches.iloc[0]

    normalized_repo_name = repo_name.casefold()
    casefolded_names = repositories["Name"].astype(str).str.casefold()
    fallback_matches = repositories[casefolded_names == normalized_repo_name]
    if not fallback_matches.empty:
        return fallback_matches.iloc[0]

    raise ValueError(f"Repository not found in cleaned dataset: {repo_name}")


def select_profile_repositories(repositories: pd.DataFrame, repo_names: list[str]) -> pd.DataFrame:
    """Return selected profile repositories in the same order as the input."""
    selected_rows = []
    seen_names = set()

    for repo_name in repo_names:
        row = find_repository(repositories, repo_name)
        canonical_name = str(row["Name"])
        if canonical_name in seen_names:
            continue

        selected_rows.append(row)
        seen_names.add(canonical_name)

    return pd.DataFrame(selected_rows).loc[:, PROFILE_COLUMNS].reset_index(drop=True)


def sample_candidates(
    repositories: pd.DataFrame,
    selected_repositories: pd.DataFrame,
    num_candidates: int,
) -> pd.DataFrame:
    """Randomly sample candidate repositories, excluding profile repositories."""
    if num_candidates <= 0:
        raise ValueError("num_candidates must be greater than 0.")

    selected_names = set(selected_repositories["Name"].astype(str))
    candidate_pool = repositories[~repositories["Name"].astype(str).isin(selected_names)].copy()
    if candidate_pool.empty:
        raise ValueError("No candidate repositories are available after excluding selected repositories.")

    sample_count = min(num_candidates, len(candidate_pool))
    return candidate_pool.sample(n=sample_count).loc[:, PROFILE_COLUMNS].reset_index(drop=True)


def format_repository_list(repositories: pd.DataFrame) -> str:
    """Format repository metadata as a numbered list for the Ollama prompt."""
    lines = []

    for index, row in repositories.iterrows():
        description = str(row["Description"]).strip()
        lines.append(
            textwrap.dedent(
                f"""
                {index + 1}. Name: {row['Name']}
                   Description: {description}
                   Stars: {row['Stars']}
                   Forks: {row['Forks']}
                """
            ).strip()
        )

    return "\n\n".join(lines)


def generate_prompt(selected_repositories: pd.DataFrame, candidate_repositories: pd.DataFrame) -> str:
    """Build the local LLM baseline prompt."""
    selected_text = format_repository_list(selected_repositories)
    candidate_text = format_repository_list(candidate_repositories)

    return "\n".join(
        [
            "Given the user likes these repositories:",
            "",
            selected_text,
            "",
            "Choose the top 5 most relevant repositories from this candidate list.",
            "Return repository names and short reasons.",
            "Use only repositories from the candidate list.",
            "",
            "Candidate repositories:",
            "",
            candidate_text,
            "",
            "Format your answer as a ranked list: repository name - short reason.",
        ]
    )


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, url: str = OLLAMA_GENERATE_URL) -> str:
    """Call the local Ollama generate API."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise OllamaUnavailableError(
            "Ollama is not running. Start it with: ollama serve"
        ) from exc

    answer = str(data.get("response", "")).strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response.")

    return answer


def safe_filename_part(repo_names: list[str]) -> str:
    """Create a filesystem-safe filename part from selected repository names."""
    joined_names = "_".join(repo_names)
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", joined_names.strip())
    return safe_value.strip("_") or "repositories"


def save_text(text: str, prefix: str, repo_names: list[str], output_dir: Path = OUTPUTS_DIR) -> Path:
    """Save a prompt or answer text file under outputs/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}_{safe_filename_part(repo_names)}.txt"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def run_ollama_baseline(
    repo_names: list[str],
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    model: str = DEFAULT_MODEL,
) -> tuple[str, Path, Path]:
    """Generate an Ollama baseline answer and save prompt/answer files."""
    repositories = load_repositories()
    selected_repositories = select_profile_repositories(repositories, repo_names)
    candidate_repositories = sample_candidates(repositories, selected_repositories, num_candidates)
    prompt = generate_prompt(selected_repositories, candidate_repositories)
    canonical_names = selected_repositories["Name"].astype(str).tolist()

    prompt_path = save_text(prompt, "ollama_prompt", canonical_names)
    answer = call_ollama(prompt, model)
    answer_path = save_text(answer, "ollama_answer", canonical_names)

    return answer, prompt_path, answer_path


def main() -> None:
    """Run the local Ollama baseline from the command line."""
    args = parse_args()

    try:
        answer, prompt_path, answer_path = run_ollama_baseline(
            args.repos,
            num_candidates=args.num_candidates,
            model=args.model,
        )
    except Exception as exc:
        print("Failed to run Ollama baseline.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(answer)
    print(f"\nPrompt path: {prompt_path}")
    print(f"Answer path: {answer_path}")


if __name__ == "__main__":
    main()
