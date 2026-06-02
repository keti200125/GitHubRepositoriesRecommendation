"""Generate local Ollama baseline recommendations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
PROFILE_COLUMNS = ["Name", "Description", "Stars", "Forks"]


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server is not available."""


@dataclass(frozen=True)
class OllamaBaselineResult:
    """Ollama baseline output and saved artifact paths."""

    answer: str
    prompt_path: Path
    answer_path: Path
    removed_selected_repositories: list[str]


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


def normalized_name(value: object) -> str:
    """Normalize repository names for case-insensitive comparison."""
    return str(value).casefold()


def select_candidate_repositories(
    repositories: pd.DataFrame,
    selected_repositories: pd.DataFrame,
) -> pd.DataFrame:
    """Return all candidate repositories, excluding selected repositories case-insensitively."""
    selected_names = set(selected_repositories["Name"].map(normalized_name))
    repository_names = repositories["Name"].map(normalized_name)
    candidates = repositories[~repository_names.isin(selected_names)].copy()
    if candidates.empty:
        raise ValueError("No candidate repositories are available after excluding selected repositories.")

    return candidates.loc[:, PROFILE_COLUMNS].reset_index(drop=True)


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
            "USER PROFILE - repositories the user already likes:",
            "",
            selected_text,
            "",
            "TASK:",
            "Choose the top 5 most relevant NEW repositories from the candidate list below.",
            "Do not recommend any repository from the user's selected repositories.",
            "Choose only from the provided candidate list.",
            "Do not invent repository names.",
            "Return exactly 5 repositories.",
            "Provide a short reason for each recommendation.",
            "",
            "CANDIDATE REPOSITORIES:",
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


def remove_selected_repositories_from_answer(
    answer: str,
    selected_repo_names: list[str],
) -> tuple[str, list[str]]:
    """Remove answer lines that contain any selected repository name."""
    selected_names = {normalized_name(name): name for name in selected_repo_names}
    removed_names = set()
    kept_lines = []

    for line in answer.splitlines():
        normalized_line = normalized_name(line)
        matched_names = [display_name for name, display_name in selected_names.items() if name in normalized_line]
        if matched_names:
            removed_names.update(matched_names)
            continue

        kept_lines.append(line)

    cleaned_answer = "\n".join(kept_lines).strip()
    return cleaned_answer, sorted(removed_names, key=normalized_name)


def run_ollama_baseline(
    repo_names: list[str],
    model: str = DEFAULT_MODEL,
) -> OllamaBaselineResult:
    """Generate an Ollama baseline answer and save prompt/answer files."""
    repositories = load_repositories()
    selected_repositories = select_profile_repositories(repositories, repo_names)
    candidate_repositories = select_candidate_repositories(repositories, selected_repositories)
    prompt = generate_prompt(selected_repositories, candidate_repositories)
    canonical_names = selected_repositories["Name"].astype(str).tolist()

    prompt_path = save_text(prompt, "ollama_prompt", canonical_names)
    raw_answer = call_ollama(prompt, model)
    answer, removed_names = remove_selected_repositories_from_answer(raw_answer, canonical_names)
    answer_path = save_text(answer, "ollama_answer", canonical_names)

    return OllamaBaselineResult(
        answer=answer,
        prompt_path=prompt_path,
        answer_path=answer_path,
        removed_selected_repositories=removed_names,
    )


def main() -> None:
    """Run the local Ollama baseline from the command line."""
    args = parse_args()

    try:
        result = run_ollama_baseline(
            args.repos,
            model=args.model,
        )
    except Exception as exc:
        print("Failed to run Ollama baseline.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(result.answer)
    if result.removed_selected_repositories:
        removed = ", ".join(result.removed_selected_repositories)
        print(f"\nRemoved selected repositories from Ollama response: {removed}")
    print(f"\nPrompt path: {result.prompt_path}")
    print(f"Answer path: {result.answer_path}")


if __name__ == "__main__":
    main()
