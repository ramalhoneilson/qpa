"""
Discovers non-Python quantum computing repositories on GitHub.

Mirrors the logic of src/data_acquisition/discover_projects.py but targets
every language except Python. Each language gets its own set of topic queries
so results are not diluted by cross-language noise.

Outputs (written to data/multilanguage/):
  - multilanguage_repo_list.txt             — plain list for automation
  - quantum_multilang_structured_<ts>.json  — full metadata per repo
  - github_multilang_summary_<ts>.txt       — human-readable summary
"""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from github import Auth, Github, GithubException

from src.conf import config

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")

TARGET_RESULT_COUNT = 200
SORT_BY = "stars"
SORT_ORDER = "desc"

# Languages commonly used for quantum computing frameworks outside Python.
# GitHub language identifiers are used (case-insensitive in search).
NON_PYTHON_LANGUAGES = [
    "julia",
    "c++",
    "rust",
    "java",
    "javascript",
    "typescript",
    "q#",
    "go",
    "haskell",
    "scala",
    "matlab",
    "fortran",
]

QUANTUM_TOPICS = [
    "quantum-computing",
    "quantum-machine-learning",
    "quantum-algorithms",
    "quantum-simulation",
    "quantum-error-correction",
    "quantum-circuit",
]

# Build one query per (topic, language) pair — same pattern as discover_projects.py.
search_queries = [
    f"topic:{topic} language:{lang}"
    for topic in QUANTUM_TOPICS
    for lang in NON_PYTHON_LANGUAGES
]

MIN_STARS = 30
MIN_CONTRIBUTORS = 10
MAX_INACTIVITY_MONTHS = 12

EXCLUSION_KEYWORDS = [
    "awesome-list",
    "awesome",
    "books",
    "book",
    "tutorial",
    "tutorials",
    "course",
    "learning",
    "learn",
    "textbook",
    "lecture",
    "lectures",
    "education",
    "cheatsheet",
]

OUTPUT_FOLDER = config.PROJECT_ROOT / "data" / "multilanguage"


def check_for_exclusion(repo):
    """Checks if a repository should be excluded based on keywords indicating
    books, awesome-lists, or educational/tutorial content."""
    repo_name = repo.full_name.lower()
    description = repo.description.lower() if repo.description else ""
    topics = [topic.lower() for topic in repo.topics]
    text_to_check = f"{repo_name} {description}"

    for keyword in EXCLUSION_KEYWORDS:
        if keyword in text_to_check:
            return True, f"Keyword '{keyword}' in name/description"
        if keyword in topics:
            return True, f"Topic '{keyword}'"
    return False, None


def is_repo_relevant(repo):
    """
    Applies all filters to a single repository.
    Returns (is_relevant, reason_or_contrib_count).
    If relevant, returns (True, contributor_count).
    If not, returns (False, reason_string).
    """
    if repo.archived:
        return False, "Archived repository"
    if repo.fork:
        return False, "Is a fork"
    if repo.stargazers_count < MIN_STARS:
        return False, f"Not enough stars ({repo.stargazers_count} < {MIN_STARS})"

    now = datetime.now(UTC)
    inactivity_threshold = now - relativedelta(months=MAX_INACTIVITY_MONTHS)
    if repo.pushed_at < inactivity_threshold:
        return False, f"Inactive since {repo.pushed_at.date()}"

    is_excluded, reason = check_for_exclusion(repo)
    if is_excluded:
        return False, f"Excluded by {reason}"

    # Reject any repo whose primary language is Python — this script is strictly
    # for non-Python frameworks.
    if repo.language and repo.language.lower() == "python":
        return False, f"Primary language is Python"

    try:
        contributors_count = repo.get_contributors().totalCount
        if contributors_count < MIN_CONTRIBUTORS:
            return (
                False,
                f"Not enough contributors ({contributors_count} < {MIN_CONTRIBUTORS})",
            )
    except GithubException as e:
        if e.status == 403:
            reason = f"Could not fetch contributors due to API limits ({e.status})"
            print(f"Warning: {reason} for {repo.full_name}. Skipping.", file=sys.stderr)
            return False, reason
        contributors_count = "N/A"

    return True, contributors_count


def generate_summary_file(
    total_candidates, final_repos, filtered_out_repos, output_folder, timestamp
):
    """Generates a text summary of the GitHub search findings."""
    num_candidates = total_candidates
    num_final = len(final_repos)
    num_filtered = len(filtered_out_repos)

    summary_file_path = output_folder / f"github_multilang_summary_{timestamp}.txt"

    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write("GitHub Multilanguage Quantum Projects Search Summary\n")
        f.write("=" * 52 + "\n")
        f.write(
            f"Summary generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Languages targeted: {', '.join(NON_PYTHON_LANGUAGES)}\n\n")

        f.write("--- Overall Statistics ---\n")
        f.write(f"Total unique projects found (candidates): {num_candidates}\n")
        f.write(f"Projects filtered out:                    {num_filtered}\n")
        f.write(f"Final projects considered for analysis:   {num_final}\n\n")

        f.write("--- Filtered-Out Projects ---\n")
        if not filtered_out_repos:
            f.write("No projects were filtered out.\n\n")
        else:
            max_name_len = max(
                (len(repo["full_name"]) for repo in filtered_out_repos), default=10
            )
            col1_width = max(len("Repository"), max_name_len) + 2

            header = f"{'Repository':<{col1_width}}{'Reason for Filtering'}\n"
            f.write(header)
            f.write("-" * (len(header) + 20) + "\n")

            filtered_out_repos.sort(key=lambda x: x["full_name"].lower())

            for repo in filtered_out_repos:
                f.write(f"{repo['full_name']:<{col1_width}}{repo['reason']}\n")
            f.write("\n")

        f.write(f"--- Final Considered Projects (Top {num_final}) ---\n")
        if not final_repos:
            f.write("No projects met the criteria.\n")
        else:
            for i, repo in enumerate(final_repos):
                f.write(f"{i + 1}. {repo.full_name}\n")

        f.write("\n" + "=" * 52 + "\n")
        f.write("End of Summary\n")

    print(f"Search summary saved to: {summary_file_path}")


def search_github_for_multilang_qc_frameworks():
    """Searches GitHub for non-Python quantum frameworks using topic+language queries
    and applies the same quality filters as discover_projects.py."""
    if not GITHUB_TOKEN:
        print("Error: GitHub PAT/TOKEN not found.", file=sys.stderr)
        return

    try:
        g = Github(auth=Auth.Token(GITHUB_TOKEN))
        repo_candidates = {}

        print("--- Phase 1: Discovering non-Python repositories via GitHub search ---", flush=True)
        print(f"Languages: {', '.join(NON_PYTHON_LANGUAGES)}", flush=True)
        print(f"Topics:    {', '.join(QUANTUM_TOPICS)}\n", flush=True)

        for query in search_queries:
            print(f"Searching with query: '{query}'...", flush=True)
            try:
                repositories = g.search_repositories(
                    query=query, sort=SORT_BY, order=SORT_ORDER
                )
                for repo in repositories[:200]:
                    if repo.full_name not in repo_candidates:
                        repo_candidates[repo.full_name] = repo
            except GithubException as e:
                print(
                    f"Warning: Search query '{query}' failed: {e.status}",
                    file=sys.stderr,
                )
            except Exception as e:
                # PyGithub can raise non-GithubException errors (e.g. IndexError
                # in its internal retry logic after a 403). Skip the query and
                # continue rather than aborting the whole run.
                print(
                    f"Warning: Search query '{query}' raised an unexpected error "
                    f"({type(e).__name__}: {e}). Skipping.",
                    file=sys.stderr,
                )

        print(
            f"\nGathered a total of {len(repo_candidates)} unique candidate repositories."
        )

        print("\n--- Phase 2: Applying quality filters ---")
        print(
            f" - Min Stars: {MIN_STARS}, Min Contributors: {MIN_CONTRIBUTORS}, "
            f"Last push <= {MAX_INACTIVITY_MONTHS} months"
        )
        print(f" - Excluding keywords: {', '.join(EXCLUSION_KEYWORDS[:4])}...")
        print(" - Excluding repos whose primary language is Python")
        print("-" * 70)

        final_repos = []
        filtered_out_repos = []
        for repo in repo_candidates.values():
            is_relevant, detail = is_repo_relevant(repo)
            if is_relevant:
                repo.contributors_count = detail
                final_repos.append(repo)
            else:
                if detail != "Is a fork":
                    print(f"[SKIPPING] {repo.full_name}: {detail}.")
                filtered_out_repos.append(
                    {"full_name": repo.full_name, "reason": detail}
                )

        final_repos.sort(key=lambda r: r.stargazers_count, reverse=True)
        top_repos = final_repos[:TARGET_RESULT_COUNT]

        repos_culled_by_rank = final_repos[TARGET_RESULT_COUNT:]
        for repo in repos_culled_by_rank:
            filtered_out_repos.append(
                {
                    "full_name": repo.full_name,
                    "reason": (
                        f"Passed filters but not in top {TARGET_RESULT_COUNT} "
                        f"by stars ({repo.stargazers_count} stars)"
                    ),
                }
            )

        print("-" * 70)
        print(
            f"Found {len(top_repos)} matching repositories after filtering "
            f"{len(repo_candidates)} candidates.\n"
        )

        structured_results = []
        for i, repo in enumerate(top_repos):
            structured_results.append(
                {
                    "rank": i + 1,
                    "full_name": repo.full_name,
                    "language": repo.language,
                    "stargazers_count": repo.stargazers_count,
                    "contributors_count": repo.contributors_count,
                    "forks_count": repo.forks_count,
                    "pushed_at": repo.pushed_at.isoformat(),
                    "description": repo.description,
                    "html_url": repo.html_url,
                    "topics": repo.topics,
                }
            )
            print(f"{i + 1}. {repo.full_name}  [{repo.language}]")
            print(
                f"   Stars: {repo.stargazers_count:<6} | Contributors: "
                f"{str(repo.contributors_count):<6} | Forks: {repo.forks_count:<6} "
                f"| Last Push: {repo.pushed_at.date()}"
            )
            desc = (
                (repo.description[:120] + "...")
                if repo.description and len(repo.description) > 120
                else repo.description
            )
            print(f"   Description: {desc}")
            print(f"   URL: {repo.html_url}\n")

        # --- SAVE OUTPUT FILES ---
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        structured_output_file = (
            OUTPUT_FOLDER / f"quantum_multilang_structured_{timestamp}.json"
        )
        with open(structured_output_file, "w", encoding="utf-8") as f:
            json.dump(structured_results, f, indent=2, ensure_ascii=False)
        print(f"Structured results saved to: {structured_output_file}")

        repo_list_file = OUTPUT_FOLDER / "multilanguage_repo_list.txt"
        with open(repo_list_file, "w", encoding="utf-8") as f:
            for repo in top_repos:
                f.write(f"{repo.full_name}\n")
        print(f"Simple repository list saved to: {repo_list_file}")

        generate_summary_file(
            total_candidates=len(repo_candidates),
            final_repos=top_repos,
            filtered_out_repos=filtered_out_repos,
            output_folder=OUTPUT_FOLDER,
            timestamp=timestamp,
        )

        print("-" * 70)
        print("Search complete.")

    except GithubException as e:
        print(
            f"An error occurred with the GitHub API: {e.status} {e.data}",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    search_github_for_multilang_qc_frameworks()
