import json
import os
import sys
from datetime import UTC, datetime

from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from github import Github, GithubException

from src.conf import config

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")

TARGET_RESULT_COUNT = 200
SORT_BY = "stars"
SORT_ORDER = "desc"


search_queries = [
    "topic:quantum-computing language:Python",
    "topic:quantum-machine-learning language:Python",
    "topic:quantum-algorithms language:Python",
]

# Important frameworks are never missed by topic searches.
known_repos = [
    # Qiskit repos
    "Qiskit/qiskit",
    "qiskit-community/qiskit-algorithms",
    "qiskit-community/qiskit-machine-learning",
    "qiskit-community/qiskit-nature",
    "qiskit-community/qiskit-finance",
    "qiskit-community/qiskit-optimization",
    "qiskit-community/qiskit-dynamics",
    "qiskit-community/qiskit-experiments",
    # Cirq
    "quantumlib/Cirq",
    "quantumlib/ReCirq",
    "quantumlib/Qualtran",
    "quantumlib/OpenFermion",
    # PennyLane
    "PennyLaneAI/pennylane",
    # Rigetti Ecosystem
    "rigetti/pyquil",
    # Other Major Frameworks & Libraries
    "qutip/qutip",
    "qiboteam/qibo",
    "ProjectQ-Framework/ProjectQ",
    "XanaduAI/strawberryfields",
    "eclipse-qrisp/Qrisp",
    "jcmgray/quimb",
    "tencent-quantum-lab/tensorcircuit",
    "Classiq/classiq-library",
    "tensorflow/quantum",
    "mit-han-lab/torchquantum",
    # Amazon Braket Libraries
    "amazon-braket/amazon-braket-sdk-python",
    "amazon-braket/amazon-braket-examples",
    "amazon-braket/amazon-braket-algorithm-library",
]

MIN_STARS = 30
MIN_CONTRIBUTORS = 10
MAX_INACTIVITY_MONTHS = 12

EXCLUSION_KEYWORDS = ["awesome-list", "books"]

OUTPUT_FOLDER = config.PROJECT_ROOT / "data"


def check_for_exclusion(repo):
    """Checks if a repository should be excluded based on keywords."""
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
        return False, "Is a fork"  # Skip forks silently but log for summary
    if repo.stargazers_count < MIN_STARS:
        return False, f"Not enough stars ({repo.stargazers_count} < {MIN_STARS})"

    now = datetime.now(UTC)
    inactivity_threshold = now - relativedelta(months=MAX_INACTIVITY_MONTHS)
    if repo.pushed_at < inactivity_threshold:
        return False, f"Inactive since {repo.pushed_at.date()}"

    is_excluded, reason = check_for_exclusion(repo)
    if is_excluded:
        return False, f"Excluded by {reason}"

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

    summary_file_path = output_folder / f"github_search_summary_{timestamp}.txt"

    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write("GitHub Quantum Projects Search Summary\n")
        f.write("=" * 40 + "\n")
        f.write(
            f"Summary generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        f.write("--- Overall Statistics ---\n")
        f.write(f"Total unique projects found (candidates): {num_candidates}\n")
        f.write(f"Projects filtered out:                    {num_filtered}\n")
        f.write(f"Final projects considered for analysis:   {num_final}\n\n")

        # Table of filtered projects
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

        # List of final projects
        f.write(f"--- Final Considered Projects (Top {num_final}) ---\n")
        if not final_repos:
            f.write("No projects met the criteria.\n")
        else:
            for i, repo in enumerate(final_repos):
                f.write(f"{i + 1}. {repo.full_name}\n")

        f.write("\n" + "=" * 40 + "\n")
        f.write("End of Summary\n")

    print(f"Search summary saved to: {summary_file_path}")


def search_github_for_qc_frameworks():
    """Searches GitHub using multiple queries and a known list, then filters."""
    if not GITHUB_TOKEN:
        print("Error: GitHub PAT/TOKEN not found.", file=sys.stderr)
        return

    try:
        g = Github(GITHUB_TOKEN)
        repo_candidates = {}

        print("--- Phase 1: Fetching known repositories ---")
        for repo_name in known_repos:
            try:
                repo = g.get_repo(repo_name)
                repo_candidates[repo.full_name] = repo
                print(f"  [OK] Fetched {repo.full_name}")
            except GithubException as e:
                print(
                    f"Warning: Could not fetch known repo '{repo_name}': {e.status}",
                    file=sys.stderr,
                )

        print("\n--- Phase 2: Discovering new repositories via search ---")
        for query in search_queries:
            print(f"Searching with query: '{query}'...")
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

        print(
            f"\nGathered a total of {len(repo_candidates)} unique candidate repositories."
        )

        print("\n--- Phase 3: Applying quality filters ---")
        print(
            f" - Min Stars: {MIN_STARS}, Min Contributors: {MIN_CONTRIBUTORS}, Last push <= {MAX_INACTIVITY_MONTHS} months"
        )
        print(f" - Excluding keywords: {', '.join(EXCLUSION_KEYWORDS[:4])}...")
        print("-" * 70)

        final_repos = []
        filtered_out_repos = []
        for repo in repo_candidates.values():
            is_relevant, detail = is_repo_relevant(repo)
            if is_relevant:
                # Add contributor count to the repo object for later use
                repo.contributors_count = detail
                final_repos.append(repo)
            else:
                if detail != "Is a fork":  # Don't print for silent fork filtering
                    print(f"[SKIPPING] {repo.full_name}: {detail}.")
                filtered_out_repos.append(
                    {"full_name": repo.full_name, "reason": detail}
                )

        # Sort the final list by stars and handle truncation
        final_repos.sort(key=lambda r: r.stargazers_count, reverse=True)
        top_repos = final_repos[:TARGET_RESULT_COUNT]

        # Add repos that passed filters but were not in the top N to the filtered list
        repos_culled_by_rank = final_repos[TARGET_RESULT_COUNT:]
        for repo in repos_culled_by_rank:
            filtered_out_repos.append(
                {
                    "full_name": repo.full_name,
                    "reason": f"Passed filters but not in top {TARGET_RESULT_COUNT} by stars ({repo.stargazers_count} stars)",
                }
            )

        print("-" * 70)
        print(
            f"Found {len(top_repos)} matching repositories after filtering {len(repo_candidates)} candidates.\n"
        )

        structured_results = []
        for i, repo in enumerate(top_repos):
            structured_results.append(
                {
                    "rank": i + 1,
                    "full_name": repo.full_name,
                    "stargazers_count": repo.stargazers_count,
                    "contributors_count": repo.contributors_count,
                    "forks_count": repo.forks_count,
                    "pushed_at": repo.pushed_at.isoformat(),
                    "description": repo.description,
                    "html_url": repo.html_url,
                }
            )
            print(f"{i + 1}. {repo.full_name}")
            print(
                f"   Stars: {repo.stargazers_count:<6} | Contributors: {str(repo.contributors_count):<6} | Forks: {repo.forks_count:<6} | Last Push: {repo.pushed_at.date()}"
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

        # Save structured JSON results
        structured_output_file = (
            OUTPUT_FOLDER / f"quantum_frameworks_structured_{timestamp}.json"
        )
        with open(structured_output_file, "w", encoding="utf-8") as f:
            json.dump(structured_results, f, indent=2, ensure_ascii=False)
        print(f"Structured results saved to: {structured_output_file}")

        # Save simple list for automation
        repo_list_file = OUTPUT_FOLDER / "filtered_repo_list.txt"
        with open(repo_list_file, "w", encoding="utf-8") as f:
            for repo in top_repos:
                f.write(f"{repo.full_name}\n")
        print(f"Simple repository list saved to: {repo_list_file}")

        # Generate and save the summary report
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
    search_github_for_qc_frameworks()
