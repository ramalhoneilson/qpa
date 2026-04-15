import subprocess
import sys
from pathlib import Path

DEFAULT_TARGET_DIR = Path("target_github_projects")


def run_command(command: list[str], cwd: Path | None = None):
    """Runs a command and returns True on success, False on failure."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,  # This will raise an exception on non-zero exit codes
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Print a clean error message if the command fails
        print(f"    - Command failed: {' '.join(command)}. Error: {e}", file=sys.stderr)
        return False


def main():
    """Main function to process the repository list."""
    # --- 1. Validate Input ---
    if len(sys.argv) < 2:
        print(
            "Error: Please provide the path to the repository list file.",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_list_file = Path(sys.argv[1])
    if not repo_list_file.is_file():
        print(
            f"Error: Repository list file not found at '{repo_list_file}'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Optional second argument overrides the clone destination directory.
    TARGET_DIR = Path(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_TARGET_DIR

    print(f">>> Cloning/updating repositories from '{repo_list_file}'...")
    print(f">>> Target directory: '{TARGET_DIR}'")

    # --- 2. Ensure Target Directory Exists ---
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # --- 3. Read and Process Repositories ---
    with open(repo_list_file, encoding="utf-8") as f:
        repos_to_process = [line.strip() for line in f if line.strip()]

    for org_repo in repos_to_process:
        dir_name = Path(org_repo).name
        # Handle special case for tensorflow/quantum
        if org_repo == "tensorflow/quantum":
            dir_name = "tensorflow-quantum"

        repo_path = TARGET_DIR / dir_name
        repo_url = f"https://github.com/{org_repo}.git"

        print(f"\n--> Processing {org_repo}")

        if repo_path.is_dir():
            print(f"    Updating {dir_name}...")
            # Use cwd to run the command inside the existing directory
            if not run_command(["git", "pull"], cwd=repo_path):
                print(f"    Could not update {dir_name}, continuing...")
        else:
            print(f"    Cloning {dir_name}...")
            run_command(["git", "clone", "--depth", "1", repo_url, str(repo_path)])

    print("\nAll filtered source repositories are up to date.")


if __name__ == "__main__":
    main()
