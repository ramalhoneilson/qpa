import shutil
from src.conf import config


TARGET_PROJECTS_BASE_PATH = config.TARGET_PROJECTS_BASE_PATH
TARGET_PROJECTS = config.TARGET_PROJECTS
NOTEBOOKS_DEST_ROOT = config.PROJECT_ROOT / "notebooks"


def archive_notebooks():
    """
    Finds all Jupyter notebooks in target projects and copies them to a
    central directory, preserving their original folder structure.
    """
    print("--- Starting Notebook Archiving Step ---")
    print(f"Source projects location: {TARGET_PROJECTS_BASE_PATH}")
    print(f"Destination archive:      {NOTEBOOKS_DEST_ROOT}")

    notebook_count = 0
    error_count = 0

    for project_subpath in TARGET_PROJECTS:
        source_project_path = TARGET_PROJECTS_BASE_PATH / project_subpath
        project_name_for_dir = project_subpath.strip("/").replace("/", "_")

        if not source_project_path.is_dir():
            print(
                f"\n[WARNING] Project path not found, skipping: {source_project_path}"
            )
            continue

        print(f"\nProcessing project: {project_subpath}...")

        try:
            for source_notebook_path in source_project_path.rglob("*.ipynb"):
                relative_path = source_notebook_path.relative_to(source_project_path)
                dest_path = NOTEBOOKS_DEST_ROOT / project_name_for_dir / relative_path

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_notebook_path, dest_path)
                notebook_count += 1
        except Exception as e:
            print(f"[ERROR] Failed processing {project_subpath}: {e}")
            error_count += 1

    print("\n--- Archiving Summary ---")
    print(f"Successfully archived: {notebook_count} notebooks.")
    if error_count > 0:
        print(f"Errors encountered:    {error_count} projects.")
    print("-------------------------")


if __name__ == "__main__":
    archive_notebooks()
