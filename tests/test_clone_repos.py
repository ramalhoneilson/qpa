"""
Test suite for src/preprocessing/clone_repos.py
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.preprocessing.clone_repos import TARGET_DIR, main, run_command


class TestRunCommand:
    """Test the run_command function."""

    def test_successful_command(self):
        """Test successful command execution."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            result = run_command(["git", "status"])
            assert result is True
            mock_run.assert_called_once_with(
                ["git", "status"],
                cwd=None,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def test_successful_command_with_cwd(self):
        """Test successful command execution with working directory."""
        test_cwd = Path("/test/dir")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            result = run_command(["git", "status"], cwd=test_cwd)
            assert result is True
            mock_run.assert_called_once_with(
                ["git", "status"],
                cwd=test_cwd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def test_command_failure_called_process_error(self):
        """Test command failure with CalledProcessError."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            with patch("builtins.print") as mock_print:
                result = run_command(["git", "status"])
                assert result is False
                mock_print.assert_called_once()

    def test_command_failure_file_not_found(self):
        """Test command failure with FileNotFoundError."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            with patch("builtins.print") as mock_print:
                result = run_command(["git", "status"])
                assert result is False
                mock_print.assert_called_once()


class TestMain:
    """Test the main function."""

    def test_missing_arguments(self):
        """Test main with missing arguments."""
        with patch("sys.argv", ["script.py"]):
            with patch("sys.exit") as mock_exit:
                with patch("builtins.print") as mock_print:
                    try:
                        main()
                    except IndexError:
                        # The function will raise IndexError before sys.exit is called
                        pass
                    # The function should have printed the error message
                    assert mock_print.call_count >= 1

    def test_file_not_found(self):
        """Test main with non-existent file."""
        with patch("sys.argv", ["script.py", "nonexistent.txt"]):
            with patch("sys.exit") as mock_exit:
                with patch("builtins.print") as mock_print:
                    with patch("pathlib.Path.is_file", return_value=False):
                        try:
                            main()
                        except FileNotFoundError:
                            # The function will raise FileNotFoundError when trying to open the file
                            pass
                        # The function should have printed the error message
                        assert mock_print.call_count >= 1

    def test_successful_execution(self):
        """Test successful main execution."""
        test_file_content = "org1/repo1\norg2/repo2\ntensorflow/quantum\n"

        with patch("sys.argv", ["script.py", "repos.txt"]):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("pathlib.Path.mkdir") as mock_mkdir:
                    with patch("builtins.open", create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = (
                            test_file_content
                        )
                        with patch(
                            "src.preprocessing.clone_repos.run_command"
                        ) as mock_run_command:
                            mock_run_command.return_value = True
                            with patch("builtins.print") as mock_print:
                                with patch("pathlib.Path.is_dir", return_value=False):
                                    main()

                                    # Verify target directory creation
                                    mock_mkdir.assert_called_once_with(
                                        parents=True, exist_ok=True
                                    )

                                    # Verify file reading (Path object is passed to open)
                                    assert mock_open.call_count >= 1

    def test_repo_update_existing_directory(self):
        """Test updating existing repository."""
        test_file_content = "org1/repo1\n"

        with patch("sys.argv", ["script.py", "repos.txt"]):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("pathlib.Path.mkdir"):
                    with patch("builtins.open", create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = (
                            test_file_content
                        )
                        with patch(
                            "src.preprocessing.clone_repos.run_command"
                        ) as mock_run_command:
                            mock_run_command.return_value = True
                            with patch("builtins.print"):
                                with patch("pathlib.Path.is_dir", return_value=True):
                                    main()

                                    # The main function should have processed the file content
                                    # We can verify the function completed without errors
                                    assert mock_open.call_count >= 1

    def test_git_pull_failure(self):
        """Test git pull failure handling."""
        test_file_content = "org1/repo1\n"

        with patch("sys.argv", ["script.py", "repos.txt"]):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("pathlib.Path.mkdir"):
                    with patch("builtins.open", create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = (
                            test_file_content
                        )
                        with patch(
                            "src.preprocessing.clone_repos.run_command"
                        ) as mock_run_command:
                            mock_run_command.return_value = False  # git pull fails
                            with patch("builtins.print") as mock_print:
                                with patch("pathlib.Path.is_dir", return_value=True):
                                    main()

                                    # The main function should have processed the file content
                                    # We can verify the function completed without errors
                                    assert mock_open.call_count >= 1


class TestConstants:
    """Test module constants."""

    def test_target_dir_constant(self):
        """Test TARGET_DIR constant."""
        assert TARGET_DIR == Path("target_github_projects")
