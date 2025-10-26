"""
Tests for src/core_concepts/extractor/storage.py

This module tests the ConceptStorage class functionality for saving
extracted concept data to JSON and CSV formats.
"""

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core_concepts.extractor.storage import ConceptStorage, Concept


class TestConceptStorage:
    """Test the ConceptStorage class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directories for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        self.json_path = self.temp_dir / "test_concepts.json"
        self.csv_path = self.temp_dir / "test_concepts.csv"
        self.storage = ConceptStorage(self.json_path, self.csv_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test ConceptStorage initialization."""
        assert self.storage.json_path == self.json_path
        assert self.storage.csv_path == self.csv_path

    def test_initialization_with_different_paths(self):
        """Test initialization with different path types."""
        json_path = Path("/some/path/concepts.json")
        csv_path = Path("/other/path/concepts.csv")
        storage = ConceptStorage(json_path, csv_path)
        
        assert storage.json_path == json_path
        assert storage.csv_path == csv_path

    def test_save_all_empty_data(self):
        """Test save_all with empty data."""
        with patch('logging.warning') as mock_warning:
            self.storage.save_all([])
            mock_warning.assert_called_once_with(
                "No concepts data was provided to save. Skipping all storage operations."
            )

    def test_save_all_none_data(self):
        """Test save_all with None data."""
        with patch('logging.warning') as mock_warning:
            self.storage.save_all(None)
            mock_warning.assert_called_once_with(
                "No concepts data was provided to save. Skipping all storage operations."
            )

    def test_save_all_creates_directories(self):
        """Test that save_all creates parent directories."""
        # Create paths with non-existent parent directories
        deep_json_path = self.temp_dir / "deep" / "nested" / "concepts.json"
        deep_csv_path = self.temp_dir / "another" / "deep" / "concepts.csv"
        storage = ConceptStorage(deep_json_path, deep_csv_path)
        
        test_data = [{"name": "test", "summary": "test summary"}]
        
        with patch.object(storage, 'save_as_json') as mock_json, \
             patch.object(storage, 'save_as_csv') as mock_csv:
            storage.save_all(test_data)
            
            # Check that directories were created
            assert deep_json_path.parent.exists()
            assert deep_csv_path.parent.exists()
            
            # Check that save methods were called
            mock_json.assert_called_once_with(test_data)
            mock_csv.assert_called_once_with(test_data)

    def test_save_as_json_success(self):
        """Test successful JSON saving."""
        test_data = [
            {
                "name": "concept1",
                "summary": "First concept summary",
                "source_code": "def func(): pass",
                "other_field": "value"
            },
            {
                "name": "concept2", 
                "summary": "Second concept summary",
                "source_code": "class MyClass: pass",
                "extra_field": "extra_value"
            }
        ]
        
        with patch('logging.info') as mock_info:
            self.storage.save_as_json(test_data)
            
            # Check that JSON file was created
            assert self.json_path.exists()
            
            # Check file content
            with open(self.json_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            # Should have 2 items
            assert len(saved_data) == 2
            
            # Should not contain source_code
            for item in saved_data:
                assert "source_code" not in item
            
            # Should contain other fields
            assert saved_data[0]["name"] == "concept1"
            assert saved_data[0]["summary"] == "First concept summary"
            assert saved_data[0]["other_field"] == "value"
            
            assert saved_data[1]["name"] == "concept2"
            assert saved_data[1]["summary"] == "Second concept summary"
            assert saved_data[1]["extra_field"] == "extra_value"
            
            # Check logging calls
            assert mock_info.call_count == 2
            mock_info.assert_any_call("Saving 2 concepts to JSON: '{}'".format(self.json_path))
            mock_info.assert_any_call("Successfully saved JSON file.")

    def test_save_as_json_empty_data(self):
        """Test JSON saving with empty data."""
        with patch('logging.info') as mock_info:
            self.storage.save_as_json([])
            
            assert self.json_path.exists()
            
            with open(self.json_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            assert saved_data == []
            mock_info.assert_any_call("Saving 0 concepts to JSON: '{}'".format(self.json_path))

    def test_save_as_json_without_source_code(self):
        """Test JSON saving when data doesn't have source_code field."""
        test_data = [
            {"name": "concept1", "summary": "summary1"},
            {"name": "concept2", "summary": "summary2"}
        ]
        
        self.storage.save_as_json(test_data)
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 2
        assert saved_data[0] == {"name": "concept1", "summary": "summary1"}
        assert saved_data[1] == {"name": "concept2", "summary": "summary2"}

    def test_save_as_json_exception_handling(self):
        """Test JSON saving with file I/O exception."""
        test_data = [{"name": "test", "summary": "test"}]
        
        # Mock the json.dump function to raise an exception
        with patch('json.dump', side_effect=IOError("Permission denied")), \
             patch('logging.error') as mock_error:
            self.storage.save_as_json(test_data)
            
            mock_error.assert_called_once()
            error_call = mock_error.call_args[0][0]
            assert "Failed to save data to JSON file" in error_call
            assert str(self.json_path) in error_call
            assert "Permission denied" in error_call

    def test_save_as_csv_success(self):
        """Test successful CSV saving."""
        test_data = [
            {
                "name": "concept1",
                "summary": "First concept summary",
                "source_code": "def func(): pass",
                "other_field": "value"
            },
            {
                "name": "concept2",
                "summary": "Second concept summary", 
                "source_code": "class MyClass: pass",
                "extra_field": "extra_value"
            }
        ]
        
        with patch('logging.info') as mock_info:
            self.storage.save_as_csv(test_data)
            
            # Check that CSV file was created
            assert self.csv_path.exists()
            
            # Check file content
            with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f, delimiter=';')
                rows = list(reader)
            
            # Should have 2 data rows plus header
            assert len(rows) == 2
            
            # Check headers
            assert set(rows[0].keys()) == {"name", "summary"}
            
            # Check data
            assert rows[0]["name"] == "concept1"
            assert rows[0]["summary"] == "First concept summary"
            assert rows[1]["name"] == "concept2"
            assert rows[1]["summary"] == "Second concept summary"
            
            # Check logging calls
            assert mock_info.call_count == 2
            mock_info.assert_any_call("Saving 2 summaries to CSV: '{}'".format(self.csv_path))
            mock_info.assert_any_call("Successfully saved CSV file.")

    def test_save_as_csv_empty_data(self):
        """Test CSV saving with empty data."""
        with patch('logging.info') as mock_info:
            self.storage.save_as_csv([])
            
            assert self.csv_path.exists()
            
            with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f, delimiter=';')
                rows = list(reader)
            
            # Should have only headers, no data rows
            assert len(rows) == 0
            mock_info.assert_any_call("Saving 0 summaries to CSV: '{}'".format(self.csv_path))

    def test_save_as_csv_missing_fields(self):
        """Test CSV saving with missing name or summary fields."""
        test_data = [
            {"name": "concept1"},  # Missing summary
            {"summary": "summary2"},  # Missing name
            {"name": "concept3", "summary": "summary3"}  # Complete
        ]
        
        self.storage.save_as_csv(test_data)
        
        with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            rows = list(reader)
        
        assert len(rows) == 3
        assert rows[0]["name"] == "concept1"
        assert rows[0]["summary"] == ""
        assert rows[1]["name"] == ""
        assert rows[1]["summary"] == "summary2"
        assert rows[2]["name"] == "concept3"
        assert rows[2]["summary"] == "summary3"

    def test_save_as_csv_exception_handling(self):
        """Test CSV saving with file I/O exception."""
        test_data = [{"name": "test", "summary": "test"}]
        
        # Mock the csv.DictWriter.writerow method to raise an exception
        with patch('csv.DictWriter.writerow', side_effect=IOError("Permission denied")), \
             patch('logging.error') as mock_error:
            self.storage.save_as_csv(test_data)
            
            mock_error.assert_called_once()
            error_call = mock_error.call_args[0][0]
            assert "Failed to save data to CSV file" in error_call
            assert str(self.csv_path) in error_call
            assert "Permission denied" in error_call

    def test_save_as_csv_delimiter(self):
        """Test that CSV uses semicolon delimiter."""
        test_data = [{"name": "test,with,commas", "summary": "summary;with;semicolons"}]
        
        self.storage.save_as_csv(test_data)
        
        with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
            content = f.read()
        
        # Should contain semicolons as delimiters
        assert ";" in content
        # Should not contain commas as delimiters (except in data)
        lines = content.strip().split('\n')
        assert len(lines) == 2  # header + data
        # Handle potential carriage return characters
        assert lines[0].strip() == "name;summary"
        # The CSV writer may quote fields with semicolons, so we check for the presence of semicolons
        assert "test,with,commas" in lines[1]
        assert "summary;with;semicolons" in lines[1] or '"summary;with;semicolons"' in lines[1]

    def test_save_all_integration(self):
        """Test the complete save_all workflow."""
        test_data = [
            {
                "name": "integration_test",
                "summary": "Integration test summary",
                "source_code": "def test(): pass",
                "extra_field": "extra"
            }
        ]
        
        with patch('logging.info') as mock_info:
            self.storage.save_all(test_data)
            
            # Check both files were created
            assert self.json_path.exists()
            assert self.csv_path.exists()
            
            # Check JSON content (should exclude source_code)
            with open(self.json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            assert len(json_data) == 1
            assert "source_code" not in json_data[0]
            assert json_data[0]["name"] == "integration_test"
            assert json_data[0]["extra_field"] == "extra"
            
            # Check CSV content (should have only name and summary)
            with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f, delimiter=';')
                csv_data = list(reader)
            assert len(csv_data) == 1
            assert csv_data[0]["name"] == "integration_test"
            assert csv_data[0]["summary"] == "Integration test summary"
            
            # Check logging calls
            assert mock_info.call_count >= 4  # At least 4 info messages

    def test_concept_type_alias(self):
        """Test that Concept type alias is properly defined."""
        # This is more of a type checking test
        concept: Concept = {"name": "test", "summary": "test", "source_code": "pass"}
        assert isinstance(concept, dict)
        assert "name" in concept
        assert "summary" in concept
        assert "source_code" in concept


class TestConceptStorageEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.json_path = self.temp_dir / "test.json"
        self.csv_path = self.temp_dir / "test.csv"
        self.storage = ConceptStorage(self.json_path, self.csv_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_save_with_none_values(self):
        """Test saving data with None values."""
        test_data = [
            {"name": None, "summary": "valid summary"},
            {"name": "valid name", "summary": None},
            {"name": None, "summary": None}
        ]
        
        self.storage.save_all(test_data)
        
        # Check CSV handles None values
        with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            rows = list(reader)
        
        assert len(rows) == 3
        assert rows[0]["name"] == ""
        assert rows[0]["summary"] == "valid summary"
        assert rows[1]["name"] == "valid name"
        assert rows[1]["summary"] == ""
        assert rows[2]["name"] == ""
        assert rows[2]["summary"] == ""

    def test_save_with_empty_strings(self):
        """Test saving data with empty strings."""
        test_data = [
            {"name": "", "summary": "valid summary"},
            {"name": "valid name", "summary": ""},
            {"name": "", "summary": ""}
        ]
        
        self.storage.save_all(test_data)
        
        # Check CSV handles empty strings
        with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            rows = list(reader)
        
        assert len(rows) == 3
        assert rows[0]["name"] == ""
        assert rows[0]["summary"] == "valid summary"
        assert rows[1]["name"] == "valid name"
        assert rows[1]["summary"] == ""
        assert rows[2]["name"] == ""
        assert rows[2]["summary"] == ""

    def test_save_with_special_characters(self):
        """Test saving data with special characters."""
        test_data = [
            {
                "name": "concept with émojis 🚀 and ñ",
                "summary": "Summary with special chars: <>&\"'",
                "source_code": "def func(): # comment with émojis 🎉\n    pass"
            }
        ]
        
        self.storage.save_all(test_data)
        
        # Check JSON preserves special characters
        with open(self.json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert json_data[0]["name"] == "concept with émojis 🚀 and ñ"
        assert json_data[0]["summary"] == "Summary with special chars: <>&\"'"
        assert "source_code" not in json_data[0]  # Should be excluded
        
        # Check CSV preserves special characters
        with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            csv_data = list(reader)
        
        assert csv_data[0]["name"] == "concept with émojis 🚀 and ñ"
        assert csv_data[0]["summary"] == "Summary with special chars: <>&\"'"

    def test_save_with_large_data(self):
        """Test saving large amounts of data."""
        # Create 1000 concepts
        test_data = []
        for i in range(1000):
            test_data.append({
                "name": f"concept_{i}",
                "summary": f"Summary for concept {i}",
                "source_code": f"def func_{i}(): pass",
                "extra_field": f"value_{i}"
            })
        
        with patch('logging.info') as mock_info:
            self.storage.save_all(test_data)
            
            # Check files were created
            assert self.json_path.exists()
            assert self.csv_path.exists()
            
            # Check JSON has correct number of items
            with open(self.json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            assert len(json_data) == 1000
            
            # Check CSV has correct number of rows
            with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f, delimiter=';')
                csv_data = list(reader)
            assert len(csv_data) == 1000
            
            # Check logging
            mock_info.assert_any_call("Saving 1000 concepts to JSON: '{}'".format(self.json_path))
            mock_info.assert_any_call("Saving 1000 summaries to CSV: '{}'".format(self.csv_path))

    def test_directory_creation_permissions(self):
        """Test behavior when directory creation fails."""
        # Create a path that would require permissions we don't have
        restricted_path = Path("/root/restricted/concepts.json")
        storage = ConceptStorage(restricted_path, self.csv_path)
        
        test_data = [{"name": "test", "summary": "test"}]
        
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                storage.save_all(test_data)


class TestConceptStorageLogging:
    """Test logging behavior of ConceptStorage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.json_path = self.temp_dir / "test.json"
        self.csv_path = self.temp_dir / "test.csv"
        self.storage = ConceptStorage(self.json_path, self.csv_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_logging_info_messages(self):
        """Test that appropriate info messages are logged."""
        test_data = [{"name": "test", "summary": "test"}]
        
        with patch('logging.info') as mock_info:
            self.storage.save_all(test_data)
            
            # Check that info messages were logged
            info_calls = [call[0][0] for call in mock_info.call_args_list]
            
            assert any("Saving 1 concepts to JSON" in msg for msg in info_calls)
            assert any("Successfully saved JSON file" in msg for msg in info_calls)
            assert any("Saving 1 summaries to CSV" in msg for msg in info_calls)
            assert any("Successfully saved CSV file" in msg for msg in info_calls)

    def test_logging_warning_empty_data(self):
        """Test that warning is logged for empty data."""
        with patch('logging.warning') as mock_warning:
            self.storage.save_all([])
            mock_warning.assert_called_once_with(
                "No concepts data was provided to save. Skipping all storage operations."
            )

    def test_logging_error_json_failure(self):
        """Test that error is logged when JSON saving fails."""
        test_data = [{"name": "test", "summary": "test"}]
        
        # Mock the json.dump function to raise an exception
        with patch('json.dump', side_effect=IOError("JSON write failed")), \
             patch('logging.error') as mock_error:
            self.storage.save_as_json(test_data)
            
            mock_error.assert_called_once()
            error_msg = mock_error.call_args[0][0]
            assert "Failed to save data to JSON file" in error_msg
            assert str(self.json_path) in error_msg
            assert "JSON write failed" in error_msg

    def test_logging_error_csv_failure(self):
        """Test that error is logged when CSV saving fails."""
        test_data = [{"name": "test", "summary": "test"}]
        
        # Mock the csv.DictWriter.writerow method to raise an exception
        with patch('csv.DictWriter.writerow', side_effect=IOError("CSV write failed")), \
             patch('logging.error') as mock_error:
            self.storage.save_as_csv(test_data)
            
            mock_error.assert_called_once()
            error_msg = mock_error.call_args[0][0]
            assert "Failed to save data to CSV file" in error_msg
            assert str(self.csv_path) in error_msg
            assert "CSV write failed" in error_msg
