"""Tests for metadata utility functions."""

import pytest
from langchain_memvid.utils.metadata import get_on_first_match


class TestGetOnFirstMatch:
    """Test cases for get_on_first_match function."""

    def test_basic_functionality(self):
        """Test basic functionality with simple key lookup."""
        metadata = {"key1": "value1", "key2": "value2"}
        result = get_on_first_match(metadata, "key1")
        assert result == "value1"

    def test_first_match_priority(self):
        """Test that the first matching key is returned."""
        metadata = {"key1": "value1", "key2": "value2", "key3": "value3"}
        result = get_on_first_match(metadata, "key1", "key2", "key3")
        assert result == "value1"

    def test_skip_none_values(self):
        """Test that None values are skipped."""
        metadata = {"key1": None, "key2": "value2", "key3": "value3"}
        result = get_on_first_match(metadata, "key1", "key2", "key3")
        assert result == "value2"

    def test_missing_keys(self):
        """Test behavior when keys are missing from metadata."""
        metadata = {"key1": "value1"}
        result = get_on_first_match(metadata, "missing_key", "key1")
        assert result == "value1"

    def test_all_keys_missing(self):
        """Test behavior when all keys are missing."""
        metadata = {"key1": "value1"}
        result = get_on_first_match(metadata, "missing_key1", "missing_key2")
        assert result is None

    def test_custom_default_value(self):
        """Test custom default value when no match is found."""
        metadata = {"key1": "value1"}
        result = get_on_first_match(metadata, "missing_key", default="default_value")
        assert result == "default_value"

    def test_type_checking_with_expected_type(self):
        """Test type checking with explicit expected_type."""
        metadata = {"key1": "string_value", "key2": 42, "key3": "another_string"}

        # Should return string value
        result = get_on_first_match(metadata, "key1", "key2", expected_type=str)
        assert result == "string_value"

        # Should return int value
        result = get_on_first_match(metadata, "key1", "key2", expected_type=int)
        assert result == 42

    def test_type_checking_skip_invalid_types(self):
        """Test that values with wrong types are skipped."""
        metadata = {"key1": "string_value", "key2": 42, "key3": "another_string"}
        result = get_on_first_match(metadata, "key1", "key2", "key3", expected_type=int)
        assert result == 42

    def test_type_checking_no_valid_types(self):
        """Test behavior when no values match the expected type."""
        metadata = {"key1": "string_value", "key2": "another_string"}
        result = get_on_first_match(metadata, "key1", "key2", expected_type=int)
        assert result is None

    def test_type_checking_with_custom_default(self):
        """Test type checking with custom default when no valid types found."""
        metadata = {"key1": "string_value", "key2": "another_string"}
        result = get_on_first_match(metadata, "key1", "key2", expected_type=int, default=100)
        assert result == 100

    def test_empty_metadata(self):
        """Test behavior with empty metadata dictionary."""
        metadata = {}
        result = get_on_first_match(metadata, "key1", "key2")
        assert result is None

    def test_empty_keys_list(self):
        """Test behavior with empty keys list."""
        metadata = {"key1": "value1"}
        result = get_on_first_match(metadata)
        assert result is None

    def test_none_metadata(self):
        """Test behavior with None metadata."""
        with pytest.raises(TypeError):
            get_on_first_match(None, "key1")

    def test_complex_types(self):
        """Test with complex data types."""
        metadata = {
            "list_key": [1, 2, 3],
            "dict_key": {"nested": "value"},
            "tuple_key": (1, 2, 3)
        }

        result = get_on_first_match(metadata, "list_key", "dict_key", expected_type=list)
        assert result == [1, 2, 3]

        result = get_on_first_match(metadata, "dict_key", "list_key", expected_type=dict)
        assert result == {"nested": "value"}

    def test_boolean_values(self):
        """Test with boolean values."""
        metadata = {"true_key": True, "false_key": False, "string_key": "value"}

        result = get_on_first_match(metadata, "true_key", "false_key", expected_type=bool)
        assert result is True

        result = get_on_first_match(metadata, "false_key", "true_key", expected_type=bool)
        assert result is False

    def test_zero_values_not_skipped(self):
        """Test that zero values (which are falsy) are not skipped."""
        metadata = {"zero_key": 0, "string_key": "value"}
        result = get_on_first_match(metadata, "zero_key", "string_key")
        assert result == 0

    def test_empty_string_not_skipped(self):
        """Test that empty strings (which are falsy) are not skipped."""
        metadata = {"empty_key": "", "string_key": "value"}
        result = get_on_first_match(metadata, "empty_key", "string_key")
        assert result == ""

    def test_false_boolean_not_skipped(self):
        """Test that False boolean (which is falsy) is not skipped."""
        metadata = {"false_key": False, "true_key": True}
        result = get_on_first_match(metadata, "false_key", "true_key")
        assert result is False
