"""
Unit tests for IPython extension module.

Tests cover all functionality including magic commands, path collection,
AST transformation, and utility functions.
"""

import ast
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import dataclass

from ipykernel_memvid_extension import IPYTHON_INSTALLED

# Skip all tests if IPython is not installed
pytestmark = pytest.mark.skipif(
    not IPYTHON_INSTALLED,
    reason="IPython is not installed"
)


from ipykernel_memvid_extension import (  # noqa: E402
    ensure_import_module,
    delete_path,
    DisplayContainer,
    strip_version,
    type_to_confirm,
    PathCollectorMeta,
    RegisterPathFunc,
    PathCollectorTransformer,
    MemvidMagics,
    load_ipython_extension
)


@pytest.fixture
def mock_shell():
    """Create a mock IPython shell for testing."""
    shell = Mock()
    shell.user_ns = {}
    shell.last_execution_result = Mock()
    shell.last_execution_result.success = True
    shell.ast_transformers = []
    shell.events = Mock()
    shell.events.register = Mock()
    shell.register_magics = Mock()
    shell.run_line_magic = Mock()
    # Make the shell a proper Configurable for traitlets
    shell.__class__.__name__ = 'InteractiveShell'
    return shell


@pytest.fixture
def temp_path(tmp_path):
    """Create a temporary path for testing."""
    return Path(tmp_path) / "test_file.txt"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_ensure_import_module_existing(self):
        """Test ensure_import_module with existing module."""
        result = ensure_import_module('pathlib')
        assert result is not None
        assert result.__name__ == 'pathlib'

    @patch('ipykernel_memvid_extension.importlib.import_module')
    def test_ensure_import_module_missing_with_shell(self, mock_import, mock_shell):
        """Test ensure_import_module with missing module and shell."""
        mock_import.side_effect = [ImportError, Mock()]
        mock_shell.run_line_magic = Mock()

        result = ensure_import_module('nonexistent_module', shell=mock_shell)

        assert result is not None
        mock_shell.run_line_magic.assert_called_once_with('pip', 'install nonexistent_module')
        assert mock_import.call_count == 2

    @patch('ipykernel_memvid_extension.importlib.import_module')
    @patch('ipykernel_memvid_extension.get_ipython')
    def test_ensure_import_module_missing_without_shell(self, mock_get_ipython, mock_import):
        """Test ensure_import_module with missing module without shell."""
        mock_import.side_effect = ImportError
        mock_shell = Mock()
        mock_get_ipython.return_value = mock_shell

        with pytest.raises(ImportError):
            ensure_import_module('nonexistent_module')

    def test_delete_path_file(self, temp_path):
        """Test delete_path with a file."""
        temp_path.write_text("test content")
        assert temp_path.exists()

        result = delete_path(temp_path)

        assert result is True
        assert not temp_path.exists()

    def test_delete_path_directory(self, tmp_path):
        """Test delete_path with a directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        (test_dir / "test_file.txt").write_text("test")
        assert test_dir.exists()

        result = delete_path(test_dir, force=True)

        assert result is True
        assert not test_dir.exists()

    def test_delete_path_nonexistent(self):
        """Test delete_path with nonexistent path."""
        nonexistent_path = Path("/nonexistent/path")
        result = delete_path(nonexistent_path)
        assert result is False

    def test_delete_path_invalid_type(self):
        """Test delete_path with invalid type."""
        result = delete_path("not a path")
        assert result is False

    @patch('ipykernel_memvid_extension.HTML')
    def test_make_html_container_as_class_method(self, mock_HTML):
        """Test make_html_container function."""
        DisplayContainer.make_html_container("Test message", "blue")

        mock_HTML.assert_called_once()
        call_args = mock_HTML.call_args[0][0]
        # Check the HTML content
        html_str = str(call_args)
        assert "Test message" in html_str
        assert "blue" in html_str

    @patch('ipykernel_memvid_extension.HTML')
    def test_make_html_container_as_instance_method(self, mock_HTML):
        """Test make_html_container function."""
        DisplayContainer.ALERT.make_html_container("Test message", "blue")

        mock_HTML.assert_called_once()
        call_args = mock_HTML.call_args[0][0]
        # Check the HTML content
        html_str = str(call_args)
        assert "Test message" in html_str
        assert "red" in html_str

    @patch('ipykernel_memvid_extension.display')
    def test_display_alert(self, mock_display):
        """Test display_alert function."""
        DisplayContainer.ALERT.display("Alert message")

        mock_display.assert_called_once()
        call_args = mock_display.call_args[0][0]
        # Check the HTML content
        html_str = str(call_args.data)
        assert "Alert message" in html_str
        assert "red" in html_str

    @patch('ipykernel_memvid_extension.display')
    def test_display_notification(self, mock_display):
        """Test display_notification function."""
        DisplayContainer.NOTIFICATION.display("Notification message")

        mock_display.assert_called_once()
        call_args = mock_display.call_args[0][0]
        # Check the HTML content
        html_str = str(call_args.data)
        assert "Notification message" in html_str
        assert "green" in html_str

    @patch('ipykernel_memvid_extension.display')
    def test_display_warning(self, mock_display):
        """Test display_warning function."""
        DisplayContainer.WARNING.display("Warning message")

        mock_display.assert_called_once()
        call_args = mock_display.call_args[0][0]
        # Check the HTML content
        html_str = str(call_args.data)
        assert "Warning message" in html_str
        assert "orange" in html_str

    def test_strip_version(self):
        """Test strip_version function."""
        assert strip_version("package==1.0.0") == "package"
        assert strip_version("package>=1.0.0") == "package"
        assert strip_version("package<=1.0.0") == "package"
        assert strip_version("package") == "package"
        assert strip_version("package-name") == "package-name"

    @patch('builtins.input')
    def test_type_to_confirm_success(self, mock_input):
        """Test type_to_confirm with successful confirmation."""
        mock_input.return_value = "test_dir"

        result = type_to_confirm("Are you sure?", "test_dir")

        assert result is True
        mock_input.assert_called_once_with("Are you sure? Type to confirm: \"test_dir\"")

    @patch('builtins.input')
    def test_type_to_confirm_wrong_input_then_correct(self, mock_input):
        """Test type_to_confirm with wrong input followed by correct input."""
        mock_input.side_effect = ["wrong_input", "test_dir"]

        result = type_to_confirm("Are you sure?", "test_dir")

        assert result is True
        assert mock_input.call_count == 2
        mock_input.assert_any_call("Are you sure? Type to confirm: \"test_dir\"")

    @patch('builtins.input')
    def test_type_to_confirm_empty_input(self, mock_input):
        """Test type_to_confirm with empty input."""
        mock_input.return_value = ""

        result = type_to_confirm("Are you sure?", "test_dir")

        assert result is False
        mock_input.assert_called_once_with("Are you sure? Type to confirm: \"test_dir\"")

    @patch('builtins.input')
    def test_type_to_confirm_keyboard_interrupt(self, mock_input):
        """Test type_to_confirm with KeyboardInterrupt."""
        mock_input.side_effect = KeyboardInterrupt()

        result = type_to_confirm("Are you sure?", "test_dir")

        assert result is False
        mock_input.assert_called_once_with("Are you sure? Type to confirm: \"test_dir\"")

    @patch('builtins.input')
    def test_type_to_confirm_empty_confirmation_string(self, mock_input):
        """Test type_to_confirm with empty confirmation string."""
        mock_input.return_value = ""

        result = type_to_confirm("Are you sure?", "")

        assert result is True
        mock_input.assert_called_once_with("Are you sure? Type to confirm: \"\"")


class TestPathCollectorMeta:
    """Test PathCollectorMeta metaclass."""

    def test_metaclass_creation(self):
        """Test that PathCollectorMeta creates a callable class."""
        class TestCollector(metaclass=PathCollectorMeta):
            pass

        assert hasattr(TestCollector, 'paths')
        assert isinstance(TestCollector.paths, list)
        assert callable(TestCollector)

    def test_metaclass_call_with_path(self):
        """Test calling the class with a Path object."""
        class TestCollector(metaclass=PathCollectorMeta):
            pass

        test_path = Path("/test/path")
        result = TestCollector(test_path)

        assert result == test_path
        assert test_path in TestCollector.paths

    def test_metaclass_call_with_invalid_args(self):
        """Test calling the class with invalid arguments."""
        class TestCollector(metaclass=PathCollectorMeta):
            pass

        with pytest.raises(TypeError):
            TestCollector("not a path")

        with pytest.raises(TypeError):
            TestCollector(Path("/test"), extra_arg="value")

        with pytest.raises(TypeError):
            TestCollector()


class TestRegisterPath:
    """Test RegisterPathFunc function."""

    def test_RegisterPathFunc_initialization(self):
        """Test that RegisterPathFunc is properly initialized."""
        assert hasattr(RegisterPathFunc, 'paths')
        assert isinstance(RegisterPathFunc.paths, list)

    def test_RegisterPathFunc_with_path(self):
        """Test RegisterPathFunc with a Path object."""
        # Clear existing paths
        RegisterPathFunc.paths.clear()

        test_path = Path("/test/path")
        result = RegisterPathFunc(test_path)

        assert result == test_path
        assert test_path in RegisterPathFunc.paths

    def test_RegisterPathFunc_with_invalid_args(self):
        """Test RegisterPathFunc with invalid arguments."""
        with pytest.raises(TypeError):
            RegisterPathFunc("not a path")

        with pytest.raises(TypeError):
            RegisterPathFunc(Path("/test"), extra_arg="value")

        with pytest.raises(TypeError):
            RegisterPathFunc()


class TestPathCollectorTransformer:
    """Test PathCollectorTransformer AST transformer."""

    def test_visit_call_with_path(self):
        """Test visiting a Call node with Path constructor."""
        transformer = PathCollectorTransformer()

        # Create AST for: Path("/test/path")
        path_call = ast.Call(
            func=ast.Name(id='Path', ctx=ast.Load()),
            args=[ast.Constant(value="/test/path")],
            keywords=[]
        )

        result = transformer.visit_Call(path_call)

        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Name)
        assert result.func.id == 'RegisterPathFunc'
        assert len(result.args) == 1
        assert result.args[0] == path_call

    def test_visit_call_with_other_function(self):
        """Test visiting a Call node with non-Path function."""
        transformer = PathCollectorTransformer()

        # Create AST for: other_function()
        other_call = ast.Call(
            func=ast.Name(id='other_function', ctx=ast.Load()),
            args=[],
            keywords=[]
        )

        result = transformer.visit_Call(other_call)

        # Should return the original call unchanged
        assert result == other_call

    def test_visit_call_with_path_static_method(self):
        """Test visiting a Call node with Path static method."""
        transformer = PathCollectorTransformer()

        # Create AST for: Path.cwd()
        path_cwd_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )

        result = transformer.visit_Call(path_cwd_call)

        # Should return the original call unchanged (not wrapped in RegisterPathFunc)
        assert result == path_cwd_call

    def test_visit_binop_with_path_division(self):
        """Test visiting a BinOp node with path division operation."""
        transformer = PathCollectorTransformer()

        # Create AST for: Path.cwd() / "subdir"
        path_cwd = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        subdir = ast.Constant(value="subdir")
        binop = ast.BinOp(left=path_cwd, op=ast.Div(), right=subdir)

        result = transformer.visit_BinOp(binop)

        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Name)
        assert result.func.id == 'RegisterPathFunc'
        assert len(result.args) == 1
        assert result.args[0] == binop

    def test_visit_binop_with_path_floor_division(self):
        """Test visiting a BinOp node with path floor division operation."""
        transformer = PathCollectorTransformer()

        # Create AST for: Path.cwd() // "subdir" (using Path.cwd() instead of just 'path')
        path_cwd = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        subdir = ast.Constant(value="subdir")
        binop = ast.BinOp(left=path_cwd, op=ast.FloorDiv(), right=subdir)

        result = transformer.visit_BinOp(binop)

        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Name)
        assert result.func.id == 'RegisterPathFunc'
        assert len(result.args) == 1
        assert result.args[0] == binop

    def test_visit_binop_with_non_path_operation(self):
        """Test visiting a BinOp node with non-path operation."""
        transformer = PathCollectorTransformer()

        # Create AST for: a + b
        binop = ast.BinOp(
            left=ast.Name(id='a', ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Name(id='b', ctx=ast.Load())
        )

        result = transformer.visit_BinOp(binop)

        # Should return the original operation unchanged
        assert result == binop

    def test_visit_binop_with_string_division(self):
        """Test visiting a BinOp node with string division (not path)."""
        transformer = PathCollectorTransformer()

        # Create AST for: "hello" / "world"
        binop = ast.BinOp(
            left=ast.Constant(value="hello"),
            op=ast.Div(),
            right=ast.Constant(value="world")
        )

        result = transformer.visit_BinOp(binop)

        # Should return the original operation unchanged
        assert result == binop

    def test_visit_attribute_with_path_method(self):
        """Test visiting an Attribute node with Path method."""
        transformer = PathCollectorTransformer()

        # Create AST for: Path.cwd().parent (using Path.cwd() instead of just 'path')
        path_cwd = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        attr = ast.Attribute(value=path_cwd, attr='parent', ctx=ast.Load())

        result = transformer.visit_Attribute(attr)

        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Name)
        assert result.func.id == 'RegisterPathFunc'
        assert len(result.args) == 1
        assert result.args[0] == attr

    def test_visit_attribute_with_path_with_suffix(self):
        """Test visiting an Attribute node with path.with_suffix method."""
        transformer = PathCollectorTransformer()

        # Create AST for: Path.cwd().with_suffix('.txt') (using Path.cwd() instead of just 'path')
        path_cwd = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        attr = ast.Attribute(value=path_cwd, attr='with_suffix', ctx=ast.Load())

        result = transformer.visit_Attribute(attr)

        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Name)
        assert result.func.id == 'RegisterPathFunc'
        assert len(result.args) == 1
        assert result.args[0] == attr

    def test_visit_attribute_with_non_path_method(self):
        """Test visiting an Attribute node with non-Path method."""
        transformer = PathCollectorTransformer()

        # Create AST for: obj.method
        obj_var = ast.Name(id='obj', ctx=ast.Load())
        attr = ast.Attribute(value=obj_var, attr='method', ctx=ast.Load())

        result = transformer.visit_Attribute(attr)

        # Should return the original attribute unchanged
        assert result == attr

    def test_visit_attribute_with_path_method_not_returning_path(self):
        """Test visiting an Attribute node with Path method that doesn't return Path."""
        transformer = PathCollectorTransformer()

        # Create AST for: Path.cwd().exists() (not in PATH_RETURNING_METHODS)
        path_cwd = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        attr = ast.Attribute(value=path_cwd, attr='exists', ctx=ast.Load())

        result = transformer.visit_Attribute(attr)

        # Should return the original attribute unchanged
        assert result == attr

    def test_visit_attribute_with_nested_path_operation(self):
        """Test visiting an Attribute node with nested path operation."""
        transformer = PathCollectorTransformer()

        # Create AST for: (Path.cwd() / "subdir").parent
        path_cwd = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        subdir = ast.Constant(value="subdir")
        binop = ast.BinOp(left=path_cwd, op=ast.Div(), right=subdir)
        attr = ast.Attribute(value=binop, attr='parent', ctx=ast.Load())

        result = transformer.visit_Attribute(attr)

        # The transformer should now wrap nested path operations
        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Name)
        assert result.func.id == 'RegisterPathFunc'
        assert len(result.args) == 1
        assert result.args[0] == attr

    def test_complex_path_expression_transformation(self):
        """Test transformation of complex path expressions."""
        transformer = PathCollectorTransformer()

        # Create AST for: (Path.cwd() / "data" / "files").parent.with_suffix('.txt')
        path_cwd = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )

        # First division: Path.cwd() / "data"
        first_div = ast.BinOp(
            left=path_cwd,
            op=ast.Div(),
            right=ast.Constant(value="data")
        )

        # Second division: (Path.cwd() / "data") / "files"
        second_div = ast.BinOp(
            left=first_div,
            op=ast.Div(),
            right=ast.Constant(value="files")
        )

        # Parent: ((Path.cwd() / "data") / "files").parent
        parent_attr = ast.Attribute(
            value=second_div,
            attr='parent',
            ctx=ast.Load()
        )

        # Final with_suffix: (((Path.cwd() / "data") / "files").parent).with_suffix('.txt')
        final_attr = ast.Attribute(
            value=parent_attr,
            attr='with_suffix',
            ctx=ast.Load()
        )

        # Transform the final expression
        result = transformer.visit_Attribute(final_attr)

        # The transformer should now wrap complex nested path operations
        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Name)
        assert result.func.id == 'RegisterPathFunc'
        assert len(result.args) == 1
        assert result.args[0] == final_attr

    def test_path_methods_coverage(self):
        """Test that all PATH_RETURNING_METHODS are covered."""
        transformer = PathCollectorTransformer()

        for method_name in transformer.PATH_RETURNING_METHODS:
            # Create AST for: Path.cwd().{method_name} (using Path.cwd() instead of just 'path')
            path_cwd = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='Path', ctx=ast.Load()),
                    attr='cwd',
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            )
            attr = ast.Attribute(value=path_cwd, attr=method_name, ctx=ast.Load())

            result = transformer.visit_Attribute(attr)

            assert isinstance(result, ast.Call)
            assert isinstance(result.func, ast.Name)
            assert result.func.id == 'RegisterPathFunc'
            assert len(result.args) == 1
            assert result.args[0] == attr

    def test_path_static_methods_coverage(self):
        """Test that all PATH_STATIC_METHODS are covered."""
        transformer = PathCollectorTransformer()

        for method_name in transformer.PATH_STATIC_METHODS:
            # Create AST for: Path.{method_name}()
            path_static_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='Path', ctx=ast.Load()),
                    attr=method_name,
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            )

            result = transformer._is_likely_path_object(path_static_call)
            assert result is True

    def test_is_likely_path_object_with_path_constructor(self):
        """Test _is_likely_path_object with Path constructor call."""
        transformer = PathCollectorTransformer()

        # Create AST for: Path("/test")
        path_call = ast.Call(
            func=ast.Name(id='Path', ctx=ast.Load()),
            args=[ast.Constant(value="/test")],
            keywords=[]
        )

        result = transformer._is_likely_path_object(path_call)
        assert result is True

    def test_is_likely_path_object_with_path_static_method(self):
        """Test _is_likely_path_object with Path static method call."""
        transformer = PathCollectorTransformer()

        # Create AST for: Path.cwd()
        path_cwd_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )

        result = transformer._is_likely_path_object(path_cwd_call)
        assert result is True

    def test_is_likely_path_object_with_path_home_method(self):
        """Test _is_likely_path_object with Path.home() call."""
        transformer = PathCollectorTransformer()

        # Create AST for: Path.home()
        path_home_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='home',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )

        result = transformer._is_likely_path_object(path_home_call)
        assert result is True

    def test_is_likely_path_object_with_attribute(self):
        """Test _is_likely_path_object with attribute access."""
        transformer = PathCollectorTransformer()

        # Create AST for: Path.cwd().parent (using Path.cwd() instead of just 'path')
        path_cwd = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        attr = ast.Attribute(value=path_cwd, attr='parent', ctx=ast.Load())

        result = transformer._is_likely_path_object(attr)
        assert result is True

    def test_is_likely_path_object_with_binop(self):
        """Test _is_likely_path_object with binary operation."""
        transformer = PathCollectorTransformer()

        # Create AST for: Path.cwd() / "subdir" (using Path.cwd() instead of just 'path')
        path_cwd = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        subdir = ast.Constant(value="subdir")
        binop = ast.BinOp(left=path_cwd, op=ast.Div(), right=subdir)

        result = transformer._is_likely_path_object(binop)
        assert result is True

    def test_is_likely_path_object_with_other_call(self):
        """Test _is_likely_path_object with non-Path call."""
        transformer = PathCollectorTransformer()

        # Create AST for: other_function()
        other_call = ast.Call(
            func=ast.Name(id='other_function', ctx=ast.Load()),
            args=[],
            keywords=[]
        )

        result = transformer._is_likely_path_object(other_call)
        assert result is False

    def test_is_likely_path_object_with_constant(self):
        """Test _is_likely_path_object with constant."""
        transformer = PathCollectorTransformer()

        # Create AST for: "string"
        const = ast.Constant(value="string")

        result = transformer._is_likely_path_object(const)
        assert result is False

    def test_is_likely_path_object_with_simple_variable(self):
        """Test _is_likely_path_object with simple variable name."""
        transformer = PathCollectorTransformer()

        # Create AST for: path (simple variable name)
        path_var = ast.Name(id='path', ctx=ast.Load())

        result = transformer._is_likely_path_object(path_var)
        assert result is False  # Can't determine if it's a Path without more context

    def test_is_path_division_operation_with_path_division(self):
        """Test _is_path_division_operation with path division."""
        transformer = PathCollectorTransformer()

        # Create AST for: Path.cwd() / "subdir" (using Path.cwd() instead of just 'path')
        path_cwd = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        subdir = ast.Constant(value="subdir")
        binop = ast.BinOp(left=path_cwd, op=ast.Div(), right=subdir)

        result = transformer._is_path_division_operation(binop)
        assert result is True

    def test_is_path_division_operation_with_string_division(self):
        """Test _is_path_division_operation with string division."""
        transformer = PathCollectorTransformer()

        # Create AST for: "hello" / "world"
        binop = ast.BinOp(
            left=ast.Constant(value="hello"),
            op=ast.Div(),
            right=ast.Constant(value="world")
        )

        result = transformer._is_path_division_operation(binop)
        assert result is True  # It's still a division operation, just not with Path objects

    def test_is_path_division_operation_with_addition(self):
        """Test _is_path_division_operation with addition (not division)."""
        transformer = PathCollectorTransformer()

        # Create AST for: a + b
        binop = ast.BinOp(
            left=ast.Name(id='a', ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Name(id='b', ctx=ast.Load())
        )

        result = transformer._is_path_division_operation(binop)
        assert result is False

    def test_is_path_division_operation_with_path_on_right(self):
        """Test _is_path_division_operation with Path on right side."""
        transformer = PathCollectorTransformer()

        # Create AST for: "prefix" / Path("suffix")
        prefix = ast.Constant(value="prefix")
        path_call = ast.Call(
            func=ast.Name(id='Path', ctx=ast.Load()),
            args=[ast.Constant(value="suffix")],
            keywords=[]
        )
        binop = ast.BinOp(left=prefix, op=ast.Div(), right=path_call)

        result = transformer._is_path_division_operation(binop)
        assert result is True

    def test_helper_methods_coverage(self):
        """Test that all helper methods work correctly."""
        transformer = PathCollectorTransformer()

        # Test _is_path_constructor
        path_call = ast.Call(
            func=ast.Name(id='Path', ctx=ast.Load()),
            args=[ast.Constant(value="/test")],
            keywords=[]
        )
        assert transformer._is_path_constructor(path_call) is True

        # Test _is_path_static_method
        path_cwd_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        assert transformer._is_path_static_method(path_cwd_call) is True

        # Test _is_path_method_call
        path_cwd = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr='cwd',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        attr = ast.Attribute(value=path_cwd, attr='parent', ctx=ast.Load())
        assert transformer._is_path_method_call(attr) is True

        # Test _is_path_division_operation
        binop = ast.BinOp(
            left=path_cwd,
            op=ast.Div(),
            right=ast.Constant(value="subdir")
        )
        assert transformer._is_path_division_operation(binop) is True

    def test_wrap_in_RegisterPathFunc(self):
        """Test _wrap_in_RegisterPathFunc helper method."""
        transformer = PathCollectorTransformer()

        # Create a simple AST node
        path_call = ast.Call(
            func=ast.Name(id='Path', ctx=ast.Load()),
            args=[ast.Constant(value="/test")],
            keywords=[]
        )

        # Wrap it in RegisterPathFunc
        result = transformer._wrap_in_RegisterPathFunc(path_call)

        # Check the result
        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Name)
        assert result.func.id == 'RegisterPathFunc'
        assert len(result.args) == 1
        assert result.args[0] == path_call


@dataclass
class DataClassUnderTest:
    """Test dataclass for testing as_table functionality."""
    name: str
    age: int


class TestMemvidMagics:
    """Test MemvidMagics class."""

    @patch('ipykernel_memvid_extension.chime')
    def test_init(self, mock_chime, mock_shell):
        """Test MemvidMagics initialization."""
        mock_chime.themes.return_value = ['theme1', 'theme2']

        # Create a proper mock that satisfies traitlets requirements
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)

            assert magics.shell == mock_shell
            mock_shell.events.register.assert_called_once_with('post_run_cell', magics.play_sound)

    @patch('ipykernel_memvid_extension.chime')
    def test_post_run_cell_success(self, mock_chime, mock_shell):
        """Test post_run_cell with successful execution."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            result = Mock()
            result.success = True

            magics.play_sound(result)

            mock_chime.notify.assert_called_once_with("success", sync=True, raise_error=False)

    @patch('ipykernel_memvid_extension.chime')
    def test_post_run_cell_failure(self, mock_chime, mock_shell):
        """Test post_run_cell with failed execution."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            result = Mock()
            result.success = False

            magics.play_sound(result)

            mock_chime.notify.assert_called_once_with("error", sync=True, raise_error=False)

    @patch('ipykernel_memvid_extension.display')
    def test_as_bullet_list_no_variable(self, mock_display, mock_shell):
        """Test as_bullet_list with no variable name."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)

            # Test with empty string - should trigger argument parsing error
            with pytest.raises(Exception):  # IPython will raise UsageError
                magics.as_bullet_list("")

    @patch('ipykernel_memvid_extension.display')
    def test_as_bullet_list_variable_not_found(self, mock_display, mock_shell):
        """Test as_bullet_list with variable not in namespace."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)

            magics.as_bullet_list("nonexistent")

            mock_display.assert_called_once()

    @patch('ipykernel_memvid_extension.display')
    def test_as_bullet_list_not_iterable(self, mock_display, mock_shell):
        """Test as_bullet_list with non-iterable variable."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.user_ns['test_var'] = 42

            magics.as_bullet_list("test_var")

            mock_display.assert_called_once()

    @patch('ipykernel_memvid_extension.display')
    def test_as_bullet_list_success(self, mock_display, mock_shell):
        """Test as_bullet_list with valid iterable."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.user_ns['test_list'] = ['item1', 'item2', 'item3']

            magics.as_bullet_list("test_list")

            mock_display.assert_called_once()
            call_args = mock_display.call_args[0][0]
            html_str = str(call_args.data)
            assert "item1" in html_str
            assert "item2" in html_str
            assert "item3" in html_str

    @patch('ipykernel_memvid_extension.display')
    def test_as_table_no_variable(self, mock_display, mock_shell):
        """Test as_table with no variable name."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)

            # Test with empty string - should trigger argument parsing error
            with pytest.raises(Exception):  # IPython will raise UsageError
                magics.as_table("")

    @patch('ipykernel_memvid_extension.display')
    def test_as_table_variable_not_found(self, mock_display, mock_shell):
        """Test as_table with variable not in namespace."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)

            magics.as_table("nonexistent")

            mock_display.assert_called_once()

    @patch('ipykernel_memvid_extension.display')
    def test_as_table_not_iterable(self, mock_display, mock_shell):
        """Test as_table with non-iterable variable."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.user_ns['test_var'] = 42

            magics.as_table("test_var")

            mock_display.assert_called_once()

    @patch('ipykernel_memvid_extension.display')
    def test_as_table_success(self, mock_display, mock_shell):
        """Test as_table with valid list of dictionaries."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.user_ns['test_data'] = [
                {'name': 'John', 'age': 30},
                {'name': 'Jane', 'age': 25}
            ]

            magics.as_table("test_data")

            mock_display.assert_called_once()
            call_args = mock_display.call_args[0][0]
            html_str = str(call_args.data)
            assert "John" in html_str
            assert "Jane" in html_str
            assert "30" in html_str
            assert "25" in html_str

    @patch('ipykernel_memvid_extension.display')
    def test_as_table_single_dict(self, mock_display, mock_shell):
        """Test as_table with single dictionary."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.user_ns['test_dict'] = {'name': 'John', 'age': 30}

            magics.as_table("test_dict")

            mock_display.assert_called_once()
            call_args = mock_display.call_args[0][0]
            html_str = str(call_args.data)
            assert "John" in html_str
            assert "30" in html_str

    @patch('ipykernel_memvid_extension.display')
    def test_as_table_dataclass(self, mock_display, mock_shell):
        """Test as_table with dataclass."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.user_ns['test_dataclass'] = [DataClassUnderTest('John', 30)]

            magics.as_table("test_dataclass")

            mock_display.assert_called_once()
            call_args = mock_display.call_args[0][0]
            html_str = str(call_args.data)
            assert "John" in html_str
            assert "30" in html_str

    @patch('ipykernel_memvid_extension.delete_path')
    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    def test_cleanup(self, mock_display_warning, mock_delete_path, mock_shell):
        """Test cleanup magic command."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)

            # Add some Path objects to user namespace
            test_path1 = Path("/test/path1")
            test_path2 = Path("/test/path2")
            magics.shell.user_ns['path1'] = test_path1
            magics.shell.user_ns['path2'] = test_path2

            # Add some paths to RegisterPathFunc
            RegisterPathFunc.paths.clear()
            RegisterPathFunc.paths.append(test_path1)

            magics.cleanup("")

            # Check that paths were deleted from namespace
            assert 'path1' not in magics.shell.user_ns
            assert 'path2' not in magics.shell.user_ns

            # Check that delete_path was called
            assert mock_delete_path.call_count == 2

            # Check that warning was displayed
            mock_display_warning.assert_called_once()

    @patch('ipykernel_memvid_extension.clear_output')
    def test_pip_install_no_command(self, mock_clear_output, mock_shell):
        """Test pip_install with no command."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)

            magics.pip_install("")

            mock_clear_output.assert_not_called()

    @patch('ipykernel_memvid_extension.clear_output')
    @patch('ipykernel_memvid_extension.capture_output')
    def test_pip_install_success(self, mock_capture_output, mock_clear_output, mock_shell):
        """Test pip_install with successful installation."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.run_line_magic = Mock()

            # Mock capture_output context manager
            mock_context = Mock()
            mock_context.__enter__ = Mock(return_value=Mock())
            mock_context.__exit__ = Mock(return_value=None)
            mock_capture_output.return_value = mock_context

            # Mock captured output
            mock_context.__enter__().stdout = "package==1.0.0\nother==2.0.0\n"

            magics.pip_install("package")

            magics.shell.run_line_magic.assert_called()
            mock_clear_output.assert_called_once()

    @patch('ipykernel_memvid_extension.clear_output')
    def test_pip_install_error(self, mock_clear_output, mock_shell):
        """Test pip_install with error."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.run_line_magic = Mock(side_effect=Exception("Installation failed"))

            magics.pip_install("package")

            mock_clear_output.assert_not_called()

    @patch('ipykernel_memvid_extension.display')
    def test_restart_kernel(self, mock_display, mock_shell):
        """Test restart_kernel magic command."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)

            magics.restart_kernel("-f")

            mock_display.assert_called_once()

    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    def test_list_sound_themes(self, mock_display_notification, mock_shell):
        """Test list_sound_themes magic command."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            # Mock chime.themes() to return test themes
            with patch('ipykernel_memvid_extension.chime') as mock_chime:
                mock_chime.themes.return_value = ['theme1', 'theme2', 'theme3']

                magics.list_sound_themes("")

                mock_display_notification.assert_called_once()
                call_args = mock_display_notification.call_args[0][0]
                assert "theme1" in call_args
                assert "theme2" in call_args
                assert "theme3" in call_args

    @patch('ipykernel_memvid_extension.chime')
    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    def test_set_sound_theme_valid(self, mock_display_notification, mock_chime, mock_shell):
        """Test set_sound_theme with valid theme."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            mock_chime.themes.return_value = ['theme1', 'theme2']

            magics.set_sound_theme("theme1")

            mock_chime.theme.assert_called_once_with("theme1")
            mock_display_notification.assert_called_once()

    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    def test_set_sound_theme_invalid(self, mock_display_alert, mock_shell):
        """Test set_sound_theme with invalid theme."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            with patch('ipykernel_memvid_extension.chime') as mock_chime:
                mock_chime.themes.return_value = ['theme1', 'theme2']

                magics.set_sound_theme("invalid_theme")

                mock_display_alert.assert_called_once()

    @patch('ipykernel_memvid_extension.Path.exists')
    @patch('ipykernel_memvid_extension.Path.read_text')
    @patch('ipykernel_memvid_extension.Path.write_text')
    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    @patch('ipykernel_memvid_extension.type_to_confirm')
    def test_dump_output_exists_no_force_no_confirm(
        self, mock_type_to_confirm, mock_display_alert, mock_write_text, mock_read_text, mock_exists, mock_shell
    ):
        """Test dump when output exists, no force, and user doesn't confirm."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            # Mock that source exists but output also exists
            mock_exists.side_effect = [True, True]  # source.ipynb exists, output.py exists
            mock_type_to_confirm.return_value = False

            magics.dump('test_notebook')

            mock_display_alert.assert_called_once()
            call_args = mock_display_alert.call_args[0][0]
            assert "already exists" in call_args

    @patch('ipykernel_memvid_extension.Path.exists')
    @patch('ipykernel_memvid_extension.Path.read_text')
    @patch('ipykernel_memvid_extension.Path.write_text')
    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    @patch('ipykernel_memvid_extension.type_to_confirm')
    def test_dump_success_no_range(
        self, mock_type_to_confirm, mock_display_notification, mock_write_text, mock_read_text, mock_exists, mock_shell
    ):
        """Test successful dump without range specification."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.user_ns['__vsc_ipynb_file__'] = 'test_notebook.ipynb'
            mock_exists.return_value = True
            mock_read_text.return_value = (
                '{"cells": [{"cell_type": "code", "source": ["print(\'hello\')"]}]}'
            )
            mock_type_to_confirm.return_value = True

            magics.dump('test_notebook')

            mock_write_text.assert_called_once()
            mock_display_notification.assert_called_once()

    @patch('ipykernel_memvid_extension.Path.exists')
    @patch('ipykernel_memvid_extension.Path.read_text')
    @patch('ipykernel_memvid_extension.Path.write_text')
    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    @patch('ipykernel_memvid_extension.type_to_confirm')
    def test_dump_with_range_specification(
        self, mock_type_to_confirm, mock_display_notification, mock_write_text, mock_read_text, mock_exists, mock_shell
    ):
        """Test dump with range specification."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.user_ns['__vsc_ipynb_file__'] = 'test_notebook.ipynb'
            mock_exists.return_value = True
            mock_read_text.return_value = (
                '{"cells": [{"cell_type": "code", "source": ["print(\'hello\')"]}, '
                '{"cell_type": "code", "source": ["print(\'world\')"]}]}'
            )
            mock_type_to_confirm.return_value = True

            magics.dump('test_notebook -r 1:2')

            mock_write_text.assert_called_once()
            mock_display_notification.assert_called_once()

    @patch('ipykernel_memvid_extension.Path.exists')
    @patch('ipykernel_memvid_extension.Path.read_text')
    @patch('ipykernel_memvid_extension.Path.write_text')
    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    @patch('ipykernel_memvid_extension.type_to_confirm')
    def test_dump_invalid_range_specification(
        self, mock_type_to_confirm, mock_display_alert, mock_write_text, mock_read_text, mock_exists, mock_shell
    ):
        """Test dump with invalid range specification."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.user_ns['__vsc_ipynb_file__'] = 'test_notebook.ipynb'
            mock_exists.return_value = True
            mock_read_text.return_value = (
                '{"cells": [{"cell_type": "code", "source": ["print(\'hello\')"]}]}'
            )
            mock_type_to_confirm.return_value = True

            magics.dump('test_notebook -r 999')

            mock_display_alert.assert_called_once()
            call_args = mock_display_alert.call_args[0][0]
            assert "Invalid range specification" in call_args

    @patch('ipykernel_memvid_extension.Path.exists')
    @patch('ipykernel_memvid_extension.Path.read_text')
    @patch('ipykernel_memvid_extension.Path.write_text')
    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    @patch('ipykernel_memvid_extension.type_to_confirm')
    def test_dump_with_markdown_cells(
        self, mock_type_to_confirm, mock_display_notification, mock_write_text, mock_read_text, mock_exists, mock_shell
    ):
        """Test dump with markdown cells."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.user_ns['__vsc_ipynb_file__'] = 'test_notebook.ipynb'
            mock_exists.return_value = True
            mock_read_text.return_value = (
                '{"cells": [{"cell_type": "markdown", "source": ["# Title", "Some text"]}]}'
            )
            mock_type_to_confirm.return_value = True

            magics.dump('test_notebook')

            mock_write_text.assert_called_once()
            # Check that the output contains comment lines
            written_content = mock_write_text.call_args[0][0]
            assert "# " in written_content

    @patch('ipykernel_memvid_extension.Path.exists')
    @patch('ipykernel_memvid_extension.Path.read_text')
    @patch('ipykernel_memvid_extension.Path.write_text')
    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    @patch('ipykernel_memvid_extension.type_to_confirm')
    def test_dump_filters_magic_commands(
        self, mock_type_to_confirm, mock_display_notification, mock_write_text, mock_read_text, mock_exists, mock_shell
    ):
        """Test that dump filters out magic commands."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.user_ns['__vsc_ipynb_file__'] = 'test_notebook.ipynb'
            mock_exists.return_value = True
            mock_read_text.return_value = (
                '{"cells": [{"cell_type": "code", "source": '
                '["%magic_command", "print(\'hello\')", "%another_magic"]}]}'
            )
            mock_type_to_confirm.return_value = True

            magics.dump('test_notebook')

            mock_write_text.assert_called_once()
            written_content = mock_write_text.call_args[0][0]
            # Should not contain magic commands
            assert "%magic_command" not in written_content
            assert "%another_magic" not in written_content
            # Should contain regular code
            assert "print('hello')" in written_content

    # Range parsing tests
    @pytest.mark.parametrize(("range_spec", "expected", "total_cells"), (
        ("", [0, 1, 2, 3, 4], 5),
        ("   ", [0, 1, 2, 3, 4], 5),  # whitespace only
        ("1", [0], 5),
        ("1:3", [0, 1, 2], 5),
        (":3", [0, 1, 2], 5),
        ("3:", [2, 3, 4], 5),
        ("1,3,5", [0, 2, 4], 5),
        ("1,1,2,2,3", [0, 1, 2], 5),  # duplicate indices
        ("1:3,2:4", [0, 1, 2, 3], 5),  # overlapping ranges
        ("-1", [4], 5),  # negative indexing
        ("-2", [3], 5),
        ("-5", [0], 5),
        ("-3:", [2, 3, 4], 5),  # negative start
        (":-1", [0, 1, 2, 3], 5),  # negative end
        ("-3:-1", [2, 3], 5),  # negative start and end
        ("-2:5", [3, 4], 5),  # negative start, positive end
        ("1:-1", [0, 1, 2, 3], 5),  # positive start, negative end
    ), ids=list(range(1, 18)))
    def test_parse_cell_range_valid_specs(self, range_spec, expected, total_cells, mock_shell):
        """Test parsing valid range specifications."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            result = magics._parse_cell_range(range_spec, total_cells)
            assert result == expected

    @pytest.mark.parametrize("range_spec,total_cells,expected_error", [
        ("0", 5, "Cell index must be between 1 and 5"),
        ("6", 5, "Cell index must be between 1 and 5"),
        ("-6", 5, "Cell index must be between 1 and 5"),  # too negative
        ("5:3", 5, "Start index.*must be <= end index"),
        ("1:2:3", 5, "Invalid range format"),
        ("1::3", 5, "Invalid range format"),
        ("abc", 5, "Invalid range format"),
        ("1-3", 5, "Invalid range format"),  # old format no longer supported
    ])
    def test_parse_cell_range_invalid_specs(self, range_spec, total_cells, expected_error, mock_shell):
        """Test parsing invalid range specifications."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            with pytest.raises(ValueError, match=expected_error):
                magics._parse_cell_range(range_spec, total_cells)

    # Cell dumping tests
    def test_dump_code_cell(self, mock_shell):
        """Test dumping a code cell."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            cell = {
                "cell_type": "code",
                "source": ["print('hello')\n", "%magic_command\n", "x = 1\n"],
                "outputs": []
            }
            result = magics._dump_code_cell(cell)
            expected = ["print('hello')\n", "x = 1\n"]
            assert result[:-1] == expected

    def test_dump_code_cell_empty(self, mock_shell):
        """Test dumping an empty code cell."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            cell = {
                "cell_type": "code",
                "source": [],
                "outputs": []
            }
            result = magics._dump_code_cell(cell)
            assert result == []

    def test_dump_code_cell_only_magic(self, mock_shell):
        """Test dumping a code cell with only magic commands."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            cell = {
                "cell_type": "code",
                "source": ["%magic1\n", "%magic2\n"],
                "outputs": []
            }
            result = magics._dump_code_cell(cell)
            assert result == []

    def test_dump_markdown_cell(self, mock_shell):
        """Test dumping a markdown cell."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            cell = {
                "cell_type": "markdown",
                "source": ["# Title\n", "This is a **bold** text\n"]
            }
            result = magics._dump_markdown_cell(cell)
            # Should contain comment lines starting with #
            assert all(line.startswith("# ") for line in result[:-1])
            assert len(result) > 0

    def test_dump_markdown_cell_empty(self, mock_shell):
        """Test dumping an empty markdown cell."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            cell = {
                "cell_type": "markdown",
                "source": []
            }
            result = magics._dump_markdown_cell(cell)
            assert result == []

    # Modern argument parsing tests
    @pytest.mark.parametrize("magic_name,args,expected_call", [
        ("dump", "test_notebook", "dump"),
        ("cleanup", "", "cleanup"),
        ("as_bullet_list", "test_var", "as_bullet_list"),
        ("as_table", "test_var", "as_table"),
        ("pip_install", "pandas numpy", "pip_install"),
        ("restart_kernel", "-f", "restart_kernel"),
        ("set_sound_theme", "zelda", "set_sound_theme"),
    ])
    def test_magic_commands_use_modern_argument_parsing(self, magic_name, args, expected_call, mock_shell):
        """Test that all magic commands use modern argument parsing."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)

            with patch('ipykernel_memvid_extension.parse_argstring') as mock_parse:
                # Create specific mock args for each command
                mock_args = Mock()
                match magic_name:
                    case "dump":
                        mock_args.notebook_name = 'test_notebook'
                        mock_args.force = False
                        mock_args.range_spec = None
                        mock_args.cell_outputs = False
                    case "cleanup":
                        mock_args.force = False
                    case "as_bullet_list" | "as_table":
                        mock_args.variable_name = ['test_var']
                    case "pip_install":
                        mock_args.packages = ['pandas', 'numpy']
                    case "restart_kernel":
                        mock_args.force = True
                    case "set_sound_theme":
                        mock_args.theme_name = ['zelda']

                mock_parse.return_value = mock_args

                # Mock necessary dependencies for each command
                with (
                    patch('ipykernel_memvid_extension.DisplayContainer.display'),
                    patch('ipykernel_memvid_extension.Path.exists', return_value=True),
                    patch('ipykernel_memvid_extension.Path.read_text', return_value='{"cells": []}'),
                    patch('ipykernel_memvid_extension.Path.write_text'),
                    patch('ipykernel_memvid_extension.type_to_confirm', return_value=True),
                    patch('ipykernel_memvid_extension.chime') as mock_chime,
                ):
                    mock_chime.themes.return_value = ['zelda']

                    # Call the magic command
                    getattr(magics, magic_name)(args)

                    # Verify parse_argstring was called
                    mock_parse.assert_called_once()


class TestLoadIPythonExtension:
    """Test load_ipython_extension function."""

    @patch('ipykernel_memvid_extension.clear_output')
    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    @patch('ipykernel_memvid_extension.chime')
    def test_load_extension_success(self, mock_chime, mock_display_notification, mock_clear_output, mock_shell):
        """Test successful loading of IPython extension."""
        mock_chime.theme = Mock()

        load_ipython_extension(mock_shell)

        # Check that AST transformer was added
        assert len(mock_shell.ast_transformers) == 1
        assert isinstance(mock_shell.ast_transformers[0], PathCollectorTransformer)

        # Check that RegisterPathFunc was added to namespace
        assert 'RegisterPathFunc' in mock_shell.user_ns

        # Check that magics were registered
        mock_shell.register_magics.assert_called_once()

        # Check that notification was displayed
        mock_display_notification.assert_called_once()

    @patch('ipykernel_memvid_extension.clear_output')
    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    @patch('ipykernel_memvid_extension.chime')
    def test_load_extension_error(self, mock_chime, mock_display_alert, mock_clear_output, mock_shell):
        """Test loading extension with error."""
        mock_shell.register_magics.side_effect = Exception("Registration failed")

        with pytest.raises(Exception):
            load_ipython_extension(mock_shell)

        mock_display_alert.assert_called_once()
