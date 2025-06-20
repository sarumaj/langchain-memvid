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
from enum import Enum
from conftest import custom_parametrize

from ipykernel_memvid_extension import IPYTHON_INSTALLED

# Skip all tests if IPython is not installed
pytestmark = pytest.mark.skipif(
    not IPYTHON_INSTALLED,
    reason="IPython is not installed"
)

# Import all required modules - these will be None if IPython is not installed
if IPYTHON_INSTALLED:
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
else:
    # Create dummy objects used for test parameterization
    DisplayContainer = Enum('DisplayContainer', ['ALERT', 'NOTIFICATION', 'WARNING'])


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


@pytest.fixture
def mock_magics(mock_shell):
    """Create a MemvidMagics instance with mocked dependencies."""
    with patch('ipykernel_memvid_extension.Magics.__init__'):
        return MemvidMagics(mock_shell)


@pytest.fixture
def ast_transformer():
    """Create a PathCollectorTransformer instance."""
    return PathCollectorTransformer()


@pytest.fixture
def path_cwd_ast():
    """Create AST for Path.cwd() call."""
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='Path', ctx=ast.Load()),
            attr='cwd',
            ctx=ast.Load()
        ),
        args=[],
        keywords=[]
    )


@pytest.fixture
def path_constructor_ast():
    """Create AST for Path constructor call."""
    return ast.Call(
        func=ast.Name(id='Path', ctx=ast.Load()),
        args=[ast.Constant(value="/test/path")],
        keywords=[]
    )


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

    @custom_parametrize(("path_type", "force"), [
        ("file", False),
        ("directory", True),
        ("nonexistent", False),
        ("invalid_type", False),
    ])
    def test_delete_path_variants(self, path_type, force, temp_path, tmp_path):
        """Test delete_path with different path types."""
        if path_type == "file":
            path = temp_path
            path.write_text("test content")
            assert path.exists()
        elif path_type == "directory":
            path = tmp_path / "test_dir"
            path.mkdir()
            (path / "test_file.txt").write_text("test")
            assert path.exists()
        elif path_type == "nonexistent":
            path = Path("/nonexistent/path")
        else:  # invalid_type
            path = "not a path"

        result = delete_path(path, force=force)

        if path_type in ["file", "directory"]:
            assert result is True
            assert not path.exists()
        else:
            assert result is False

    @custom_parametrize(("container_type", "color"), [
        (DisplayContainer.ALERT, "red"),
        (DisplayContainer.NOTIFICATION, "green"),
        (DisplayContainer.WARNING, "orange"),
    ])
    @patch('ipykernel_memvid_extension.display')
    def test_display_container_methods(self, mock_display, container_type, color):
        """Test display methods for different container types."""
        message = f"Test {container_type.name} message"
        container_type.display(message)

        mock_display.assert_called_once()
        call_args = mock_display.call_args[0][0]
        html_str = str(call_args.data)
        assert message in html_str
        assert color in html_str

    @patch('ipykernel_memvid_extension.HTML')
    def test_make_html_container(self, mock_HTML):
        """Test make_html_container function."""
        DisplayContainer.make_html_container("Test message", "blue")

        mock_HTML.assert_called_once()
        call_args = mock_HTML.call_args[0][0]
        html_str = str(call_args)
        assert "Test message" in html_str
        assert "blue" in html_str

    @custom_parametrize(("package_spec", "expected"), [
        ("package==1.0.0", "package"),
        ("package>=1.0.0", "package"),
        ("package<=1.0.0", "package"),
        ("package", "package"),
        ("package-name", "package-name"),
    ])
    def test_strip_version(self, package_spec, expected):
        """Test strip_version function."""
        assert strip_version(package_spec) == expected

    @custom_parametrize(("input_value", "confirmation", "expected"), [
        ("test_dir", "test_dir", True),
        ("wrong_input", "test_dir", True),  # Will retry
        ("", "test_dir", False),
    ])
    @patch('builtins.input')
    def test_type_to_confirm_variants(self, mock_input, input_value, confirmation, expected):
        """Test type_to_confirm with different input scenarios."""
        if input_value == "wrong_input":
            mock_input.side_effect = ["wrong_input", confirmation]
        else:
            mock_input.return_value = input_value

        result = type_to_confirm("Are you sure?", confirmation)

        assert result is expected

    @patch('builtins.input')
    def test_type_to_confirm_keyboard_interrupt(self, mock_input):
        """Test type_to_confirm with KeyboardInterrupt."""
        mock_input.side_effect = KeyboardInterrupt()

        result = type_to_confirm("Are you sure?", "test_dir")

        assert result is False


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

    @custom_parametrize(("ast_node", "should_wrap"), [
        ("path_constructor_ast", True),
        ("path_cwd_ast", False),  # Static method calls are not wrapped
    ])
    def test_visit_call_variants(self, ast_transformer, ast_node, should_wrap, request):
        """Test visiting Call nodes with different types."""
        node = request.getfixturevalue(ast_node)

        # For non-Path calls, create a different call
        if ast_node == "path_constructor_ast" and not should_wrap:
            node = ast.Call(
                func=ast.Name(id='other_function', ctx=ast.Load()),
                args=[],
                keywords=[]
            )

        result = ast_transformer.visit_Call(node)

        if should_wrap:
            assert isinstance(result, ast.Call)
            assert isinstance(result.func, ast.Name)
            assert result.func.id == 'RegisterPathFunc'
            assert len(result.args) == 1
            assert result.args[0] == node
        else:
            assert result == node

    @custom_parametrize(("operation", "left_type", "right_type", "should_wrap"), [
        (ast.Div(), "path_cwd_ast", "constant", True),
        (ast.FloorDiv(), "path_cwd_ast", "constant", True),
        (ast.Add(), "name", "name", False),
        (ast.Div(), "constant", "constant", False),
    ])
    def test_visit_binop_variants(self, ast_transformer, operation, left_type, right_type, should_wrap, request):
        """Test visiting BinOp nodes with different operations."""
        # Get left operand
        if left_type == "path_cwd_ast":
            left = request.getfixturevalue("path_cwd_ast")
        elif left_type == "name":
            left = ast.Name(id='a', ctx=ast.Load())
        else:  # constant
            left = ast.Constant(value="hello")

        # Get right operand
        if right_type == "constant":
            right = ast.Constant(value="subdir")
        elif right_type == "name":
            right = ast.Name(id='b', ctx=ast.Load())
        else:  # path
            right = request.getfixturevalue("path_constructor_ast")

        binop = ast.BinOp(left=left, op=operation, right=right)
        result = ast_transformer.visit_BinOp(binop)

        if should_wrap:
            assert isinstance(result, ast.Call)
            assert isinstance(result.func, ast.Name)
            assert result.func.id == 'RegisterPathFunc'
            assert len(result.args) == 1
            assert result.args[0] == binop
        else:
            assert result == binop

    @custom_parametrize(("method_name", "should_wrap"), [
        ("parent", True),
        ("with_suffix", True),
        ("exists", False),  # Not in PATH_RETURNING_METHODS
    ])
    def test_visit_attribute_variants(self, ast_transformer, method_name, should_wrap, path_cwd_ast):
        """Test visiting Attribute nodes with different methods."""
        attr = ast.Attribute(value=path_cwd_ast, attr=method_name, ctx=ast.Load())
        result = ast_transformer.visit_Attribute(attr)

        if should_wrap:
            assert isinstance(result, ast.Call)
            assert isinstance(result.func, ast.Name)
            assert result.func.id == 'RegisterPathFunc'
            assert len(result.args) == 1
            assert result.args[0] == attr
        else:
            assert result == attr

    def test_complex_path_expression_transformation(self, ast_transformer, path_cwd_ast):
        """Test transformation of complex path expressions."""
        # Create AST for: (Path.cwd() / "data" / "files").parent.with_suffix('.txt')
        first_div = ast.BinOp(
            left=path_cwd_ast,
            op=ast.Div(),
            right=ast.Constant(value="data")
        )
        second_div = ast.BinOp(
            left=first_div,
            op=ast.Div(),
            right=ast.Constant(value="files")
        )
        parent_attr = ast.Attribute(value=second_div, attr='parent', ctx=ast.Load())
        final_attr = ast.Attribute(value=parent_attr, attr='with_suffix', ctx=ast.Load())

        result = ast_transformer.visit_Attribute(final_attr)

        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Name)
        assert result.func.id == 'RegisterPathFunc'
        assert len(result.args) == 1
        assert result.args[0] == final_attr

    @custom_parametrize("method_name", [
        "parent", "name", "suffix", "stem", "with_name", "with_stem", "with_suffix",
        "absolute", "resolve", "expanduser", "glob", "rglob", "iterdir", "mkdir",
        "touch", "unlink", "rmdir", "rename", "replace", "symlink_to", "hardlink_to",
        "samefile", "owner", "group", "stat", "lstat", "chmod", "lchmod", "unlink",
        "link_to", "readlink", "touch", "mkdir", "rmdir", "rename", "replace",
        "symlink_to", "hardlink_to", "samefile", "owner", "group", "stat", "lstat",
        "chmod", "lchmod", "unlink", "link_to", "readlink"
    ])
    def test_path_methods_coverage(self, ast_transformer, method_name, path_cwd_ast):
        """Test that all PATH_RETURNING_METHODS are covered."""
        if method_name in ast_transformer.PATH_RETURNING_METHODS:
            attr = ast.Attribute(value=path_cwd_ast, attr=method_name, ctx=ast.Load())
            result = ast_transformer.visit_Attribute(attr)

            assert isinstance(result, ast.Call)
            assert isinstance(result.func, ast.Name)
            assert result.func.id == 'RegisterPathFunc'
            assert len(result.args) == 1
            assert result.args[0] == attr

    @custom_parametrize("method_name", ["cwd", "home"])
    def test_path_static_methods_coverage(self, ast_transformer, method_name):
        """Test that all PATH_STATIC_METHODS are covered."""
        path_static_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='Path', ctx=ast.Load()),
                attr=method_name,
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )

        result = ast_transformer._is_likely_path_object(path_static_call)
        assert result is True

    @custom_parametrize("ast_type,should_be_path", [
        ("path_constructor_ast", True),
        ("path_cwd_ast", True),
        ("constant", False),
        ("name", False),
    ])
    def test_is_likely_path_object_variants(self, ast_transformer, ast_type, should_be_path, request):
        """Test _is_likely_path_object with different AST types."""
        if ast_type == "constant":
            node = ast.Constant(value="string")
        elif ast_type == "name":
            node = ast.Name(id='path', ctx=ast.Load())
        else:
            node = request.getfixturevalue(ast_type)

        result = ast_transformer._is_likely_path_object(node)
        assert result is should_be_path

    def test_wrap_in_RegisterPathFunc(self, ast_transformer, path_constructor_ast):
        """Test _wrap_in_RegisterPathFunc helper method."""
        result = ast_transformer._wrap_in_RegisterPathFunc(path_constructor_ast)

        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Name)
        assert result.func.id == 'RegisterPathFunc'
        assert len(result.args) == 1
        assert result.args[0] == path_constructor_ast


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

        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)

            assert magics.shell == mock_shell
            mock_shell.events.register.assert_called_once_with('post_run_cell', magics.play_sound)

    @custom_parametrize("success", [True, False])
    @patch('ipykernel_memvid_extension.chime')
    def test_post_run_cell(self, mock_chime, success, mock_shell):
        """Test post_run_cell with different execution results."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            result = Mock()
            result.success = success

            magics.play_sound(result)

            expected_sound = "success" if success else "error"
            mock_chime.notify.assert_called_once_with(expected_sound, sync=True, raise_error=False)

    @custom_parametrize(("command_type", "args", "expected_behavior"), [
        ("as_bullet_list", "nonexistent", "display_error"),
        ("as_bullet_list", "test_var", "display_error"),  # non-iterable
        ("as_bullet_list", "test_list", "display_success"),
        ("as_table", "nonexistent", "display_error"),
        ("as_table", "test_var", "display_error"),  # non-iterable
        ("as_table", "test_data", "display_success"),
        ("as_table", "test_dict", "display_success"),
        ("as_table", "test_dataclass", "display_success"),
    ])
    @patch('ipykernel_memvid_extension.display')
    def test_display_commands(self, mock_display, command_type, args, expected_behavior, mock_shell):
        """Test display commands with different scenarios."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)

            # Set up test data in namespace
            if args == "test_list":
                magics.shell.user_ns['test_list'] = ['item1', 'item2', 'item3']
            elif args == "test_data":
                magics.shell.user_ns['test_data'] = [
                    {'name': 'John', 'age': 30},
                    {'name': 'Jane', 'age': 25}
                ]
            elif args == "test_dict":
                magics.shell.user_ns['test_dict'] = {'name': 'John', 'age': 30}
            elif args == "test_dataclass":
                magics.shell.user_ns['test_dataclass'] = [DataClassUnderTest('John', 30)]
            elif args == "test_var":
                magics.shell.user_ns['test_var'] = 42

            # Test empty string (should raise exception)
            if args == "":
                with pytest.raises(Exception):
                    getattr(magics, command_type)("")
                return

            getattr(magics, command_type)(args)

            mock_display.assert_called_once()
            if expected_behavior == "display_success":
                call_args = mock_display.call_args[0][0]
                html_str = str(call_args.data)
                # Check for expected content based on command type
                if command_type == "as_bullet_list":
                    assert "item1" in html_str
                elif command_type == "as_table":
                    assert "John" in html_str

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

    @custom_parametrize(("command", "expected_clear_called"), [
        ("", False),
        ("package", True),
    ])
    @patch('ipykernel_memvid_extension.clear_output')
    def test_pip_install_variants(self, mock_clear_output, command, expected_clear_called, mock_shell):
        """Test pip_install with different commands."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            magics.shell.run_line_magic = Mock()

            if command:
                # Mock capture_output context manager
                with patch('ipykernel_memvid_extension.capture_output') as mock_capture_output:
                    mock_context = Mock()
                    mock_context.__enter__ = Mock(return_value=Mock())
                    mock_context.__exit__ = Mock(return_value=None)
                    mock_capture_output.return_value = mock_context
                    mock_context.__enter__().stdout = "package==1.0.0\nother==2.0.0\n"

                    magics.pip_install(command)

                    magics.shell.run_line_magic.assert_called()
            else:
                magics.pip_install(command)

            if expected_clear_called:
                mock_clear_output.assert_called_once()
            else:
                mock_clear_output.assert_not_called()

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
            with patch('ipykernel_memvid_extension.chime') as mock_chime:
                mock_chime.themes.return_value = ['theme1', 'theme2', 'theme3']

                magics.list_sound_themes("")

                mock_display_notification.assert_called_once()
                call_args = mock_display_notification.call_args[0][0]
                assert "theme1" in call_args
                assert "theme2" in call_args
                assert "theme3" in call_args

    @custom_parametrize("theme_name,is_valid", [
        ("theme1", True),
        ("invalid_theme", False),
    ])
    @patch('ipykernel_memvid_extension.chime')
    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    def test_set_sound_theme_variants(self, mock_display, mock_chime, theme_name, is_valid, mock_shell):
        """Test set_sound_theme with valid and invalid themes."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            mock_chime.themes.return_value = ['theme1', 'theme2']

            magics.set_sound_theme(theme_name)

            if is_valid:
                mock_chime.theme.assert_called_once_with(theme_name)
                # Should call notification display
                mock_display.assert_called_once()
            else:
                # Should call alert display
                mock_display.assert_called_once()

    @custom_parametrize(("cell_type", "source", "expected_content"), [
        ("code", ["print('hello')\n", "%magic_command\n", "x = 1\n"], ["print('hello')\n", "x = 1\n"]),
        ("code", [], []),
        ("code", ["%magic1\n", "%magic2\n"], []),
    ])
    def test_dump_code_cell_variants(self, cell_type, source, expected_content, mock_shell):
        """Test dumping code cells with different content."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            cell = {
                "cell_type": cell_type,
                "source": source,
                "outputs": []
            }
            result = magics._dump_code_cell(cell)

            if expected_content:
                assert result[:-1] == expected_content
            else:
                assert result == []

    @custom_parametrize(("source", "expected_has_content"), [
        (["# Title\n", "This is a **bold** text\n"], True),
        ([], False),
    ])
    @patch('ipykernel_memvid_extension.markdown')
    @patch('ipykernel_memvid_extension.bs4')
    def test_dump_markdown_cell_variants(self, mock_bs4, mock_markdown, source, expected_has_content, mock_shell):
        """Test dumping markdown cells with different content."""
        # Configure mocks
        mock_markdown.markdown.return_value = "<p>Title</p><p>This is a bold text</p>" if source else ""
        mock_soup = Mock()
        mock_soup.find_all.return_value = ["Title", "This is a bold text"] if source else []
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            cell = {
                "cell_type": "markdown",
                "source": source
            }
            result = magics._dump_markdown_cell(cell)

            if expected_has_content:
                assert all(line.startswith("# ") for line in result[:-1])
                assert len(result) > 0
            else:
                assert result == []

    @custom_parametrize(("range_spec", "expected", "total_cells"), [
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
    ])
    def test_parse_cell_range_valid_specs(self, range_spec, expected, total_cells, mock_shell):
        """Test parsing valid range specifications."""
        with patch('ipykernel_memvid_extension.Magics.__init__'):
            magics = MemvidMagics(mock_shell)
            result = magics._parse_cell_range(range_spec, total_cells)
            assert result == expected

    @custom_parametrize(("range_spec", "total_cells", "expected_error"), [
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

    @custom_parametrize(("magic_name", "args"), [
        ("dump", "test_notebook"),
        ("cleanup", ""),
        ("as_bullet_list", "test_var"),
        ("as_table", "test_var"),
        ("pip_install", "pandas numpy"),
        ("restart_kernel", "-f"),
        ("set_sound_theme", "zelda"),
    ])
    def test_magic_commands_use_modern_argument_parsing(self, magic_name, args, mock_shell):
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

    @custom_parametrize("should_succeed", [True, False])
    @patch('ipykernel_memvid_extension.clear_output')
    @patch('ipykernel_memvid_extension.DisplayContainer.display')
    @patch('ipykernel_memvid_extension.chime')
    def test_load_extension(self, mock_chime, mock_display, mock_clear_output, should_succeed, mock_shell):
        """Test loading IPython extension with success and failure scenarios."""
        mock_chime.theme = Mock()

        if not should_succeed:
            mock_shell.register_magics.side_effect = Exception("Registration failed")

        if should_succeed:
            load_ipython_extension(mock_shell)

            # Check that AST transformer was added
            assert len(mock_shell.ast_transformers) == 1
            assert isinstance(mock_shell.ast_transformers[0], PathCollectorTransformer)

            # Check that RegisterPathFunc was added to namespace
            assert 'RegisterPathFunc' in mock_shell.user_ns

            # Check that magics were registered
            mock_shell.register_magics.assert_called_once()

            # Check that notification was displayed
            mock_display.assert_called_once()
        else:
            with pytest.raises(Exception):
                load_ipython_extension(mock_shell)

            mock_display.assert_called_once()
