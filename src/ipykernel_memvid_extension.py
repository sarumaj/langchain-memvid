"""
IPython Magic Commands for LangChain MemVid

This module provides standalone IPython magic commands that can be loaded
independently of the main package.
"""

import importlib
try:
    module = importlib.import_module('IPython')
    IPYTHON_INSTALLED = True and module is not None
except ImportError:
    IPYTHON_INSTALLED = False

if IPYTHON_INSTALLED:
    import re
    import ast
    import shutil
    import traceback
    from pathlib import Path
    from typing import Optional, Any, Callable
    from types import ModuleType
    from dataclasses import is_dataclass, asdict
    from collections.abc import Iterable
    from enum import Enum
    from functools import wraps
    from IPython import InteractiveShell
    from IPython.core.interactiveshell import ExecutionResult
    from IPython.core.magic import Magics, magics_class, line_magic
    from IPython.display import clear_output, display, HTML, DisplayHandle
    from IPython.utils.io import capture_output

    def ensure_import_module(name: str, shell: Optional[InteractiveShell] = None) -> Optional[ModuleType]:
        """Ensure a module is imported, installing it if necessary.

        Args:
            name (str): The name of the module to import.
            shell (Optional[InteractiveShell], optional): The IPython shell. Defaults to None.

        Returns:
            Optional[ModuleType]: The imported module, or None if the module is not found.
        """
        try:
            return importlib.import_module(name)
        except (ImportError, ModuleNotFoundError):
            if shell is not None:
                shell.run_line_magic('pip', f'install {name}')
                return importlib.import_module(name)
            return None

    chime = ensure_import_module('chime')              # required for sound notifications
    ipywidgets = ensure_import_module('ipywidgets')    # required for tqdm (progress bar)

    def type_to_confirm(prompt: str, confirmation_str: str) -> bool:
        """Interactive confirmation of a string.

        Args:
            prompt (str): The prompt to display.
            confirmation_str (str): The string to confirm.

        Returns:
            bool: True if the string was confirmed, False otherwise.
        """

        ans = None
        while ans != confirmation_str:
            try:
                ans = input(f"{prompt} Type to confirm: \"{confirmation_str}\"")
                if (ans == "" and confirmation_str != "") or ans is None:
                    return False
            except (KeyboardInterrupt, EOFError):
                return False

        return True

    def delete_path(path: Path, force: bool = False) -> bool:
        """Delete a path.

        Args:
            path (Path): The path to delete.
            force (bool): If True, skip confirmation prompt. Defaults to False.
        """
        if not isinstance(path, Path) or not path.exists():
            return False

        if path.is_dir():
            if not force and not type_to_confirm(
                f"Are you sure you want to delete {path}? It is a directory.",
                str(path.name)
            ):
                return False
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        return True

    def class_or_instance_method(func: Callable) -> Callable:
        """A decorator that allows a method to be called as a class method or an instance method.

        Args:
            func (Callable): The method to decorate.

        Returns:
            Callable: The decorated method.
        """

        class ClassOrInstanceMethod:
            def __init__(self, func: Callable):
                self.func = func

            def __get__(self, obj: Any, objtype: Any = None) -> Any:
                @wraps(self.func)
                def wrapper(*args, **kwargs):
                    if obj is None:
                        return self.func(objtype, *args, **kwargs)
                    return self.func(obj, *args, **kwargs)
                return wrapper

        return ClassOrInstanceMethod(func)

    class DisplayContainer(str, Enum):
        ALERT = "red"
        NOTIFICATION = "green"
        WARNING = "orange"

        @class_or_instance_method
        def make_html_container(cls_or_instance, message: str, color: str = "red") -> HTML:
            """Make an HTML container.

            Args:
                message (str): The message to display.
                color (str): The color of the message. Ignored if called as an instance method.

            Returns:
                HTML: The HTML container.
            """

            if isinstance(cls_or_instance, DisplayContainer):
                return HTML(f"<div style='color: {cls_or_instance.value}'>{message}</div>")

            return HTML(f"<div style='color: {color}'>{message}</div>")

        def display(self, message: str) -> Optional[DisplayHandle]:
            """Display an HTML container.

            Args:
                message (str): The message to display.

            Returns:
                Optional[DisplayHandle]: The display handle.
            """
            return display(self.make_html_container(message, self.value), display_id=True)

    def strip_version(package: str) -> str:
        """Strip the version from a package name.

        Args:
            package (str): The package name to strip the version from.
        """
        return re.compile(r'[<>=]+.*').sub('', package)

    class PathCollectorMeta(type):
        """A metaclass to create a callable class (function) that collects Path objects."""

        def __new__(cls, name: str, bases: tuple, attrs: dict):
            """Create a new class (function) that collects Path objects."""
            new_cls = super().__new__(cls, name, bases, attrs)
            new_cls.paths = []
            return new_cls

        def __call__(cls, *args, **kwargs):
            """Make the class itself callable (aka a function)."""
            if len(args) == 1 and isinstance(args[0], Path) and not kwargs:
                cls.paths.append(args[0])
                return args[0]

            raise TypeError(
                f"{cls.__name__}() expected 1 required positional argument of type {Path}, "
                f"got {len(args)} positional arguments and {len(kwargs)} keyword arguments."
            )

    class RegisterPathFunc(metaclass=PathCollectorMeta):
        """A function that collects Path objects.

        Args:
            path: A Path object to collect.

        Returns:
            The Path object.

        Raises:
            TypeError: If the function is called with a non-Path object.
        """

    class PathCollectorTransformer(ast.NodeTransformer):
        """A transformer that collects Path objects from code."""

        # Static methods that return Path objects
        PATH_STATIC_METHODS = {
            'cwd', 'home',
        }

        # Instance methods that return Path objects
        PATH_RETURNING_METHODS = {
            'parent', 'with_name', 'with_suffix', 'with_stem', 'absolute',
            'resolve', 'expanduser', 'expandvars', 'relative_to', 'joinpath'
        }

        def _wrap_in_RegisterPathFunc(self, node: ast.AST) -> ast.Call:
            """Wrap a node in a RegisterPathFunc call."""
            return ast.fix_missing_locations(ast.Call(
                func=ast.Name(id=RegisterPathFunc.__name__, ctx=ast.Load()),
                args=[node],
                keywords=[]
            ))

        def _is_path_constructor(self, node: ast.AST) -> bool:
            """Check if node is a Path constructor call."""
            return (
                isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == Path.__name__
            )

        def _is_path_static_method(self, node: ast.AST) -> bool:
            """Check if node is a Path static method call."""
            return (
                isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == Path.__name__ and
                node.func.attr in self.PATH_STATIC_METHODS
            )

        def _is_path_method_call(self, node: ast.AST) -> bool:
            """Check if node is a Path method call that returns a Path."""
            return (
                isinstance(node, ast.Attribute) and
                node.attr in self.PATH_RETURNING_METHODS
            )

        def _is_path_division_operation(self, node: ast.AST) -> bool:
            """Check if node is a division operation that could result in a Path."""
            return (
                isinstance(node, ast.BinOp) and
                isinstance(node.op, (ast.Div, ast.FloorDiv))
            )

        def _is_likely_path_object(self, node: ast.AST) -> bool:
            """Check if a node is likely to be a Path object."""
            # Direct Path constructor or static method call
            if self._is_path_constructor(node) or self._is_path_static_method(node):
                return True

            # Path method call
            if self._is_path_method_call(node):
                return self._is_likely_path_object(node.value)

            # Path division operation
            if self._is_path_division_operation(node):
                return (
                    self._is_likely_path_object(node.left) or
                    self._is_likely_path_object(node.right)
                )

            return False

        def visit_Call(self, node: ast.Call) -> ast.Call:
            """Visit a Call node and collect Path objects."""
            self.generic_visit(node)

            # Wrap Path constructor calls
            if self._is_path_constructor(node):
                return self._wrap_in_RegisterPathFunc(node)

            return node

        def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
            """Visit a BinOp node and collect Path objects from path operations."""
            self.generic_visit(node)

            # Wrap path division operations
            if self._is_path_division_operation(node) and self._is_likely_path_object(node):
                return self._wrap_in_RegisterPathFunc(node)

            return node

        def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
            """Visit an Attribute node and collect Path objects from method calls."""
            # Wrap Path method calls that return Path objects
            if self._is_path_method_call(node) and self._is_likely_path_object(node.value):
                return self._wrap_in_RegisterPathFunc(node)

            # If no transformation needed, visit children
            self.generic_visit(node)
            return node

    @magics_class
    class MemvidMagics(Magics):
        """IPython magic commands for LangChain MemVid."""
        def __init__(self, shell: InteractiveShell):
            """Initialize the MemvidMagics class.

            Args:
                shell (InteractiveShell): The IPython shell.
            """
            super().__init__(shell)

            # define options table
            self.options_table = dict.fromkeys(("cleanup", "restart_kernel"), "")

            self.shell = shell
            self.shell.events.register('post_run_cell', self.play_sound)

            global chime
            self.chime_themes = chime.themes() if chime is not None else []

        def play_sound(self, result: ExecutionResult):
            """Post-run cell event handler.

            If chime is installed, it will play a notification sound based on the
            success or failure of the cell execution.

            Args:
                result (ExecutionResult): The result of the cell execution.
            """
            global chime
            if chime is not None:
                chime.notify("success" if result.success else "error", sync=True, raise_error=False)
            else:
                DisplayContainer.ALERT.display(
                    "<b>chime [pip install chime] is not installed, "
                    "so no notification sound will be played.</b>",
                )

        @line_magic
        def as_bullet_list(self, var: str):
            """Display a bullet list of the user's namespace.

            The command will display a bullet list of the user's namespace.
            The line argument is the name of the variable to display.
            The variable must be an iterable.

            Args:
                var (str): The variable to display.
            """

            var = var.strip()
            if not var:
                DisplayContainer.ALERT.display(
                    "<b>No variable name provided. "
                    "Use %as_bullet_list <variable_name> to display the variable.</b>"
                )
                return

            if var not in self.shell.user_ns:
                DisplayContainer.ALERT.display(
                    f"<b>Variable {var} not found in the user's namespace.</b>"
                )
                return

            obj = self.shell.user_ns[var]
            if not isinstance(obj, Iterable):
                DisplayContainer.ALERT.display(
                    f"<b>Variable {var} is not an iterable.</b>"
                )
                return

            display(HTML(
                "<p><b>Output:</b></p><ul>{list}</ul>".format(
                    list="".join(f"<li>{item}</li>" for item in obj)
                )
            ))

        @line_magic
        def as_table(self, var: str):
            """Display a table of the user's namespace.

            The command will display a table of the user's namespace.
            The line argument is the name of the variable to display.
            The variable must be an iterable.

            Args:
                var (str): The variable to display.
            """
            var = var.strip()
            if not var:
                DisplayContainer.ALERT.display(
                    "<b>No variable name provided. "
                    "Use %as_table <variable_name> to display the variable.</b>"
                )
                return

            if var not in self.shell.user_ns:
                DisplayContainer.ALERT.display(
                    f"<b>Variable {var} not found in the user's namespace.</b>"
                )
                return

            obj = self.shell.user_ns[var]
            if isinstance(obj, dict) or is_dataclass(obj):
                obj = [obj]

            if not isinstance(obj, Iterable):
                DisplayContainer.ALERT.display(
                    f"<b>Variable {var} is not an iterable.</b>"
                )
                return

            obj = [asdict(item) if is_dataclass(item) else item for item in obj]
            if not all(isinstance(item, dict) for item in obj):
                DisplayContainer.ALERT.display(
                    f"<b>Variable {var} is not a list of dictionaries.</b>"
                )
                return

            if len(obj) == 0:
                DisplayContainer.ALERT.display(
                    f"<b>Variable {var} is an empty iterable.</b>"
                )
                return

            if len(obj) == 1:
                headers = ["Name", "Value"]
                rows = [[str(k).replace("_", " ").title(), v] for k, v in obj[0].items()]
            else:
                headers = [str(k).replace("_", " ").title() for k in obj[0].keys()]
                rows = [[i.get(k, "") for k in obj[0].keys()] for i in obj]

            display(HTML(
                "<table><tr>{headers_html}</tr>{rows_html}</table>".format(
                    headers_html=''.join((
                        f'<th style="text-align: left"><b>{header}</b></th>'
                        for header in headers
                    )),
                    rows_html='<tr>' + '</tr><tr>'.join((
                        ''.join((
                            f"<td style='text-align: left'>{cell}</td>"
                            for cell in row
                        ))
                        for row in rows
                    )) + '</tr>'
                )
            ))

        @line_magic
        def cleanup(self, params: str):
            """Garbage collect paths from the user's namespace.

            The command will remove paths from the user's namespace and the RegisterPathFunc function.
            The line argument is the name of the variable to display.
            The variable must be an iterable.

            Args:
                params (str): The options and arguments to evaluate.

            Options:
                -f, --force: Force the cleanup.

            Examples:
                %cleanup -f
            """
            opts, args = self.parse_options(params, "f", "force", mode="list", list_all=True)
            force = "f" in opts or "force" in args

            """Garbage collect paths from the user's namespace."""
            # Collect unique named Path objects from the user's namespace
            garbage = {k for k, v in self.shell.user_ns.items()if isinstance(v, Path)}
            lookup = {k: v for k, v in self.shell.user_ns.items() if k in garbage}
            # Collect un-named Path objects from the RegisterPathFunc function
            lookup.update({f"object_{id(v)}": v for v in RegisterPathFunc.paths if v not in lookup.values()})

            removed = {}
            for name, obj in lookup.copy().items():
                if name in self.shell.user_ns:
                    self.shell.user_ns.pop(name, None)
                if obj in RegisterPathFunc.paths:
                    RegisterPathFunc.paths.remove(obj)
                if delete_path(obj, force=force):
                    removed[name] = lookup.pop(name)

            if not removed:
                DisplayContainer.NOTIFICATION.display("<b>No paths to clean up.</b>")
                return

            DisplayContainer.WARNING.display(
                (
                    "<b>Cleaned up:</b>"
                    "<table><tr>{headers}</tr>{rows}</table>"
                ).format(
                    headers="".join(
                        f'<th style="text-align: left"><b>{header}</b></th>'
                        for header in ("Name", "Type", "Object")
                    ),
                    rows="".join(
                        '<tr>'
                        f'<td style="text-align: left">{name}</td>'
                        f'<td style="text-align: left">{type(obj).__name__}</td>'
                        f'<td style="text-align: left">{str(obj)}</td>'
                        '</tr>'
                        for name, obj in removed.items()
                    )
                )
            )

        @line_magic
        def mute(self, _: str):
            """Mute the IPython shell.

            The command will unregister the post_run_cell event handler.
            The line argument is the name of the variable to display.
            The variable must be an iterable.
            """
            self.shell.events.unregister('post_run_cell', self.play_sound)
            DisplayContainer.NOTIFICATION.display("<b>Muted IPython shell.</b>")

        @line_magic
        def unmute(self, _: str):
            """Unmute the IPython shell.

            The command will register the post_run_cell event handler.
            The line argument is the name of the variable to display.
            The variable must be an iterable.
            """
            self.shell.events.register('post_run_cell', self.play_sound)
            DisplayContainer.NOTIFICATION.display("<b>Unmuted IPython shell.</b>")

        @line_magic
        def pip_install(self, packages: str):
            """Install packages.

            The command will install packages using the pip command.
            The line argument is the name of the package to install.
            The variable must be an iterable.

            Args:
                packages (str): The packages to install.
            """
            if not (packages := packages.strip().strip(';')):
                DisplayContainer.ALERT.display("<b>No packages to install.</b>")
                return

            try:
                self.shell.run_line_magic('pip', f"install {packages}")
                clear_output(wait=False)

                with capture_output() as captured:
                    self.shell.run_line_magic('pip', "freeze")

                available_packages = captured.stdout.strip().splitlines()
                installed_packages = [
                    item for item in available_packages if any(
                        item.startswith(strip_version(package).strip())
                        for package in packages.split(' ')
                        if package
                    )
                ]

                DisplayContainer.NOTIFICATION.display(
                    "<p><b>Installed:</b></p>"
                    "<table><tr>{headers}</tr>{rows}</table>".format(
                        headers="".join(
                            f'<th style="text-align: left"><b>{header}</b></th>'
                            for header in ('Package', 'Version')
                        ),
                        rows="".join(
                            '<tr>' + ''.join(
                                f'<td style="text-align: left">{element}</td>'
                                for element in item.split("==")
                            ) + '</tr>'
                            for item in installed_packages
                        )
                    )
                )

            except Exception as e:
                DisplayContainer.ALERT.display(
                    "<b>Error installing:</b>"
                    f"<ul>{''.join(f'<li>{item}</li>' for item in packages.split(' '))}</ul>"
                    f"<p>{str(e)}</p>"
                    f"<p><pre>{traceback.format_exc()}</pre></p>"
                )

        @line_magic
        def restart_kernel(self, params: str):
            """Restart the Jupyter kernel.

            The command will restart the Jupyter kernel.
            The line argument is the name of the variable to display.
            The variable must be an iterable.

            Args:
                params (str): The options and arguments to evaluate.

            Options:
                -f, --force: Force the restart.
            """
            opts, args = self.parse_options(params, "f", "force", mode="list", list_all=True)
            force = "f" in opts or "force" in args

            if force or type_to_confirm("Are you sure you want to restart the Jupyter kernel?", "yes"):
                handle = DisplayContainer.WARNING.display(
                    "<b>Restarting Jupyter kernel...</b>"
                    "<script>"
                    "Jupyter.notebook.kernel.restart();"
                    "</script>"
                )
                handle.update(DisplayContainer.NOTIFICATION.make_html_container(
                    "<b>Jupyter kernel restarted.</b>"
                ))
            else:
                DisplayContainer.WARNING.display("<b>Jupyter kernel restart cancelled.</b>")

        @line_magic
        def list_sound_themes(self, _: str):
            """Display the available sound themes.

            The command will display the available sound themes.
            The line argument is the name of the variable to display.
            The variable must be an iterable.
            """
            if self.chime_themes:
                DisplayContainer.NOTIFICATION.display(
                    f"<p><b>Available sound themes:</b></p>"
                    f"<ul>{''.join(f'<li>{theme}</li>' for theme in self.chime_themes)}</ul>"
                )
            else:
                DisplayContainer.ALERT.display(
                    "<b>chime [pip install chime] is not installed, "
                    "so no sound themes will be displayed.</b>"
                )

        @line_magic
        def set_sound_theme(self, theme: str):
            """Set the chime theme.

            The command will set the chime theme.
            The line argument is the name of the variable to display.
            The variable must be an iterable.

            Args:
                theme (str): The chime theme to set.
            """
            if (theme := theme.strip().strip(';')) not in self.chime_themes:
                DisplayContainer.ALERT.display(
                    f"<p><b>Sound theme {theme} not found in sound themes:</b></p>"
                    f"<ul>{''.join(f'<li>{theme}</li>' for theme in self.chime_themes)}</ul>"
                )
                return

            global chime
            if chime is not None:
                chime.theme(theme)
                DisplayContainer.NOTIFICATION.display(f"<b>Set sound theme to {theme}.</b>")
            else:
                DisplayContainer.ALERT.display(
                    "<b>chime [pip install chime] is not installed, "
                    "so no sound theme will be set.</b>"
                )

    def load_ipython_extension(shell: InteractiveShell):
        """Load the IPython extension."""
        global chime       # required for sound notification
        if not chime:
            chime = ensure_import_module('chime', shell)
            chime.theme('big-sur') if chime is not None else None

        global ipywidgets  # required for tqdm (progress bar)
        if not ipywidgets:
            ipywidgets = ensure_import_module('ipywidgets', shell)

        if not shell.last_execution_result or shell.last_execution_result.success:
            clear_output(wait=False)

        try:
            shell.ast_transformers.append(PathCollectorTransformer())
            shell.user_ns[RegisterPathFunc.__name__] = RegisterPathFunc
            shell.register_magics(MemvidMagics)
            DisplayContainer.NOTIFICATION.display("<b>Successfully loaded langchain_memvid IPython extension.</b>")
        except Exception as e:
            DisplayContainer.ALERT.display(
                "<p><b>Error loading langchain_memvid IPython extension:</b></p>"
                f"<p>{str(e)}</p>"
                f"<pre>{traceback.format_exc()}</pre>"
            )
            raise
