"""
IPython Magic Commands for LangChain MemVid

This module provides standalone IPython magic commands that can be loaded
independently of the main package.
"""

import importlib
try:
    module = importlib.import_module('IPython')
    IPYTHON_INSTALLED = True and module is not None
except (ImportError, ModuleNotFoundError):
    IPYTHON_INSTALLED = False

if IPYTHON_INSTALLED:
    import re
    import ast
    import json
    import shutil
    import traceback
    import textwrap
    from pathlib import Path
    from typing import Optional, Any, Callable, List
    from types import ModuleType
    from dataclasses import is_dataclass, asdict
    from collections.abc import Iterable
    from enum import Enum
    from functools import wraps
    from IPython import InteractiveShell, get_ipython
    from IPython.core.interactiveshell import ExecutionResult
    from IPython.core.magic import Magics, magics_class, line_magic
    from IPython.display import clear_output, display, HTML, DisplayHandle
    from IPython.utils.io import capture_output
    from IPython.core.magic_arguments import magic_arguments, argument, parse_argstring

    def ensure_import_module(
        name: str, *,
        package_name: Optional[str] = None,
        shell: Optional[InteractiveShell] = None,
    ) -> Optional[ModuleType]:
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
            if shell is None:
                shell = get_ipython()
            if shell is not None:
                shell.run_line_magic('pip', f'install {package_name if package_name else name}')
                return importlib.import_module(name)
            return None

    chime = ensure_import_module('chime')                             # required for sound notifications
    ipywidgets = ensure_import_module('ipywidgets')                   # required for tqdm (progress bar)
    markdown = ensure_import_module('markdown')                       # required for markdown parsing
    markdownify = ensure_import_module('markdownify')                 # required for html parsing
    bs4 = ensure_import_module('bs4', package_name='beautifulsoup4')  # required for html parsing

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
        return re.sub(r'[<>=]+.*', '', package)

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

            self.shell = shell
            self.shell.events.register('post_run_cell', self.play_sound)

        def _dump_code_cell(self, cell: dict) -> list[str]:
            """Dump a code cell.

            Args:
                raw_cell (str): The cell to dump.

            Returns:
                list[str]: The dumped cell. Lines end with a newline.
            """
            lines = [
                f"{line}\n" for raw_line in cell.get("source", [])
                for line in [raw_line.rstrip().rstrip("\n")]
                if line and not line.startswith("%")
            ]

            output_lines = []
            for output in cell.get("outputs", []):
                match output.get("output_type", ""):
                    case "stream":
                        if output.get("name", "") == "stderr":
                            continue

                        output_lines.extend([
                            f"# >>> {line}\n" for raw_line in output.get("text", [])
                            for trimmed_line in [raw_line.rstrip().rstrip('\n')]
                            for line in textwrap.wrap(trimmed_line, width=115)
                            if line
                        ])

                    case "display_data" | "execute_result":
                        data = output.get("data", {})
                        if html_data := data.get("text/html", []):
                            markdown_string = markdownify.markdownify(''.join(html_data))
                            output_lines.extend([
                                f"# >>> {line}\n" for raw_line in markdown_string.splitlines()
                                for trimmed_line in [raw_line.rstrip().rstrip('\n')]
                                for line in textwrap.wrap(trimmed_line, width=115)
                                if line
                            ])

                        elif text_data := data.get("text/plain", []):
                            output_lines.extend([
                                f"# >>> {line}\n" for raw_line in text_data
                                for trimmed_line in [raw_line.rstrip().rstrip('\n')]
                                for line in textwrap.wrap(trimmed_line, width=115)
                                if line
                            ])

                    case "error" | "_":
                        continue

            if output_lines:
                lines += ["\n", "# Output:\n"] + output_lines

            return lines + ["\n"] if lines else []

        def _dump_markdown_cell(self, cell: dict) -> List[str]:
            """Dump a markdown cell.

            Args:
                cell (dict): The cell to dump.

            Returns:
                list[str]: The dumped cell.
                    Lines are prefixed with a comment character and wrapped to 120 characters.
                    Lines end with a newline.
            """
            markdown_string = "\n".join(cell.get("source", []))
            html_string = markdown.markdown(markdown_string)
            soup = bs4.BeautifulSoup(html_string, 'html.parser')
            lines = [
                f"# {line}\n" for raw_line in ''.join(soup.find_all(string=True)).splitlines()
                for trimmed_line in [raw_line.rstrip().rstrip('\n')]
                for line in textwrap.wrap(trimmed_line, width=118)
                if line
            ]
            return lines + ["\n"] if lines else []

        def _parse_cell_range(self, range_spec: str, total_cells: int) -> List[int]:
            """Parse cell range specification following IPython conventions.

            This method is inspired by IPython's history_manager.get_range_by_str()
            but adapted for notebook cell indices.

            Args:
                range_spec (str): Range specification (e.g., "1:5" or "1,3,5")
                total_cells (int): Total number of cells

            Returns:
                list: List of cell indices (0-based)

            Raises:
                ValueError: If range specification is invalid

            Examples:
                "1"      -> [0]
                "1:5"    -> [0, 1, 2, 3, 4]
                "1,3,5"  -> [0, 2, 4]
                "1:"     -> [0, 1, 2, ..., total_cells-1]
                ":5"     -> [0, 1, 2, 3, 4]
                ":-2"    -> [0, 1, 2, ..., total_cells-2]
                "-2:"    -> [total_cells-2, total_cells-1]
            """
            if not (range_spec := range_spec.strip()):
                return list(range(total_cells))

            indices = set()

            single_pattern = r'^(-?\d+)$'
            range_pattern = r'^(-?\d*):(-?\d*)$'

            for part in range_spec.split(','):
                if not (part := part.strip()):
                    continue

                if (single_match := re.match(single_pattern, part)):
                    if (idx := int(single_match.group(1))) < 0:
                        idx += total_cells + 1
                    if idx < 1 or idx > total_cells:
                        raise ValueError(f"Cell index must be between 1 and {total_cells}, got {idx}")
                    indices.add(idx - 1)
                    continue

                if (range_match := re.match(range_pattern, part)):
                    start_str, end_str = range_match.groups()
                    if (start_idx := 1 if not start_str else int(start_str)) < 0:
                        start_idx += total_cells + 1
                    if start_idx < 1:
                        raise ValueError(f"Start index must be >= 1, got {start_idx}")

                    if (end_idx := total_cells if not end_str else int(end_str)) < 0:
                        end_idx += total_cells
                    if end_idx < 1:
                        raise ValueError(f"End index must be >= 1, got {end_idx}")

                    if start_idx > end_idx:
                        raise ValueError(f"Start index ({start_idx}) must be <= end index ({end_idx})")
                    if start_idx > total_cells:
                        raise ValueError(f"Start index ({start_idx}) must be <= total cells ({total_cells})")
                    if end_idx > total_cells:
                        raise ValueError(f"End index ({end_idx}) must be <= total cells ({total_cells})")

                    indices.update(range(start_idx - 1, end_idx))
                    continue

                raise ValueError(f"Invalid range format: {part}")

            if not indices:
                raise ValueError("No valid cell indices found in range specification")

            return sorted(indices)

        def play_sound(self, result: ExecutionResult):
            """Post-run cell event handler.

            If chime is installed, it will play a notification sound based on the
            success or failure of the cell execution.

            Args:
                result (ExecutionResult): The result of the cell execution.
            """
            chime.notify("success" if result.success else "error", sync=True, raise_error=False)

        @magic_arguments()
        @argument('variable_name', nargs=1, help="Name of the variable to display as a bullet list.")
        @line_magic
        def as_bullet_list(self, parameter_s=''):
            """Display a bullet list of an iterable variable.

            This command displays an iterable variable from the user's namespace
            as a formatted bullet list.

            Examples:
                %as_bullet_list my_list
            """
            args = parse_argstring(self.as_bullet_list, parameter_s)
            var = args.variable_name[0] if args.variable_name else None

            if not var or not var.strip():
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

        @magic_arguments()
        @argument('variable_name', nargs=1, help="Name of the variable to display as a table.")
        @line_magic
        def as_table(self, parameter_s=''):
            """Display a table of an iterable variable.

            This command displays an iterable variable from the user's namespace
            as a formatted table. Works with lists of dictionaries or single dictionaries.

            Examples:
                %as_table my_data
            """
            args = parse_argstring(self.as_table, parameter_s)
            var = args.variable_name[0] if args.variable_name else ""

            if not (var := var.strip()):
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

        @magic_arguments()
        @argument(
            '-f', '--force', dest='force', action='store_true', default=False,
            help="Force cleanup without confirmation."
        )
        @line_magic
        def cleanup(self, parameter_s=''):
            """Garbage collect paths from the user's namespace.

            This command removes Path objects from the user's namespace and the
            RegisterPathFunc function, optionally deleting the files/directories.

            Examples:
                %cleanup
                %cleanup -f
            """
            args = parse_argstring(self.cleanup, parameter_s)

            # Collect unique named Path objects from the user's namespace
            garbage = {k for k, v in self.shell.user_ns.items() if isinstance(v, Path)}
            lookup = {k: v for k, v in self.shell.user_ns.items() if k in garbage}
            # Collect un-named Path objects from the RegisterPathFunc function
            lookup.update({f"object_{id(v)}": v for v in RegisterPathFunc.paths if v not in lookup.values()})

            removed = {}
            for name, obj in lookup.copy().items():
                if name in self.shell.user_ns:
                    self.shell.user_ns.pop(name, None)
                if obj in RegisterPathFunc.paths:
                    RegisterPathFunc.paths.remove(obj)
                if delete_path(obj, force=args.force):
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

        @magic_arguments()
        @argument(
            '-f', '--force', dest='force', action='store_true', default=False,
            help="Force overwrite if output file exists."
        )
        @argument(
            '-r', '--range', dest='range_spec', type=str, default=None,
            help="Range of cells to dump (e.g., '1-5' or '1,3,5')."
        )
        @argument(
            '-o', '--output', dest='cell_outputs', action='store_true', default=False,
            help="Dump cell outputs."
        )
        @argument('notebook_name', nargs="?", default=None, help="Name of the notebook to dump.")
        @line_magic
        def dump(self, parameter_s=''):
            """Dump notebook content to a Python file with markdown cells as comments and omit macro calls.

            This command reads a .ipynb file and converts it to a .py file, filtering out
            magic commands and converting markdown cells to comments.

            Examples:
                %dump                       # auto-detect notebook name
                %dump my_notebook
                %dump -f my_notebook
                %dump -r 1:5 my_notebook
                %dump -r 1: my_notebook
                %dump -r :5 my_notebook
                %dump -r -2: my_notebook
                %dump -r -2:5 my_notebook
                %dump -r 1,3,5 my_notebook
                %dump -o my_notebook        # dump cell outputs
            """
            args = parse_argstring(self.dump, parameter_s)

            notebook_name = (
                args.notebook_name[0] if args.notebook_name else
                self.shell.user_ns.get("__session__", "") or
                self.shell.user_ns.get("__vsc_ipynb_file__", "")
            )
            if not (notebook_name := notebook_name.strip()):
                DisplayContainer.ALERT.display(
                    "<b>No notebook name provided and could not be auto-detected. "
                    "Use %dump [options] <notebook_name> to dump the notebook.</b>"
                )
                return

            source = Path(notebook_name).with_suffix(".ipynb")
            output = Path(notebook_name).with_suffix(".py")

            if not source.exists():
                DisplayContainer.ALERT.display(f"<b>Notebook {source} not found.</b>")
                return

            if (
                not args.force and
                output.exists() and
                not type_to_confirm(f"Are you sure you want to overwrite the file {output.name}?", output.name)
            ):
                DisplayContainer.ALERT.display(
                    f"<b>Output {output} already exists. Use %dump -f <notebook_name> to force overwrite.</b>"
                )
                return

            source_code = json.loads(source.read_text())
            cells = source_code.get("cells", [])
            selected_indices = list(range(len(cells)))
            if args.range_spec:
                try:
                    selected_indices = self._parse_cell_range(args.range_spec, len(cells))
                    cells = [cells[i] for i in selected_indices]
                except ValueError as e:
                    DisplayContainer.ALERT.display(
                        f"<b>Invalid range specification: {str(e)}</b>"
                    )
                    return

            lines = []
            for cell in cells:
                match cell.get("cell_type", ""):
                    case "code":
                        if args.cell_outputs:
                            lines.extend(self._dump_code_cell(cell))
                        else:  # remove outputs from code cell to avoid dumping them
                            lines.extend(self._dump_code_cell({k: v for k, v in cell.items() if k != "outputs"}))
                    case "markdown":
                        lines.extend(self._dump_markdown_cell(cell))
                    case _:
                        continue

            if lines:
                output.write_text(
                    "".join(
                        [f"# Generated by ipykernel_memvid_extension from %dump in {source.name}. DO NOT EDIT.\n\n"] +
                        lines
                    ).rstrip("\n") + "\n"
                )

            DisplayContainer.NOTIFICATION.display(
                f"<b>Dumped cells {[i+1 for i in selected_indices]} from {source.name} to {output.name}.</b>"
            )

        @line_magic
        def mute(self, parameter_s=''):
            """Mute the IPython shell.

            This command disables sound notifications by unregistering the
            post_run_cell event handler.

            Examples:
                %mute
            """

            try:
                self.shell.events.unregister('post_run_cell', self.play_sound)
            except ValueError:
                pass
            DisplayContainer.NOTIFICATION.display("<b>Muted IPython shell.</b>")

        @line_magic
        def unmute(self, parameter_s=''):
            """Unmute the IPython shell.

            This command enables sound notifications by registering the
            post_run_cell event handler.

            Examples:
                %unmute
            """

            self.shell.events.register('post_run_cell', self.play_sound)
            DisplayContainer.NOTIFICATION.display("<b>Unmuted IPython shell.</b>")

        @magic_arguments()
        @argument('packages', nargs='*', help="Packages to install via pip.")
        @line_magic
        def pip_install(self, parameter_s=''):
            """Install packages using pip with visual feedback.

            This command installs packages using pip and provides visual feedback
            about the installation process and results.

            Examples:
                %pip_install pandas numpy
                %pip_install requests
            """
            args = parse_argstring(self.pip_install, parameter_s)
            packages = ' '.join(args.packages) if args.packages else ''

            if not (packages := packages.strip()):
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

        @magic_arguments()
        @argument(
            '-f', '--force', dest='force', action='store_true', default=False,
            help="Force restart without confirmation."
        )
        @line_magic
        def restart_kernel(self, parameter_s=''):
            """Restart the Jupyter kernel.

            This command restarts the Jupyter kernel, optionally without confirmation.

            Examples:
                %restart_kernel
                %restart_kernel -f
            """
            args = parse_argstring(self.restart_kernel, parameter_s)
            force = args.force

            if force or type_to_confirm("Are you sure you want to restart the Jupyter kernel?", "yes"):
                handle = DisplayContainer.WARNING.display("<b>Restarting Jupyter kernel...</b>")
                self.shell.run_cell(
                    raw_cell=(
                        "%%javascript\n"
                        "(() => {\n"
                        "    let kernel = null;\n"
                        "    if (typeof IPython !== 'undefined' && IPython.notebook?.kernel) {\n"
                        "        kernel = IPython.notebook.kernel;\n"
                        "    } else if (typeof Jupyter !== 'undefined' && Jupyter.notebook?.kernel) {\n"
                        "        kernel = Jupyter.notebook.kernel;\n"
                        "    } else if (typeof JupyterLab !== 'undefined' && JupyterLab.serviceManager) {\n"
                        "        try {\n"
                        "            const sessions = JupyterLab.serviceManager.sessions.running();\n"
                        "            for (const session of sessions) {\n"
                        "                if (session.kernel) {\n"
                        "                    kernel = session.kernel;\n"
                        "                    break;\n"
                        "                }\n"
                        "            }\n"
                        "        } catch (e) {\n"
                        "            console.warn('JupyterLab kernel detection failed:', e);\n"
                        "        }\n"
                        "    }\n"
                        "    if (kernel?.restart) {\n"
                        "        kernel.restart();\n"
                        "    } else {\n"
                        "        console.error('No compatible kernel found');\n"
                        "    }\n"
                        "})();"
                    ),
                    silent=True,
                    store_history=False,
                )
                handle.update(DisplayContainer.NOTIFICATION.make_html_container(
                    "<b>Jupyter kernel restarted.</b>"
                ))
            else:
                DisplayContainer.WARNING.display("<b>Jupyter kernel restart cancelled.</b>")

        @line_magic
        def list_sound_themes(self, parameter_s=''):
            """Display the available sound themes.

            This command lists all available sound themes for chime notifications.

            Examples:
                %list_sound_themes
            """
            if (chime_themes := chime.themes()):
                DisplayContainer.NOTIFICATION.display(
                    f"<p><b>Available sound themes:</b></p>"
                    f"<ul>{''.join(f'<li>{theme}</li>' for theme in chime_themes)}</ul>"
                )
            else:
                DisplayContainer.ALERT.display("<b>No sound themes available.</b>")

        @magic_arguments()
        @argument('theme_name', nargs=1, help="Name of the sound theme to set.")
        @line_magic
        def set_sound_theme(self, parameter_s=''):
            """Set the chime sound theme.

            This command sets the sound theme for chime notifications.

            Examples:
                %set_sound_theme zelda
            """
            args = parse_argstring(self.set_sound_theme, parameter_s)
            theme = args.theme_name[0] if args.theme_name else ""

            if not (theme := theme.strip()):
                DisplayContainer.ALERT.display("<b>No theme name provided.</b>")
                return

            if theme not in (chime_themes := chime.themes()):
                DisplayContainer.ALERT.display(
                    f"<p><b>Sound theme {theme} not found in sound themes:</b></p>"
                    f"<ul>{''.join(f'<li>{theme}</li>' for theme in chime_themes)}</ul>"
                )
                return

            chime.theme(theme)
            DisplayContainer.NOTIFICATION.display(f"<b>Set sound theme to {theme}.</b>")

    def load_ipython_extension(shell: InteractiveShell):
        """Load the IPython extension."""
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
