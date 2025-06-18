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
    import shutil
    import traceback
    from pathlib import Path
    from typing import Optional
    from types import ModuleType
    from dataclasses import is_dataclass, asdict
    from collections.abc import Iterable
    from IPython import InteractiveShell
    from IPython.core.interactiveshell import ExecutionResult
    from IPython.core.magic import Magics, magics_class, line_cell_magic, line_magic
    from IPython.display import clear_output, display, HTML

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

    chime = ensure_import_module('chime')              # required for chime (sound notification)
    ipywidgets = ensure_import_module('ipywidgets')    # required for tqdm (progress bar)

    def delete_path(path: Path):
        """Delete a path.

        Args:
            path (Path): The path to delete.
        """
        if not isinstance(path, Path) or not path.exists():
            return

        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    def display_html(message: str, color: str = "red"):
        """Display an HTML message.

        Args:
            message (str): The message to display.
            color (str): The color of the message.
        """
        display(HTML(
            f"<div style='color: {color}'>{message}</div>"
        ))

    def display_alert(message: str):
        """Display an alert message.

        Args:
            message (str): The message to display.
        """
        display_html(message, "red")

    def display_notification(message: str):
        """Display a notification message.

        Args:
            message (str): The message to display.
        """
        display_html(message, "green")

    def display_warning(message: str):
        """Display a warning message.

        Args:
            message (str): The message to display.
        """
        display_html(message, "orange")

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
            self.shell.events.register('post_run_cell', self.post_run_cell)

        def post_run_cell(self, result: ExecutionResult):
            """Post-run cell event handler.

            If chime is installed, it will play a notification sound based on the
            success or failure of the cell execution.

            Args:
                result (ExecutionResult): The result of the cell execution.
            """
            global chime
            if chime is not None:
                chime.notify("success" if result.success else "error", sync=False, raise_error=False)
            else:
                display_alert(
                    "<b>chime [pip install chime] is not installed, "
                    "so no notification sound will be played.</b>"
                )

        @line_magic
        def as_bullet_list(self, line: str):
            """Display a bullet list of the user's namespace.

            The command will display a bullet list of the user's namespace.
            The line argument is the name of the variable to display.
            The variable must be an iterable.

            Args:
                line (str): The line to display.
            """

            line = line.strip()
            if not line:
                display_alert(
                    "<b>No variable name provided. "
                    "Use %as_bullet_list <variable_name> to display the variable.</b>"
                )
                return

            if line not in self.shell.user_ns:
                display_alert(f"<b>Variable {line} not found in the user's namespace.</b>")
                return

            obj = self.shell.user_ns[line]
            if not isinstance(obj, Iterable):
                display_alert(f"<b>Variable {line} is not an iterable.</b>")
                return

            display(HTML(
                "<p><b>Output:</b></p><ul>{list}</ul>".format(
                    list="".join(f"<li>{item}</li>" for item in obj)
                )
            ))

        @line_magic
        def as_table(self, line: str):
            """Display a table of the user's namespace.

            The command will display a table of the user's namespace.
            The line argument is the name of the variable to display.
            The variable must be an iterable.

            Args:
                line (str): The line to display.
            """
            line = line.strip()
            if not line:
                display_alert(
                    "<b>No variable name provided. "
                    "Use %as_table <variable_name> to display the variable.</b>"
                )
                return

            if line not in self.shell.user_ns:
                display_alert(f"<b>Variable {line} not found in the user's namespace.</b>")
                return

            obj = self.shell.user_ns[line]
            if isinstance(obj, dict) or is_dataclass(obj):
                obj = [obj]

            if not isinstance(obj, Iterable):
                display_alert(f"<b>Variable {line} is not an iterable.</b>")
                return

            obj = [asdict(item) if is_dataclass(item) else item for item in obj]
            if not all(isinstance(item, dict) for item in obj):
                display_alert(f"<b>Variable {line} is not a list of dictionaries.</b>")
                return

            if len(obj) == 0:
                display_alert(f"<b>Variable {line} is an empty iterable.</b>")
                return

            if len(obj) == 1:
                headers = []
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

        @line_cell_magic
        def cleanup(self, line: str, cell: Optional[str] = None):
            """Garbage collect paths from the user's namespace."""
            garbage = {k for k, v in self.shell.user_ns.items()if isinstance(v, Path)}
            lookup = {k: v for k, v in self.shell.user_ns.items() if k in garbage}

            for name in garbage:
                delete_path(self.shell.user_ns.pop(name, None))

            display_warning(
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
                        for name, obj in lookup.items()
                    )
                )
            )

        @line_magic
        def pip_install(self, line: str):
            """Install packages."""
            if line.startswith('install '):
                line = line.lstrip('install ')

            if not line:
                display_alert("<b>No command provided.</b>")

            try:
                self.shell.run_line_magic('pip', f"install {line}")
                clear_output()
                display_notification(
                    "<b>Installed:</b>"
                    f"<ul>{''.join(f'<li>{item}</li>' for item in line.split(' '))}</ul>"
                )

            except Exception as e:
                display_alert(
                    "<b>Error installing:</b>"
                    f"<ul>{''.join(f'<li>{item}</li>' for item in line.split(' '))}</ul>"
                    f"<p>{str(e)}</p>"
                    f"<p><pre>{traceback.format_exc()}</pre></p>"
                )

        @line_cell_magic
        def restart_kernel(self, line: str, cell: Optional[str] = None):
            """Restart the Jupyter kernel."""
            display(HTML("<script>Jupyter.notebook.kernel.restart()</script>"))

    def load_ipython_extension(shell: InteractiveShell):
        """Load the IPython extension."""
        global chime       # required for chime (sound notification)
        if not chime:
            chime = ensure_import_module('chime', shell)
            chime.theme('big-sur')

        global ipywidgets  # required for tqdm (progress bar)
        if not ipywidgets:
            ipywidgets = ensure_import_module('ipywidgets', shell)

        if not shell.last_execution_result or shell.last_execution_result.success:
            clear_output()

        try:
            shell.register_magics(MemvidMagics)
            display_notification("<b>Successfully loaded langchain_memvid IPython extension.</b>")
        except Exception as e:
            display_alert(
                "<p><b>Error loading langchain_memvid IPython extension:</b></p>"
                f"<p>{str(e)}</p>"
                f"<pre>{traceback.format_exc()}</pre>"
            )
            raise
