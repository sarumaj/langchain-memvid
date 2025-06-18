"""
LangChain MemVid - Video-based Vector Storage for LangChain

This package provides a video-based vector storage solution that is compatible
with the LangChain ecosystem. It uses QR codes to store document embeddings
in video format, enabling efficient storage and retrieval of document vectors.
"""

from .config import VideoConfig, QRCodeConfig, IndexConfig, VectorStoreConfig
from .vectorstore import VectorStore
from .retriever import Retriever
from .encoder import Encoder
from .index import IndexManager
from .exceptions import (
    MemVidError,
    EncodingError,
    RetrievalError,
    MemVidIndexError,
    VideoProcessingError,
    QRCodeError,
)

__all__ = [
    k for k, v in globals().items() if v in (
        # Core components
        VectorStore,
        Retriever,
        Encoder,
        IndexManager,

        # Configuration classes
        VideoConfig,
        QRCodeConfig,
        IndexConfig,
        VectorStoreConfig,

        # Exceptions
        MemVidError,
        EncodingError,
        RetrievalError,
        MemVidIndexError,
        VideoProcessingError,
        QRCodeError,
    )
]


import importlib
try:
    _ = importlib.import_module('IPython')
    IPYTHON_INSTALLED = True
except (ImportError, ModuleNotFoundError):
    IPYTHON_INSTALLED = False

if IPYTHON_INSTALLED:
    import shutil
    from pathlib import Path
    from typing import Optional
    from types import ModuleType
    from dataclasses import is_dataclass, asdict
    from IPython import InteractiveShell
    from IPython.core.interactiveshell import ExecutionResult
    from IPython.core.magic import Magics, magics_class, line_cell_magic, line_magic
    from IPython.display import clear_output, display, HTML

    def import_module(name: str, shell: Optional[InteractiveShell] = None) -> Optional[ModuleType]:
        try:
            return importlib.import_module(name)
        except (ImportError, ModuleNotFoundError):
            if shell is not None:
                shell.run_line_magic('pip', f'install {name}')
                return importlib.import_module(name)
            return None

    chime = import_module('chime')              # required for chime (sound notification)
    ipywidgets = import_module('ipywidgets')    # required for tqdm (progress bar)

    @magics_class
    class Watchdog(Magics):
        def __init__(self, shell: InteractiveShell):
            super().__init__(shell)
            self.shell = shell
            self.shell.events.register('post_run_cell', self.post_run_cell)

        def post_run_cell(self, result: ExecutionResult):
            global chime
            if chime is not None:
                chime.notify("success" if result.success else "error", sync=False, raise_error=False)
            else:
                display(HTML(
                    "<b style='color: red'>"
                    "chime [pip install chime] is not installed, so no notification sound will be played."
                    "</b>"
                ))

        @line_magic
        def display_as_bullet_list(self, line: str):
            """Display a bullet list of the user's namespace."""
            line = line.strip()
            if not line:
                display(HTML(
                    "<b style='color: red'>"
                    "No variable name provided. Use %display_as_bullet_list <variable_name> to display the variable."
                    "</b>"
                ))
                return

            if line not in self.shell.user_ns:
                display(HTML(
                    "<b style='color: red'>"
                    f"Variable {line} not found in the user's namespace."
                    "</b>"
                ))
                return

            obj = self.shell.user_ns[line]
            if not isinstance(obj, (list, tuple)):
                display(HTML(
                    "<b style='color: red'>"
                    f"Variable {line} is not a list or tuple."
                    "</b>"
                ))
                return

            display(HTML(
                "<ul>" + "".join(f"<li>{item}</li>" for item in obj) + "</ul>"
            ))

        @line_magic
        def display_as_table(self, line: str):
            """Display a table of the user's namespace."""
            line = line.strip()
            if not line:
                display(HTML(
                    "<b style='color: red'>"
                    "No variable name provided. Use %display_table <variable_name> to display the variable."
                    "</b>"
                ))
                return

            if line not in self.shell.user_ns:
                display(HTML(
                    "<b style='color: red'>"
                    f"Variable {line} not found in the user's namespace."
                    "</b>"
                ))
                return

            obj = self.shell.user_ns[line]
            if isinstance(obj, dict):
                obj = [obj]

            if is_dataclass(obj):
                obj = [asdict(obj)]

            if not isinstance(obj, (list, tuple)):
                display(HTML(
                    "<b style='color: red'>"
                    f"Variable {line} is not a list or tuple."
                    "</b>"
                ))
                return

            if not all(isinstance(item, dict) for item in obj):
                display(HTML(
                    "<b style='color: red'>"
                    f"Variable {line} is not a list of dictionaries."
                    "</b>"
                ))
                return

            if len(obj) == 0:
                display(HTML(
                    "<b style='color: red'>"
                    f"Variable {line} is an empty list or tuple."
                    "</b>"
                ))
                return

            if len(obj) == 1:
                headers = []
                rows = [[str(k).replace("_", " ").title(), v] for k, v in obj[0].items()]
            else:
                headers = [str(k).replace("_", " ").title() for k in obj[0].keys()]
                rows = [
                    [item.get(k, "") for k in obj[0].keys()]
                    for item in obj
                ]

            display(HTML(
                "<table><tr>{headers_html}</tr>{rows_html}</table>".format(
                    headers_html=''.join((
                        f'<th style="text-align: left"><b>{header}</b></th>'
                        for header in headers
                    )),
                    rows_html=''.join((
                        f'<tr>{''.join((
                            f"<td style='text-align: left'>{cell}</td>"
                            for cell in row
                        ))}</tr>'
                        for row in rows
                    ))
                )
            ))

        @line_magic
        def pip_install(self, line: str):
            """Install a packages."""
            if line.startswith('install '):
                line = line.lstrip('install ')

            if not line:
                display(HTML(
                    "<b style='color: red'>"
                    "No command provided."
                    "</b>"
                ))

            try:
                self.shell.run_line_magic('pip', f"install {line}")
                clear_output()
                display(HTML(f"<b style='color: green'>Installed [{', '.join(line.split(' '))}]</b>"))
            except Exception as e:
                display(HTML(
                    f"<b style='color: red'>Error installing [{', '.join(line.split(' '))}]: "
                    f"{str(e)}</b>"
                ))

        @line_cell_magic
        def cleanup(self, line: str, cell: Optional[str] = None):
            """Garbage collect paths from the user's namespace."""
            variables_to_cleanup = {
                k for k, v in self.shell.user_ns.items()
                if isinstance(v, Path)
            }

            display(HTML(f"<b style='color: red'>Garbage collecting: [{', '.join(variables_to_cleanup)}]</b>"))
            for name in variables_to_cleanup:
                obj = self.shell.user_ns.pop(name)
                if isinstance(obj, Path) and obj.exists():
                    display(HTML(f"<b style='color: red'>Removing {obj}</b>"))
                    if obj.is_dir():
                        shutil.rmtree(obj, ignore_errors=True)
                    elif obj.is_file():
                        obj.unlink(missing_ok=True)
                    else:
                        display(HTML(f"<b style='color: red'>{obj} is neither a file nor a directory</b>"))

    def load_ipython_extension(shell: InteractiveShell):
        """Load the IPython extension."""
        global chime       # required for chime (sound notification)
        if not chime:
            chime = import_module('chime', shell)
            chime.theme('big-sur')

        global ipywidgets  # required for tqdm (progress bar)
        if not ipywidgets:
            ipywidgets = import_module('ipywidgets', shell)

        if not shell.last_execution_result or shell.last_execution_result.success:
            clear_output()

        try:
            shell.register_magics(Watchdog)
            display(HTML("<b style='color: green'>Successfully loaded langchain_memvid extension.</b>"))
        except Exception as e:
            display(HTML(f"<b style='color: red'>Error loading langchain_memvid IPython extension: {str(e)}</b>"))
            raise
