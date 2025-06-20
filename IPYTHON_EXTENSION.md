# IPython Extension for LangChain MemVid

This package includes an optional IPython extension that provides magic commands and utilities for working with LangChain MemVid in Jupyter notebooks and IPython environments.

## Installation

To install the package with IPython extension support:

```bash
pip install langchain-memvid jupyter ipython
```

## Usage

### Method 1: Load extension in a notebook

In any Jupyter notebook or IPython session:

```python
%load_ext ipykernel_memvid_extension
```

### Method 2: Auto-load extension

You can configure IPython to automatically load the extension by adding this to your IPython profile:

```python
# In ~/.ipython/profile_default/ipython_config.py
c.InteractiveShellApp.extensions = ['ipykernel_memvid_extension']
```

## Available Magic Commands

Once loaded, the extension provides several magic commands:

### `%as_bullet_list`
Display an iterable as a bullet list:

```python
my_list = ["item1", "item2", "item3"]
%as_bullet_list my_list
```

### `%as_table`
Display a list of dictionaries as a table:

```python
data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
%as_table data
```

### `%cleanup`
Clean up Path objects from the namespace:

```python
from pathlib import Path
temp_file = Path("temp.txt")
temp_file.write_text("hello")
# ... use the file ...
%cleanup  # This will delete temp_file
%cleanup -f  # Force cleanup without confirmation
```

### `%pip_install`
Install packages with visual feedback:

```python
%pip_install pandas numpy
%pip_install requests
```

### `%restart_kernel`
Restart the Jupyter kernel:

```python
%restart_kernel
%restart_kernel -f  # Force restart without confirmation
```

### `%list_sound_themes`
List available sound themes:

```python
%list_sound_themes
```

### `%set_sound_theme`
Set a sound theme on completion of cell execution:

```python
%set_sound_theme zelda
```

### `%mute`
Disable sound notifications:

```python
%mute
```

### `%unmute`
Enable sound notifications:

```python
%unmute
```

### `%dump`
Dump notebook content to a Python file with markdown cells as comments and omit macro calls:

```python
%dump                               # Auto-discover notebook name
%dump my_notebook                   # Dump all cells
%dump -f my_notebook                # Force overwrite existing file
%dump -r 1:5 my_notebook            # Dump only cells 1-5
%dump -r 1,3,5 my_notebook          # Dump only cells 1, 3, and 5
%dump -r :5 my_notebook             # Dump cells 1-5 (from beginning)
%dump -r 1: my_notebook             # Dump cells 1 to end
%dump -r 1:-2 my_notebook           # Dump cells 1 to end but last
%dump -o my_notebook                # Dump cells with their outputs (html will be converted to markdown comments)
```

This command reads a `.ipynb` file and converts it to a `.py` file, filtering out magic commands and converting markdown cells to comments. The range specification follows IPython conventions:
- **No range specified**: Dumps all cells
- Single indices: `1`, `3`, `5`
- Ranges: `1-5`, `-5` (from beginning), `1-` (to end)
- Mixed: `1,3,5` (comma-separated)
- Duplicate indices are automatically deduplicated

## Features

- **Sound notifications**: Plays a sound when cells complete (requires `chime`)
- **Visual feedback**: Color-coded messages for success, warnings, and errors
- **Automatic cleanup**: Helps manage temporary files and paths
- **Progress bars**: Enhanced progress tracking with `tqdm`
- **Table formatting**: Pretty display of data structures


## Development

To develop or modify the extension:

1. Install in development mode:
   ```bash
   pip install -e .
   ```

2. The extension file is located at `src/ipykernel_memvid_extension.py`

3. Reload the extension after changes:
   ```python
   %reload_ext ipykernel_memvid_extension
   ``` 