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
```

### `%pip_install`
Install packages with visual feedback:

```python
%pip_install pandas numpy
```

### `%restart_kernel`
Restart the Jupyter kernel:

```python
%restart_kernel
```

### `%list_sound_themes`
List available sound themes:

```python
%list_sound_themes
```

### `%set_sound_theme`
Set a sound theme on completion of cell execution:

```python
%set_sound_themes zelda
```

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