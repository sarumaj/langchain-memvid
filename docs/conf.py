# Configuration file for the Sphinx documentation builder.
#
# This file reads most configuration from pyproject.toml and only contains
# essential Python code that cannot be moved to TOML.

import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# -- Path setup --------------------------------------------------------------

# Add the src directory to the Python path for autodoc
sys.path.insert(0, os.path.abspath('../src'))

# Add docs to include extensions
sys.path.insert(0, os.path.abspath('.'))

# -- Read configuration from pyproject.toml ----------------------------------

# Get the project root directory (parent of docs/)
project_root = Path(__file__).parent.parent
pyproject_path = project_root / "pyproject.toml"

# Read pyproject.toml
with open(pyproject_path, 'rb') as f:
    pyproject = tomllib.load(f)

# Get project metadata
project_metadata = pyproject.get('project', {})
sphinx_config = pyproject.get('tool', {}).get('sphinx', {})

# -- Project information -----------------------------------------------------

project = sphinx_config.get('project', project_metadata.get('name', 'Unknown Project'))
author = sphinx_config.get('author', project_metadata.get('authors', [{}])[0].get('name', 'Unknown Author'))
copyright = sphinx_config.get('copyright', f'2025, {author}')

# Use the project version from pyproject.toml
release = project_metadata.get('version', '0.0.0')

# -- General configuration ---------------------------------------------------

# Extensions
extensions = sphinx_config.get('extensions', [])

# Templates and static files
templates_path = sphinx_config.get('templates_path', ['_templates'])
html_static_path = sphinx_config.get('html_static_path', ['_static'])
for path in html_static_path:
    if not os.path.exists(path):
        os.makedirs(path)

# Exclude patterns
exclude_patterns = sphinx_config.get('exclude_patterns', ['_build', 'Thumbs.db', '.DS_Store'])

# -- Options for HTML output -------------------------------------------------

html_theme = sphinx_config.get('html_theme', 'alabaster')
html_theme_options = sphinx_config.get('html_theme_options', {})

# -- Extension configuration -------------------------------------------------

autodoc_default_options = sphinx_config.get('autodoc_default_options', {})
autodoc_typehints = sphinx_config.get('autodoc_typehints', 'description')
autodoc_typehints_format = sphinx_config.get('autodoc_typehints_format', 'short')

# Build intersphinx_mapping from arrays, adding None as second element if missing
intersphinx_mapping = {
    key: (value[0], value[1] if len(value) > 1 else None)
    for key, value in sphinx_config.get('intersphinx_mapping', {}).items()
    if isinstance(value, list)
}

# Napoleon settings
globals().update({
    key: value for key, value in sphinx_config.items()
    if key.startswith('napoleon_')
})

# MyST settings
myst_enable_extensions = sphinx_config.get('myst_enable_extensions', [])
