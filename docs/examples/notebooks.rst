Jupyter Notebook Examples
=========================

Interactive Jupyter notebooks provide hands-on examples with detailed explanations and visual outputs.

Quick Start Notebook
-------------------

.. _quickstart-notebook:

**File**: `quickstart.ipynb <../../examples/quickstart.ipynb>`_

This notebook demonstrates the basic usage of the LangChain MemVid library:

* Setting up the environment and dependencies
* Creating a vector store with sample data
* Performing similarity searches
* Understanding the results and metadata

**Download**: `examples/quickstart.ipynb <../../examples/quickstart.ipynb>`_

**Generated Python**: `examples/quickstart.py <../../examples/quickstart.py>`_

Advanced Usage Notebook
-----------------------

.. _advanced-notebook:

**File**: `advanced.ipynb <../../examples/advanced.ipynb>`_

This notebook covers advanced features and customization options:

* Working with individual components (Encoder, IndexManager, etc.)
* Custom video and QR code configurations
* Direct video processing and manipulation
* Building complete systems from components
* Testing and verification of functionality

**Download**: `examples/advanced.ipynb <../../examples/advanced.ipynb>`_

**Generated Python**: `examples/advanced.py <../../examples/advanced.py>`_

Running the Notebooks
---------------------

To run these notebooks:

1. Install the development dependencies:

   .. code-block:: bash
      
      pip install -e ".[test]"

2. Start Jupyter:

   .. code-block:: bash
      
      jupyter notebook

3. Navigate to the `examples/` directory and open the desired notebook

4. Run all cells to see the examples in action

Note: The Python files (`.py`) are automatically generated when the notebooks are executed. 