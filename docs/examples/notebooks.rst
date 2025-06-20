Jupyter Notebook Examples
=========================

Interactive Jupyter notebooks provide hands-on examples with detailed explanations and visual outputs.

Quick Start Notebook
--------------------

.. _quickstart-notebook:

This notebook demonstrates the basic usage of the LangChain MemVid library:

* Setting up the environment and dependencies
* Creating a vector store with sample data
* Performing similarity searches
* Understanding the results and metadata

.. only:: builder_html

    **Download**: :download:`examples/quickstart.ipynb <../../examples/quickstart.ipynb>`

    **Generated Python**: :download:`examples/quickstart.py <../../examples/quickstart.py>`

Advanced Usage Notebook
-----------------------

.. _advanced-notebook:

This notebook covers advanced features and customization options:

* Working with individual components (Encoder, IndexManager, etc.)
* Custom video and QR code configurations
* Direct video processing and manipulation
* Building complete systems from components
* Testing and verification of functionality

.. only:: builder_html

    **Download**: :download:`examples/advanced.ipynb <../../examples/advanced.ipynb>`

    **Generated Python**: :download:`examples/advanced.py <../../examples/advanced.py>`

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