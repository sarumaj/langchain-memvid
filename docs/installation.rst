Installation
============

Requirements
------------

- Python 3.12 or higher
- FFmpeg (for video processing)

Installing FFmpeg
-----------------

**Ubuntu/Debian:**

.. code:: bash

   sudo apt update
   sudo apt install ffmpeg

**macOS (using Homebrew):**

.. code:: bash

   brew install ffmpeg

**Windows:**
Download from `FFmpeg official website <https://ffmpeg.org/download.html>`_ or install via Chocolatey:

.. code:: bash

   choco install ffmpeg

Installing langchain-memvid
---------------------------

Install from PyPI:

.. code:: bash

   pip install langchain-memvid

Install with development dependencies:

.. code:: bash

   pip install langchain-memvid[test]

Install from source:

.. code:: bash

   git clone https://github.com/sarumaj/langchain-memvid.git
   cd langchain-memvid
   pip install -e . 