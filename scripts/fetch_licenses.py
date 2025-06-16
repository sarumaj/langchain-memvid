#!/usr/bin/env python3
import requests
from pathlib import Path

# Create licenses directory if it doesn't exist
LICENSES_DIR = Path(__file__).parent.parent / "licenses"
LICENSES_DIR.mkdir(exist_ok=True)

# Dictionary of dependencies and their license URLs
LICENSE_URLS = {
    "faiss": "https://raw.githubusercontent.com/facebookresearch/faiss/main/LICENSE",
    "opencv": "https://raw.githubusercontent.com/opencv/opencv/4.x/LICENSE",
    "langchain": "https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/langchain/LICENSE",
    "qrcode": "https://raw.githubusercontent.com/lincolnloop/python-qrcode/main/LICENSE",
    "pydantic": "https://raw.githubusercontent.com/pydantic/pydantic/main/LICENSE",
    "orjson": "https://raw.githubusercontent.com/ijl/orjson/master/LICENSE-APACHE",
    "msgpack": "https://raw.githubusercontent.com/msgpack/msgpack-python/main/COPYING",
    "msgpack-numpy": "https://raw.githubusercontent.com/lebedov/msgpack-numpy/master/LICENSE.md",
    "tqdm": "https://raw.githubusercontent.com/tqdm/tqdm/master/LICENCE",
    "pypdf": "https://raw.githubusercontent.com/py-pdf/pypdf/main/LICENSE",
    "ebooklib": "https://raw.githubusercontent.com/aerkalov/ebooklib/master/LICENSE.txt",
    "nest_asyncio": "https://raw.githubusercontent.com/erdewit/nest_asyncio/master/LICENSE",
    "pytest": "https://raw.githubusercontent.com/pytest-dev/pytest/main/LICENSE",
    "beautifulsoup4": "https://raw.githubusercontent.com/akalongman/python-beautifulsoup/master/LICENSE",
    "memvid": "https://raw.githubusercontent.com/Olow304/memvid/refs/heads/main/LICENSE"
}


def fetch_license(name: str, url: str) -> None:
    """Fetch license from URL and save to file."""
    try:
        response = requests.get(url)
        response.raise_for_status()

        license_file = LICENSES_DIR / f"{name}.txt"
        license_file.write_text(response.text)
        print(f"✓ Downloaded {name} license")
    except Exception as e:
        print(f"✗ Failed to download {name} license: {e}")


def main():
    """Download all license files."""
    print("Downloading license files...")
    for name, url in LICENSE_URLS.items():
        fetch_license(name, url)
    print("\nDone! License files are in the 'licenses' directory.")


if __name__ == "__main__":
    main()
