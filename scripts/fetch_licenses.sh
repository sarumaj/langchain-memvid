#!/bin/bash

# Create licenses directory if it doesn't exist
LICENSES_DIR="$(dirname "$(dirname "$0")")/licenses"
mkdir -p "$LICENSES_DIR"

# Function to download a license file
fetch_license() {
    local name=$1
    local url=$2
    local output_file="$LICENSES_DIR/$name.txt"
    
    if curl -s -f "$url" > "$output_file"; then
        echo "✓ Downloaded $name license"
    else
        echo "✗ Failed to download $name license"
    fi
}

echo "Downloading license files..."

# Download licenses
fetch_license "faiss" "https://raw.githubusercontent.com/facebookresearch/faiss/main/LICENSE"
fetch_license "opencv" "https://raw.githubusercontent.com/opencv/opencv/4.x/LICENSE"
fetch_license "langchain" "https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/langchain/LICENSE"
fetch_license "qrcode" "https://raw.githubusercontent.com/lincolnloop/python-qrcode/main/LICENSE"
fetch_license "pydantic" "https://raw.githubusercontent.com/pydantic/pydantic/main/LICENSE"
fetch_license "orjson" "https://raw.githubusercontent.com/ijl/orjson/master/LICENSE-APACHE"
fetch_license "tqdm" "https://raw.githubusercontent.com/tqdm/tqdm/master/LICENCE"
fetch_license "nest_asyncio" "https://raw.githubusercontent.com/erdewit/nest_asyncio/master/LICENSE"
fetch_license "pytest" "https://raw.githubusercontent.com/pytest-dev/pytest/main/LICENSE"
fetch_license "pytest_asyncio" "https://raw.githubusercontent.com/pytest-dev/pytest-asyncio/main/LICENSE"
fetch_license "memvid" "https://raw.githubusercontent.com/Olow304/memvid/refs/heads/main/LICENSE"

echo -e "\nDone! License files are in the 'licenses' directory." 