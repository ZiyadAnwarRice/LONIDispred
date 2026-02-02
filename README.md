# One-time: install uv (skip if uv already works)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is on PATH
export PATH="$HOME/.local/bin:$PATH"

# Move uv cache to work (avoid /home quota)
export UV_CACHE_DIR="/ddnB/work/$USER/.cache/uv"
mkdir -p "$UV_CACHE_DIR"
uv cache dir

# Go to your project
cd /ddnB/work/$USER/Research/LORADispred

# Install and pin Python 3.11.5 (managed by uv)
uv python install 3.11.5
uv python pin 3.11.5

# Initialize project files (only if pyproject.toml does NOT exist yet)
# If pyproject.toml already exists, SKIP this line.
uv init

# Configure torch cu121 index in pyproject.toml ----
# Add/ensure this is in pyproject.toml:
#
# [project]
# requires-python = ">=3.11.5,<3.12"
#
# [[tool.uv.index]]
# name = "pytorch-cu121"
# url = "https://download.pytorch.org/whl/cu121"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu121" }
# torchvision = { index = "pytorch-cu121" }
# torchaudio = { index = "pytorch-cu121" }

# Import requirements into the project
uv add -r requirements.txt --frozen

# Lock + sync (creates/uses .venv)
uv lock
uv sync

# Sanity check
which python
python --version
python -c "import torch; print(torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available())"
