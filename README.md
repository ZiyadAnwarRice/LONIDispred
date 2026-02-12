## One-time: install uv (skip if uv already works)

command -v uv >/dev/null 2>\&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

## Ensure uv is on PATH

export PATH="$HOME/.local/bin:$PATH"

## Move uv cache to work (avoid /home quota)

export UV\_CACHE\_DIR="/ddnB/work/$USER/.cache/uv"
mkdir -p "$UV\_CACHE\_DIR"

## Go to your project

cd /ddnB/work/$USER/Research/LORADispred

## Install and pin Python 3.11.5 (managed by uv)

uv python install 3.11.5
uv python pin 3.11.5

## Initialize project files (only if pyproject.toml does NOT exist yet)

## If pyproject.toml already exists, SKIP this line.

test -f pyproject.toml || uv init

## Lock + sync (creates/uses .venv)

uv lock
uv sync

## Sanity check (use the project venv)

uv run python --version
uv run python -c "import torch; print('torch:', torch.**version**); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is\_available())"





watchlastgpu () {

&nbsp; jid=$(squeue -u "$USER" -h -o "%i" | head -n 1)

&nbsp; \[ -z "$jid" ] \&\& echo "No running jobs." \&\& return 1

&nbsp; node=$(squeue -j "$jid" -h -o "%N")

&nbsp; echo "Watching JOBID=$jid on NODE=$node ..."

&nbsp; srun --jobid "$jid" -w "$node" --pty bash -lc 'nvidia-smi dmon -s pucm'

}

watchlastgpu



