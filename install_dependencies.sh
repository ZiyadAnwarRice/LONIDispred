# Put caches + temp on /ddnB/work (big filesystem)
export WORKCACHE="/ddnB/work/$USER/.cache"
export UV_CACHE_DIR="$WORKCACHE/uv"
export XDG_CACHE_HOME="$WORKCACHE"
export TMPDIR="/ddnB/work/$USER/.tmp"

mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

# sanity prints
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "TMPDIR=$TMPDIR"

# OPTIONAL: clean any partial uv temp in work cache (safe)
rm -rf "$UV_CACHE_DIR/.tmp"* 2>/dev/null || true

# Now retry
cd /ddnB/work/$USER/LONIDispred   # (or wherever your pyproject.toml is)
uv sync
