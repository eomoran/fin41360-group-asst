# Project-local cache paths for matplotlib/fontconfig in restricted environments.
# Usage: source ./env.workspace.sh

export MPLCONFIGDIR="$(pwd)/.cache/matplotlib"
export XDG_CACHE_HOME="$(pwd)/.cache"
