#!/bin/sh
# Install the systemd user units for fgc-matchwatch. Leaves the timer DISABLED.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
exe="$here/.venv/bin/fgc-matchwatch"
[ -x "$exe" ] || { echo "no $exe; create the venv first (see README)" >&2; exit 1; }
dest="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$dest"
sed "s#@EXE@#$exe#" "$here/systemd/fgc-matchwatch.service" > "$dest/fgc-matchwatch.service"
cp "$here/systemd/fgc-matchwatch.timer" "$dest/fgc-matchwatch.timer"
systemctl --user daemon-reload
echo "installed to $dest (timer NOT enabled)." >&2
echo "enable for the event:  systemctl --user enable --now fgc-matchwatch.timer" >&2
