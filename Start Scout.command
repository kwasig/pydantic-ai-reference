#!/usr/bin/env bash
# Double-click in Finder: installs, verifies against Token Factory, starts the Studio and opens it in your browser.
cd "$(dirname "$0")"
exec bash scripts/run_live.sh
