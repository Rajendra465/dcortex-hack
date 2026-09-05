#!/usr/bin/env bash
set -e
PORT=${1:-8787}
echo "========================================================"
echo "  dCortex Airline OCC Decision Command Center"
echo "  CAR Section 7 Series J Legality & Recovery Engine"
echo "========================================================"
echo "Starting local OCC server at http://127.0.0.1:${PORT}"
echo "Press Ctrl+C to stop."
echo "========================================================"
python3 -m crewops serve --port "${PORT}"
