#!/usr/bin/env bash
set -euo pipefail

NETWORK_ROOT="co2_pore_networks_power22"
OUTPUT_ROOT="co2_adsorption_campaign_5000s"
SOLVER="solve_co2_pore_adsorption.py"

mkdir -p "$OUTPUT_ROOT"

mapfile -t NETWORKS < <(
    find "$NETWORK_ROOT" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name 'seed_*' \
        | sort -V
)

if [[ "${#NETWORKS[@]}" -ne 20 ]]; then
    echo "ERROR: expected 20 networks but found ${#NETWORKS[@]}"
    printf '%s\n' "${NETWORKS[@]}"
    exit 1
fi

for NETWORK in "${NETWORKS[@]}"; do
    SEED_NAME="$(basename "$NETWORK")"
    OUTPUT="$OUTPUT_ROOT/$SEED_NAME"
    LOG="$OUTPUT_ROOT/${SEED_NAME}.log"

    if [[ -f "$OUTPUT/timeseries.csv" &&
          -f "$OUTPUT/final_state.npz" &&
          -f "$OUTPUT/run_metadata.json" ]]; then
        echo "SKIP $SEED_NAME: completed output already exists"
        continue
    fi

    echo "START $SEED_NAME: $(date --iso-8601=seconds)"

    python3 "$SOLVER" \
        --network "$NETWORK" \
        --output "$OUTPUT" \
        --mode adsorption \
        --diffusivity 1.5e-5 \
        --inlet-concentration 6.05 \
        --initial-concentration 0.0 \
        --temperature 298.15 \
        --particle-density 1600 \
        --qsat 5.332 \
        --b-pa 5.093e-5 \
        --toth-exponent 1.0 \
        --k-ldf 8.55e-3 \
        --initial-loading 0.0 \
        --t-end 5000 \
        --outputs 301 \
        --rtol 1e-6 \
        --atol 1e-10 \
        >"$LOG" 2>&1

    echo "DONE  $SEED_NAME: $(date --iso-8601=seconds)"
done

echo "All 20 adsorption cases completed."
