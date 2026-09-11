set -euo pipefail

OUTPUT_ROOT="co2_adsorption_campaign_finite_inlet_5000s"
NETWORK_ROOT="co2_pore_networks_power22"

mkdir -p "$OUTPUT_ROOT/logs"

for seed in \
  18427 27183 31991 40213 47837 \
  52691 61417 69371 74167 81929 \
  90709 98251 105019 113117 121021 \
  129121 137077 145093 153133 161123
do
    result="$OUTPUT_ROOT/seed_$seed"
    log="$OUTPUT_ROOT/logs/seed_$seed.log"

    if [[ -f "$result/run_metadata.json" ]]; then
        echo "SKIP completed seed $seed"
        continue
    fi

    echo "RUN seed $seed"

    python3 solve_co2_pore_adsorption.py \
      --network "$NETWORK_ROOT/seed_$seed" \
      --output "$result" \
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
      --inlet-boundary finite \
      --tube-radius 0.008 \
      --bed-bottom 0.0 \
      --inlet-min-distance-dp 0.5 \
      --inlet-area-weighting equal \
      --t-end 5000 \
      --outputs 301 \
      --rtol 1e-6 \
      --atol 1e-10 \
      > "$log" 2>&1

    echo "DONE seed $seed"
done

echo "All finite-inlet cases completed."
