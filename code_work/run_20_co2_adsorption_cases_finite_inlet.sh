#!/usr/bin/env bash
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

    if python3 - "$result" <<'CHECK'
import csv, json, sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1])
try:
    meta=json.loads((p/'run_metadata.json').read_text())
    if not meta.get('solver_success') or not meta.get('physical_bounds_ok'):
        raise ValueError('unsuccessful run')
    expected=dict(mode='adsorption',diffusivity=1.5e-5,inlet_concentration=6.05,
        initial_concentration=0.,temperature=298.15,particle_density=1600.,qsat=5.332,
        b_pa=5.093e-5,toth_exponent=1.,k_ldf=8.55e-3,initial_loading=0.,
        inlet_boundary='finite',tube_radius=.008,bed_bottom=0.,inlet_min_distance_dp=.5,
        inlet_area_weighting='equal',t_end=5000.,outputs=301,rtol=1e-6,atol=1e-10)
    if any(meta['parameters'].get(k)!=v for k,v in expected.items()):
        print(f'Conflicting settings in {p}; choose a new output directory.',file=sys.stderr)
        sys.exit(2)
    with np.load(p/'final_state.npz') as z:
        if float(z['time_s'])!=5000. or not np.isfinite(z['pore_concentration']).all():
            raise ValueError('incomplete final state')
    with (p/'timeseries.csv').open() as f: rows=list(csv.DictReader(f))
    if not rows or float(rows[-1]['time_s'])!=5000.: raise ValueError('incomplete timeseries')
except (OSError,ValueError,KeyError):
    sys.exit(1)
CHECK
    then
        echo "SKIP completed seed $seed"
        continue
    else
        status=$?
        if [[ "$status" -eq 2 ]]; then exit 2; fi
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
