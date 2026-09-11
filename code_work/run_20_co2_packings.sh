#!/usr/bin/env bash
set -euo pipefail

# Sequentially generate the 20 prescribed zeolite 13X packings and then
# convert every final particle/contact dump to ParaView VTP.
#
# Optional environment variables:
#   LMP_BIN=/path/to/lmp      LAMMPS executable (default: lmp)
#   NPROCS=1                  MPI processes per case (default: 1)
#   MPI_LAUNCHER=mpirun       MPI launcher (default: mpirun)
#
# Examples:
#   ./run_20_co2_packings.sh
#   LMP_BIN=$HOME/lammps/build/lmp NPROCS=4 ./run_20_co2_packings.sh

input_file="in.generate_co2_13x_packed_bed.lammps"
converter="convert_co2_packings_to_vtp.py"
lmp_bin="${LMP_BIN:-lmp}"
nprocs="${NPROCS:-1}"
mpi_launcher="${MPI_LAUNCHER:-mpirun}"
log_dir="batch_logs_co2_packings"

seeds=(
    18427 27183 31991 40213 47837
    52691 61417 69371 74167 81929
    90709 98251 105019 113117 121021
    129121 137077 145093 153133 161123
)

[[ -f "$input_file" ]] || { echo "Missing $input_file" >&2; exit 2; }
[[ -f "$converter" ]] || { echo "Missing $converter" >&2; exit 2; }
command -v "$lmp_bin" >/dev/null 2>&1 || { echo "LAMMPS executable not found: $lmp_bin" >&2; exit 2; }
[[ "$nprocs" =~ ^[1-9][0-9]*$ ]] || { echo "NPROCS must be a positive integer" >&2; exit 2; }
if (( nprocs > 1 )); then
    command -v "$mpi_launcher" >/dev/null 2>&1 || { echo "MPI launcher not found: $mpi_launcher" >&2; exit 2; }
fi

mkdir -p "$log_dir"

for index in "${!seeds[@]}"; do
    seed="${seeds[$index]}"
    case_dir="co2_13x_seed_${seed}"
    log_file="${log_dir}/seed_${seed}.log"
    echo "[$((index + 1))/${#seeds[@]}] Running seed $seed"
    if [[ -e "$case_dir" ]]; then
        echo "Refusing to overwrite existing case directory: $case_dir" >&2
        echo "Run ./clean_co2_packing_results.sh --yes for a clean campaign." >&2
        exit 3
    fi

    if (( nprocs == 1 )); then
        "$lmp_bin" -var seed_realization "$seed" -in "$input_file" 2>&1 | tee "$log_file"
    else
        "$mpi_launcher" -np "$nprocs" "$lmp_bin" \
            -var seed_realization "$seed" -in "$input_file" 2>&1 | tee "$log_file"
    fi

    summary="${case_dir}/packing_summary.dat"
    [[ -s "$summary" ]] || { echo "Missing case summary: $summary" >&2; exit 4; }
    nfinal="$(awk 'NR==2 {print $3}' "$summary")"
    [[ "$nfinal" == "1850" ]] || { echo "Seed $seed inserted only $nfinal particles" >&2; exit 4; }
done

echo "All 20 LAMMPS cases completed. Converting final dumps to ParaView VTP."
python3 "$converter" --root . --output paraview_co2_packings
echo "Campaign complete. Open files below paraview_co2_packings/ in ParaView."
