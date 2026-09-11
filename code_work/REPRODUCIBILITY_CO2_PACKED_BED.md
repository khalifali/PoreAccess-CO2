# Beginner's guide and reproducibility record: zeolite 13X packed-bed CO2 adsorption

## 1. The research question

This project asks whether two packed beds made from the same zeolite beads can
adsorb CO2 at different rates because their internal void spaces are connected
differently, even when their overall porosities are almost equal.

There is no CFD. The workflow has three physical stages:

1. LAMMPS DEM generates mechanically realistic beds of spherical beads.
2. A pore network represents the empty space as pore nodes connected by throat
   edges. CO2 diffuses through this graph.
3. Each zeolite bead has its own adsorbed loading. Gas removed from surrounding
   pores is added to that bead.

The present study uses transparent physical descriptors instead of a graph
neural network. This keeps the result easy to interpret and is appropriate for
the current ensemble of only 20 beds.

## 2. Physical problem and limitations

The bed initially contains no CO2. Its bottom is coupled to an external CO2
reservoir through a finite diffusive boundary. CO2 moves upward through
stagnant gas by diffusion and is adsorbed by zeolite beads.

Boundary and initial conditions:

- bottom: finite diffusive coupling to a reservoir at 6.05 mol/m3;
- top and cylindrical wall: closed, with no external CO2 flux;
- internal gas concentration at the start: zero in every pore;
- initial particle loading: zero;
- temperature: constant.

This is not a flowing-column breakthrough simulation. It contains no gas
velocity, CFD, pressure drop, adsorption heat, humidity, competitive adsorption
or particle deformation. Results describe transient one-sided
diffusion--adsorption, not complete industrial capture performance.

## 3. Software and scripts

LAMMPS stable 22 Jul 2025 Update 5 with the GRANULAR package was used. Python
post-processing requires NumPy and SciPy; Matplotlib is used for porosity plots.

| File | Purpose |
|---|---|
| 'in.generate_co2_13x_packed_bed.lammps' | Generate one gravity-filled packing |
| 'run_20_co2_packings.sh' | Generate all 20 DEM realizations |
| 'clean_co2_packing_results.sh' | Safely preview or delete generated DEM outputs |
| 'summarize_co2_packing_campaign.py' | Check completeness, settling and overlaps |
| 'convert_co2_packings_to_vtp.py' | Create ParaView particle/contact files |
| 'analyze_co2_packing_porosity.py' | Calculate bulk, interior, axial and radial porosity |
| 'extract_co2_pore_network.py' | Convert particle geometry into pores and throats |
| 'solve_co2_pore_adsorption.py' | Solve diffusion alone or diffusion with adsorption |
| 'summarize_co2_adsorption_campaign.py' | Check the 20 adsorption runs and extract response targets |
| 'extract_co2_inlet_accessibility.py' | Calculate physical inlet-to-interior accessibility descriptors |
| 'analyze_co2_structure_performance.py' | Compare low-dimensional structure models using leave-one-bed-out validation |
| 'solve_co2_homogeneous_1d.py' | Solve or fit the conservative one-dimensional homogeneous bed model |
| 'fit_co2_homogeneous_campaign.py' | Fit the verified homogeneous model sequentially to all accepted beds |
| 'analyze_co2_effective_diffusivity_closure.py' | Relate fitted homogeneous diffusivity to pore-network accessibility and test the closure |

The prescribed random seeds are:

~~~text
18427 27183 31991 40213 47837
52691 61417 69371 74167 81929
90709 98251 105019 113117 121021
129121 137077 145093 153133 161123
~~~

Only the seed changes. All other settings are identical.

## 4. DEM packing generation

Run one case with:

~~~bash
lmp -var seed_realization 18427 \
    -in in.generate_co2_13x_packed_bed.lammps
~~~

'seed_realization' controls the random insertion sequence and output name, such
as 'co2_13x_seed_18427'.

### 4.1 Geometry and particle parameters

| Parameter | Value | Explanation |
|---|---:|---|
| 'Ntarget' | 1850 | Required bead count |
| 'dp' | 1.60e-3 m | Bead diameter |
| 'density' | 1600 kg/m3 | Apparent bead density used for mass |
| 'Dtube' | 16.0e-3 m | Tube inner diameter |
| 'Rtube' | 8.0e-3 m | Tube inner radius |
| 'Hbox' | 70.0e-3 m | Tube plus empty headspace |
| 'Rinsert' | 'Rtube - dp/2' | Keeps new bead centres away from the wall |
| 'zPourLo', 'zPourHi' | 38, 58 mm | Vertical insertion region |

The tube diameter equals ten bead diameters. Wall effects are therefore part of
the studied geometry. The density is an apparent bead density, not crystalline
skeletal density.

### 4.2 Contact and time parameters

| Parameter | Value | Explanation |
|---|---:|---|
| 'young' | 1.0e8 Pa | Softened numerical contact stiffness |
| 'poisson' | 0.25 | Poisson ratio used by Hertz contact |
| 'restitution' | 0.30 | Collision rebound level |
| 'friction' | 0.40 | Tangential friction coefficient |
| gravity | 9.81 m/s2 downward | Fills and settles the bed |
| timestep | 2.0e-7 s | Time represented by one DEM step |
| pouring damping | 1.0e-4 | Mild viscous damping during filling |
| final damping | 5.0e-4 | Stronger damping during relaxation |
| 'nPour' | 3,000,000 | Filling steps |
| 'nRelax' | 4,000,000 | Final settling steps |

The real zeolite modulus is not claimed. The softer value makes the simulation
affordable and is accepted only because normalized overlaps are checked. The
contact law is Hertz normal contact, Mindlin tangential friction and Tsuji
damping. Because the tube is a region, its correct command is
'fix wall/gran/region'.

### 4.3 DEM outputs

Each seed directory contains:

- 'particles_final.dump': final bead positions, radii, velocities and wall data;
- 'contacts_final.dump': interacting bead pairs;
- 'packing_final.data' and 'packing_final.restart': reusable LAMMPS states;
- 'packing_summary.dat': particle count, height, porosity and settling data;
- 'filling_*.dump': intermediate filling snapshots.

LAMMPS calculates mechanics only. Adsorption is added later in Python.

The early command 'shell mkdir -p ...' created a directory literally named
'-p' because LAMMPS did not interpret the Unix option. The correct LAMMPS
command is 'shell mkdir output_directory'.

## 5. Running and checking 20 DEM cases

Run:

~~~bash
./run_20_co2_packings.sh
~~~

Optional environment settings:

| Setting | Default | Meaning |
|---|---|---|
| 'LMP_BIN' | 'lmp' | LAMMPS executable |
| 'NPROCS' | 1 | MPI processes per case |
| 'MPI_LAUNCHER' | 'mpirun' | MPI launcher |

The runner refuses to overwrite existing cases, verifies 1850 particles, writes
logs below 'batch_logs_co2_packings', creates the quality report and converts
the packings for ParaView.

Preview cleanup with './clean_co2_packing_results.sh'. Actual deletion requires
'./clean_co2_packing_results.sh --yes'. This two-step design prevents accidental
removal.

### 5.1 Packing-summary options

~~~bash
python3 summarize_co2_packing_campaign.py
~~~

| Option | Default | Explanation |
|---|---:|---|
| '--root' | current directory | Where cases are searched |
| '--output-prefix' | 'co2_packing_campaign_summary' | Report filename prefix |
| '--target-particles' | 1850 | Required bead count |
| '--particle-diameter' | 1.6e-3 m | Used to normalize overlap |
| '--tube-radius' | 8e-3 m | Tube geometry check |
| '--max-speed' | 1e-4 m/s | Settling limit |
| '--max-ke-per-particle' | 1e-15 J | Kinetic-energy settling limit |
| '--max-normalized-overlap' | 1.25e-3 | Overlap divided by bead diameter |
| '--porosity-match-tolerance' | 0.005 | Diagnostic distance from median porosity |
| '--seeds' | prescribed 20 | Cases to check |

'analysis_ready' requires complete outputs, correct particle count, a settled
bed, safe neighbor lists and acceptable overlaps. 'porosity_matched' is only an
ensemble comparison and is not a mechanical acceptance rule.

All 20 cases were complete, settled and analysis-ready. Mean preliminary
porosity was 0.460076, mean bed height 36.550 mm, mean coordination 4.430,
largest final speed 6.87e-5 m/s and largest normalized overlap 1.174e-3.

## 6. ParaView conversion

~~~bash
python3 convert_co2_packings_to_vtp.py \
    --root . \
    --output paraview_co2_packings
~~~

'--root' selects the search directory. '--output' selects the VTP result
directory. The script writes particle points and contact lines without needing
the Python VTK package.

In ParaView, open 'particles_final.vtp', apply Glyph, choose Sphere, use
'diameter' as Scale Array, set Scale Factor to 1 and select All Points.
'contacts_final.vtp' displays interacting-particle connections.

## 7. Robust porosity analysis

A single high bead can make the maximum bed height misleading. The robust bed
top is therefore the 99th percentile of all particle-top elevations. The script
also integrates the actual fraction of every sphere crossing a bin boundary
instead of assigning its complete volume to one bin.

~~~bash
python3 analyze_co2_packing_porosity.py \
    --root . --tube-radius 0.008 --bottom 0.0 \
    --top-percentile 99 --axial-bins 30 --radial-bins 10
~~~

| Option | Default | Explanation |
|---|---:|---|
| '--root' | current directory | Search root |
| '--pattern' | 'particles_final*.dump' | Input filename pattern |
| '--tube-radius' | 0.008 m | Analysis-cylinder radius |
| '--bottom' | 0 m | Lower bed boundary |
| '--axial-bins' | 30 | Horizontal slices |
| '--radial-bins' | 10 | Radial annuli |
| '--top-percentile' | 99 | Robust bed-top definition |
| three '--interior-...-exclusion-dp' options | 1.0 each | Bead diameters removed near wall, bottom and top |
| '--quadrature-order' | 48 | Sphere-intersection integration accuracy |
| '--output-dir' | 'co2_porosity_analysis' | Output directory |
| '--no-plots' | off | Disables plotting if supplied |

Porosity equals one minus solid volume divided by analysis volume. Bulk
porosity uses the complete tube. Interior porosity removes one bead diameter
near the wall, bottom and top and describes the more homogeneous core.

Mean robust bulk porosity was 0.446649, mean interior porosity 0.410040 and mean
robust bed height 35.609 mm. Their small coefficients of variation, about
0.55%, show that the 20 packings have closely matched global properties. All
20 were retained.

## 8. Pore-network extraction

### 8.1 Meaning

The empty space is replaced by pore nodes that store gas and throat edges that
carry diffusion. Four neighboring bead centres define a Delaunay tetrahedron;
its circumcentre becomes a pore. Tetrahedra sharing a triangular face become
connected by a throat. The particle--pore incidence file records which beads
surround each pore.

Ordinary and radical Delaunay topology are identical for equal bead radii. The
script rejects polydisperse packings because they require weighted
triangulation.

### 8.2 Commands

Test one packing:

~~~bash
python3 extract_co2_pore_network.py \
    --case co2_13x_seed_18427 \
    --output-root co2_pore_networks_power22 \
    --tube-radius 0.008 --bottom 0 \
    --bed-top-percentile 99 --sobol-power 22
~~~

Extract all packings:

~~~bash
python3 extract_co2_pore_network.py \
    --all-cases --root . \
    --output-root co2_pore_networks_power22 \
    --tube-radius 0.008 --bottom 0 \
    --bed-top-percentile 99 --sobol-power 22
~~~

### 8.3 All extraction parameters

| Option | Default | Production value and meaning |
|---|---:|---|
| '--case' / '--all-cases' | one is required | One case or complete campaign |
| '--root' | current directory | Search root |
| '--pattern' | 'particles_final*.dump' | Input dump pattern |
| '--output-root' | 'co2_pore_networks' | 'co2_pore_networks_power22' |
| '--tube-radius' | 0.008 m | Tube radius |
| '--bottom' | 0 m | Lower boundary |
| '--bed-top-percentile' | 99 | Robust upper boundary |
| '--boundary-layer-dp' | 1.25 | Boundary-label thickness in bead diameters |
| '--sobol-power' | 20 | 22; sample count equals 2 raised to this power |
| '--sobol-seed' | 18427 | Sampling scramble seed if case seed is unavailable |
| '--quadrature-order' | 48 | Solid-volume integration accuracy |
| '--radius-tolerance' | 1e-6 | Allowed relative radius spread |
| '--domain-tolerance' | 2e-6 m | Pore acceptance tolerance near domain boundary |
| '--clearance-tolerance' | 2e-6 m | Tolerance for pore clearance from particles |
| '--min-throat-radius' | 1e-9 m | Numerical throat-radius floor |
| '--min-throat-length' | 1e-12 m | Rejects zero-length throats |
| '--min-largest-component-fraction' | 0.98 | Required connected fraction |

The total void volume is cylinder volume minus integrated solid volume. Sobol
points divide this exact total among pore nodes. Power 22 means 4,194,304
samples. Testing powers 18, 20 and 22 showed that topology and throat geometry
were unchanged; only local volume allocation changed. Pores with at most five
void samples contained only 0.003621% of total void volume.

Each network directory contains 'pores.csv', 'throats.csv',
'particle_pore_incidence.csv', 'pore_network.npz', 'pore_network.vtp' and
'network_qa.json'. 'campaign_network_qa.json' summarizes all cases.

Transport readiness requires finite geometry, positive volumes and lengths,
unique throats, inlet/outlet labels, inlet-to-outlet connectivity, at least 98%
of pores in the largest component and exact total-volume conservation.

Seed 18427 contained 9397 pores and 18104 throats, one connected component and
passed all checks. Its robust porosity was 0.449938. The network was visually
overlaid with particles in ParaView before production extraction.

## 9. Diffusion--adsorption solver

### 9.1 Diffusion equation

Pore p stores concentration 'c_p' and volume 'V_p'. For a throat with area A
and length L, conductance is:

~~~text
G = D A / L.
~~~

The flow contribution is 'G (c_neighbor - c_p)'. Larger D or A increases
transport; larger L reduces it. No velocity is solved.

### 9.2 Adsorption equilibrium and rate

The local ideal-gas pressure is 'P = c R T', with
'R = 8.314462618 J/(mol K)'.

Toth equilibrium is:

~~~text
q* = q_s b P / [1 + (b P)^t]^(1/t).
~~~

'q_s' is saturation loading, 'b' is affinity and 't' describes heterogeneity.
Here 't = 1', giving Langmuir equilibrium.

The bead rate follows the linear-driving-force equation:

~~~text
dq/dt = k_LDF (q* - q).
~~~

'q' is current loading; 'q*' is local equilibrium loading; 'k_LDF' controls how
quickly the bead approaches equilibrium.

Each bead is coupled to surrounding pores using weights proportional to their
pore volumes. Positive adsorption is withdrawn only from incident pores with
positive concentration. The amount removed from gas exactly equals the amount
added to particles.

### 9.3 Final inlet boundary

The production model does not fix the concentration of an internal pore.
Instead, every hull-labelled inlet pore is connected to an external reservoir:

~~~text
boundary flux_i = G_boundary,i (c_in - c_i)
G_boundary,i = D A_i / L_i.
~~~

The nominal entrance area is the tube cross-section,
'A_in = pi Rtube^2 = 2.010619298e-4 m2'. It is interpreted as the superficial
reactor inlet area and is divided equally among the original hull-labelled
inlet pores, so 'sum(A_i) = A_in' for every packing. The distance is
'L_i = max(z_i - z_bottom, 0.5 dp)'. The lower limit prevents singular
conductance for a pore centre very close to the inlet plane. Internal particle
obstruction is represented by pore and throat geometry rather than by applying
a second inlet-porosity factor.

This boundary assumes a uniformly accessible external reservoir and negligible
distributor resistance. It preserves a common physical area across packings,
while inlet connectivity remains part of the packing structure.

Small pore volumes and adsorption create different timescales, making the
equations stiff. The script therefore uses the implicit BDF integrator and an
analytical sparse Jacobian. Sparse means that each pore interacts with nearby
pores and beads, not every unknown.

### 9.4 Every solver option

| Option | Default | Campaign value | Explanation |
|---|---:|---:|---|
| '--network' | required | one 'seed_*' network | Input geometry |
| '--output' | 'co2_transport_result' | unique seed output | Result directory |
| '--mode' | 'diffusion' | 'adsorption' | Activate bead uptake |
| '--diffusivity' | required | 1.5e-5 m2/s | Throat gas diffusivity |
| '--inlet-concentration' | required | 6.05 mol/m3 | External reservoir concentration |
| '--initial-concentration' | 0 | 0 mol/m3 | Initial concentration in all pores |
| '--temperature' | 298.15 K | 298.15 K | Isothermal temperature |
| '--particle-density' | 1600 kg/m3 | 1600 kg/m3 | Converts bead volume to mass |
| '--qsat' | required for adsorption | 5.332 mol/kg | Saturation loading |
| '--b-pa' | required for adsorption | 5.093e-5 1/Pa | Affinity |
| '--toth-exponent' | 1 | 1 | Langmuir when equal to one |
| '--k-ldf' | required for adsorption | 8.55e-3 1/s | Uptake-rate coefficient |
| '--initial-loading' | 0 | 0 mol/kg | Initial adsorbed amount per mass |
| '--inlet-boundary' | 'fixed' | 'finite' | Selects the production finite-flux boundary; 'fixed' reproduces the early baseline |
| '--tube-radius' | 8e-3 m | 8e-3 m | Defines the nominal total inlet area |
| '--bed-bottom' | 0 m | 0 m | Physical inlet-plane position |
| '--inlet-min-distance-dp' | 0.5 | 0.5 | Minimum boundary length divided by bead diameter |
| '--inlet-area-weighting' | 'equal' | 'equal' | Divides the total area among labelled inlet pores |
| '--t-end' | 500 s | 5000 s | Final simulation time |
| '--outputs' | 201 | 301 | Saved time points |
| '--linear-output-times' | off | off | If supplied, use equally spaced saved times |
| '--rtol' | 1e-6 | 1e-6 | Relative solver tolerance |
| '--atol' | 1e-10 | 1e-10 | Absolute solver tolerance |
| '--max-step' | unlimited | unlimited | Optional maximum internal step |

Without '--linear-output-times', output times are logarithmic, giving more
records during the fast early response. '--outputs' changes saved times, not
the integrator's internal accuracy.

The inlet concentration corresponds to about 14,998 Pa or 15 kPa CO2 at
298.15 K.

### 9.5 Adopted material values

| Parameter | Value | Basis |
|---|---:|---|
| 'q_s' | 5.332 mol/kg | Jedli et al. 298 K capacity |
| 'b' | 5.093e-5 1/Pa | Assumed missing 1e-3 multiplier in the reference |
| 't' | 1 | Langmuir form |
| 'k_LDF' | 8.55e-3 1/s | Spherical-LDF conversion from Hu et al. |
| bead density | 1600 kg/m3 | Project assumption |
| gas diffusivity | 1.5e-5 m2/s | Dry CO2--N2 baseline |

Jedli et al. report 'q_m = 5.332 mol/kg' and list affinity as
'6.79 mmHg^-1' at 298 K:
[Heliyon 10 (2024), e40672](https://doi.org/10.1016/j.heliyon.2024.e40672).
Literal conversion gives '5.093e-2 1/Pa', which caused almost instantaneous
local saturation and severe stiffness. We explicitly assume the table omitted
a factor of 1e-3 and use '6.79e-3 mmHg^-1 = 5.093e-5 1/Pa'. This assumption
must be stated in publications.

Hu et al. measured commercial 13X beads near our 0.8 mm radius and reported
'D/R^2 = 5.7e-4 1/s' for CO2 in nitrogen at 38 degC and 0.1 bar. The spherical
conversion is 'k_LDF = 15 D/R^2 = 8.55e-3 1/s'. This is our conversion, not a
directly reported LDF coefficient:
[Adsorption 20 (2014), 121--135](https://doi.org/10.1007/s10450-013-9554-z).

### 9.6 Reference command

~~~bash
python3 solve_co2_pore_adsorption.py \
    --network co2_pore_networks_power22/seed_18427 \
    --output adsorption_13x_b5p093e-5_seed_18427_5000s \
    --mode adsorption \
    --diffusivity 1.5e-5 \
    --inlet-concentration 6.05 \
    --initial-concentration 0 \
    --temperature 298.15 \
    --particle-density 1600 \
    --qsat 5.332 \
    --b-pa 5.093e-5 \
    --toth-exponent 1 \
    --k-ldf 8.55e-3 \
    --initial-loading 0 \
    --inlet-boundary finite \
    --tube-radius 0.008 \
    --bed-bottom 0 \
    --inlet-min-distance-dp 0.5 \
    --inlet-area-weighting equal \
    --t-end 5000 --outputs 301 \
    --rtol 1e-6 --atol 1e-10
~~~

### 9.7 Solver outputs

| File | Meaning |
|---|---|
| 'timeseries.csv' | Bed totals and means versus time |
| 'final_state.npz' | Final concentration of every pore and loading of every bead |
| 'run_metadata.json' | Parameters, units, status and quality measures |

'timeseries.csv' columns:

| Column | Meaning |
|---|---|
| 'time_s' | Time |
| 'mean_gas_concentration_mol_m3' | Void-volume-weighted mean concentration |
| 'gas_inventory_mol' | Gas-phase CO2 |
| 'adsorbed_inventory_mol' | CO2 on beads |
| 'total_inventory_mol' | Gas plus adsorbed CO2 |
| 'cumulative_reservoir_input_mol' | CO2 supplied to maintain the inlet |
| 'inventory_change_mol' | Change in total bed inventory |
| 'mass_balance_residual_mol' | Inventory change minus reservoir input |
| 'relative_mass_balance_error' | Normalized conservation error |
| 'mean_loading_mol_kg' | Adsorbent-mass-weighted mean loading |

## 10. Solver validation and reference result

Diffusion-only testing was completed first. Tightening tolerances changed gas
inventory and mean concentration by at most 3.90e-7 relative. The tighter run
had maximum relative mass-balance error 4.79e-9, supporting the standard
tolerances.

For seed 18427 with the final finite-inlet boundary at 5000 s:

| Quantity | Result |
|---|---:|
| Pores / particles / throats | 9397 / 1850 / 18104 |
| Gas inventory | 1.847561e-6 mol |
| Adsorbed inventory | 1.804906e-3 mol |
| Mean gas concentration | 0.569981 mol/m3 |
| Mean loading | 0.284318 mol/kg |
| Maximum relative mass-balance error | 9.458e-15 |
| Final relative mass-balance error | 7.561e-15 |

All inventories and loading increased monotonically.

The inlet-equilibrium loading from the adopted Langmuir equation is about
2.309 mol/kg. Mean loading at 5000 s is only 12.31% of this value, and mean gas
concentration is 9.42% of the reservoir concentration. The bed is still filling
from the bottom. Therefore, 5000 s is a common transient comparison time, not
an equilibrium-capacity calculation.

## 11. Twenty-case finite-inlet adsorption campaign

All networks use exactly the same physical and numerical parameters; only the
packing structure changes. Production results are stored below
'co2_adsorption_campaign_finite_inlet_5000s/seed_*'. The campaign is summarized
with:

~~~bash
python3 summarize_co2_adsorption_campaign.py \
    --results-root co2_adsorption_campaign_finite_inlet_5000s \
    --network-root co2_pore_networks_power22 \
    --output-dir co2_adsorption_finite_inlet_analysis \
    --expected-cases 20 --strict
~~~

All 20 runs reached 5000 s, passed physical bounds and network QA, and conserved
mass. The largest relative mass-balance error was 2.89e-12. No run reached the
chosen practical-equilibrium criterion, which is expected for the transient
study.

| Transient response | Mean | Range | Coefficient of variation |
|---|---:|---:|---:|
| Uptake at 5000 s [mol/kg] | 0.264573 | 0.205553--0.314837 | 11.04% |
| t10 relative to final uptake [s] | 96.28 | 82.67--114.79 | 8.70% |
| t50 relative to final uptake [s] | 1501.14 | 1388.94--1625.18 | 4.29% |
| t90 relative to final uptake [s] | 4195.77 | 4139.11--4246.57 | 0.67% |
| Particle-loading CV | 1.8856 | 1.7926--2.0338 | 3.45% |

An early fixed-concentration boundary was retained as a validation baseline.
Replacing it with the finite boundary lowered mean 5000 s uptake by about 3%
and increased mean t50 by about 2.7%, while preserving the structural rankings
(Spearman correlation 0.994 for uptake and 0.973 for t50). A deliberately broad
inlet-labelling sensitivity test motivated the finite-area normalization but is
not part of the production dataset.

The number of hull-labelled inlet pores remains correlated with uptake even
after total boundary area is fixed. This is interpreted as an entrance
connectivity descriptor, not a change of imposed area. It must be included in
simple baselines. Inlet count alone gives leave-one-out R2 about 0.525 for
uptake; adding geometric tortuosity raises it to about 0.585, leaving meaningful
unexplained structural variation.

## 12. Structure--performance analysis

The present analysis asks why the 20 beds give different transient responses
although their particle properties and bulk porosities are nearly the same.
All predictive scores use leave-one-bed-out cross-validation: one complete bed
is withheld, the model is fitted to the other 19, and this is repeated for
every bed. These are therefore not training scores.

Porosity alone was not predictive. Its leave-one-out R2 values were -0.221 for
uptake at 5000 s, -0.081 for t50 and -0.227 for the underutilized-particle
fraction. A negative value means that porosity predicted a withheld bed worse
than simply assigning the campaign mean. Raw inlet-pore count was more useful,
but it depends partly on boundary labelling. Physical inlet-accessibility
descriptors were therefore extracted.

~~~bash
python3 extract_co2_inlet_accessibility.py \
    --network-root co2_pore_networks_power22 \
    --output co2_inlet_accessibility.csv \
    --campaign-dataset co2_adsorption_finite_inlet_analysis/co2_adsorption_campaign_dataset.csv \
    --merged-output co2_adsorption_structure_merged.csv \
    --diffusivity 1.5e-5 --tube-radius 0.008 \
    --particle-diameter 0.0016 --interior-depth-dp 5
~~~

For every packing, the script calculates inlet pore volume, inlet-to-interior
throat count, unique second-layer pores, cross-sectional coverage and diffusive
conductances. Throat conductance is 'G = D A/L'. The principal descriptor,
'effective_inlet_to_interior_conductance_m3_s', comes from a small steady
network calculation. Inlet pores have concentration one, pores at least five
particle diameters into the bed have concentration zero, and conservation gives
the remaining concentrations. Total flux for this unit difference is the
effective entrance conductance. This is a geometry diagnostic, not another
transient adsorption calculation.

| Response | Physical descriptor | Spearman rho | Bootstrap 95% interval | FDR-adjusted p-value |
|---|---|---:|---:|---:|
| Uptake at 5000 s | inlet-to-interior conductance | 0.968 | 0.867--0.995 | 9.79e-11 |
| Underutilized-particle fraction | inlet-to-interior conductance | -0.959 | -0.989 to -0.847 | 9.90e-10 |
| t50 | unique second-layer pores | -0.814 | -0.918 to -0.576 | 4.72e-4 |

Higher conductance accompanies greater transient uptake and fewer poorly
supplied particles. More distinct second-layer pathways accompany shorter t50.
Polar inlet coverage is also associated with all three responses, indicating
that spatial distribution matters in addition to path count. These are strong
associations within 20 realizations, not causal laws or external validation.

The reduced analysis compares predefined models: porosity alone, inlet count
alone, conductance alone, second-layer count alone, conductance plus tortuosity,
conductance plus polar coverage, and a three-term physical model.

~~~bash
python3 analyze_co2_structure_performance.py \
    --dataset co2_adsorption_structure_merged.csv \
    --output co2_reduced_physical_model_analysis \
    --bootstrap 5000 --permutations 500
~~~

The final closure is selected only after inspecting leave-one-bed-out and
permutation results. A GNN is not part of the current paper workflow. It may be
reconsidered later only with a larger, more varied geometry dataset and only if
transparent descriptors leave important reproducible variation.

### 12.1 Reduced-model result

The reduced comparison selected different simple descriptors for different
questions. Inlet-to-interior conductance alone predicts uptake at 5000 s with
leave-one-out R2 = 0.924 and the underutilized-particle fraction with R2 =
0.863. Unique second-layer pore count predicts t50 with R2 = 0.651. Each model
has one-sided target-permutation p = 0.001996 using 500 permutations. The
smallest attainable value with this calculation is 1/(500+1), so it should be
reported as p < 0.002 rather than as higher numerical precision.

Adding tortuosity or polar coverage did not improve the conductance-only model.
A three-variable model raised uptake R2 only from 0.924 to 0.931 and worsened
the underutilization result. The single-descriptor models are therefore retained
for interpretation. Uptake and underutilized fraction are related outputs from
the same particle-loading field and are not presented as independent evidence.

### 12.2 Accessibility-depth sensitivity

The interior diagnostic plane was moved from 5 dp to 3, 7 and 10 dp. Mean
conductance decreased with depth, as expected for a longer diffusion path:

| Interior depth | Mean conductance [m3/s] | Coefficient of variation |
|---:|---:|---:|
| 3 dp | 4.194e-8 | 12.77% |
| 5 dp | 2.843e-8 | 9.11% |
| 7 dp | 2.158e-8 | 6.73% |
| 10 dp | 1.593e-8 | 5.39% |

Packing rankings remained stable. Spearman rank correlations between depths
were 0.953--0.993. Correlations with uptake remained 0.950--0.970, and those
with underutilized fraction remained -0.946 to -0.968. The result is therefore
not specific to the 5 dp choice. Five dp remains the primary definition because
it is outside the immediate boundary layer while retaining sensitivity to
entrance structure; the other depths are robustness results.

### 12.3 Homogeneous one-dimensional model

The homogeneous bed divides the axial direction into finite-volume cells. Gas
concentration 'c' and mean solid loading 'q' satisfy:

~~~text
epsilon dc/dt = D_eff d2c/dz2 - (1-epsilon) rho_particle dq/dt
dq/dt = k_LDF [q*(c) - q].
~~~

The material isotherm and LDF coefficient are fixed at the pore-network values;
only 'D_eff' is fitted. The top is closed. The bottom Robin flux contains the
external reservoir resistance and diffusion across half of the first cell:

~~~text
J_in = (c_reservoir - c_1) /
       [1/h_in + dz/(2 D_eff)].
~~~

The half-cell term is required for grid convergence. An earlier development
version omitted it; results from that version are superseded and are not used.
The fit minimizes a time-integrated squared loading error. This avoids bias
toward early time caused by the logarithmically dense output schedule.

For seed 18427, corrected fits using 100, 200 and 400 cells gave:

| Cells | Fitted D_eff [m2/s] | Time-weighted NRMSE | Final loading [mol/kg] |
|---:|---:|---:|---:|
| 100 | 8.74218e-7 | 1.412% | 0.276298 |
| 200 | 8.72860e-7 | 1.423% | 0.276197 |
| 400 | 8.72523e-7 | 1.427% | 0.276172 |

The 100-to-200 diffusivity change is 0.156%, and the 200-to-400 change is
0.039%. One hundred cells are therefore frozen for the 20-bed campaign. The
maximum absolute mass-balance residual in the accepted 100-cell test was
1.38e-13 mol; the final relative error was 7.03e-11. The maximum relative error
of 1.29e-7 occurred at 0.218 s when the total inventory was still very small.

The verified 100-cell model was fitted separately to all 20 pore-network uptake
curves. Every fit passed quality checks. The fitted effective diffusivity had a
mean of 7.584e-7 m2/s, a standard deviation of 1.754e-7 m2/s and a range from
4.340e-7 to 1.096e-6 m2/s. Its coefficient of variation was 23.13%, although
the bulk-porosity coefficient of variation was only about 0.55%. The mean
time-weighted fit NRMSE was 1.579%, with a range of 0.840--2.460%. Thus, one
fitted transport coefficient reproduces the network uptake curves well while
retaining strong packing-to-packing variation.

### 12.4 Effective-diffusivity closure

The final analysis connects the homogeneous transport coefficient to a physical
pore-network descriptor:

~~~bash
python3 analyze_co2_effective_diffusivity_closure.py \
  --homogeneous-summary co2_homogeneous_campaign_100cells/homogeneous_campaign_summary.csv \
  --accessibility \
    co2_inlet_accessibility_3dp.csv \
    co2_inlet_accessibility_5dp.csv \
    co2_inlet_accessibility_7dp.csv \
    co2_inlet_accessibility_10dp.csv \
  --primary-depth 5 --tube-radius 0.008 \
  --bootstrap 10000 --permutations 5000 \
  --output co2_effective_diffusivity_closure
~~~

The accessibility diffusivity is:

~~~text
D_access = G_access H / A_tube,
A_tube = pi R_tube^2.
~~~

'G_access' is the steady pore-network conductance from the inlet to the chosen
interior plane. Multiplication by bed height and division by tube area converts
it to diffusivity units.

At the primary depth of 5 dp, 'D_access' and fitted 'D_eff' have Pearson
correlation 0.951 and Spearman correlation 0.953. Porosity alone gives
leave-one-bed-out R2 = -0.234 and cannot predict the fitted diffusivity.

| Model | Leave-one-bed-out R2 | Meaning |
|---|---:|---|
| porosity-only linear | -0.234 | global void fraction is not predictive |
| proportional 'D_eff = alpha D_access' | 0.555 | simple and physical, but too restrictive |
| affine accessibility model | 0.881 | strong prediction, but has a negative intercept |
| positive accessibility power law | 0.890 | strong prediction and always positive |
| raw-conductance affine model | 0.898 | best numerical score, but geometry-specific and has a negative intercept |

For the proportional model, 'alpha = 0.1523' with bed-bootstrap 95% interval
0.1427--0.1616. Its predictive error shows that direct proportionality is not
enough. The positive power law is therefore the preferred interpretable
empirical closure:

~~~text
D_eff = C D_access^n,
C = 1.2622e7 and n = 2.4969 when SI units are inserted.
~~~

The exponent bootstrap median is 2.504 with a 95% interval of 2.057--2.768.
Because the dimensional value of 'C' depends on the chosen units, the paper
should present the normalized form using 'D_m = 1.5e-5 m2/s':

~~~text
D_eff / D_m = 0.7589 (D_access / D_m)^2.4969.
~~~

The raw-conductance linear model is not selected even though its score is
slightly higher. Its intercept is negative, so it can predict negative
diffusivity outside the fitted range. The power law is positive and loses
little predictive accuracy.

Moving the diagnostic plane from 3 to 10 dp preserves the result. Pearson
correlations remain 0.936--0.951 and affine leave-one-bed-out R2 values remain
0.834--0.881. Five dp remains the primary definition because it gives the
strongest prediction while lying beyond the immediate inlet layer.

The closure is internally validated by leaving out complete beds, bootstrap
resampling and permutation testing. It is not yet a universal correlation:
all 20 beds use one bead size, one tube diameter and one material parameter
set. It should only be used within that studied range until new geometries are
tested.

## 13. Complete workflow in one view

~~~text
LAMMPS packing
 -> mechanical QA
 -> ParaView inspection
 -> robust porosity
 -> power-22 pore network
 -> network QA
 -> diffusion validation
 -> finite-inlet adsorption validation
 -> 20 finite-inlet adsorption cases
 -> campaign QA and structure--response table
 -> inlet-accessibility extraction
 -> low-dimensional physical baselines
 -> homogeneous one-dimensional comparison
 -> accessibility-to-effective-diffusivity closure
~~~

## 14. Assumptions that must always be reported

1. Spherical, monodisperse, dry zeolite beads are used.
2. Tube diameter equals ten bead diameters.
3. Gas transport is diffusion only.
4. Temperature is fixed at 298.15 K.
5. Initial gas concentration and loading are zero.
6. The bottom is coupled to a 6.05 mol/m3 reservoir through finite conductances;
   the total nominal area equals the tube cross-section; top and wall are closed.
7. The affinity assumes a missing 1e-3 reference multiplier.
8. The LDF coefficient is inferred from a bead time constant measured at a
   somewhat higher temperature.
9. The apparent density is assumed to be 1600 kg/m3.
10. The 5000 s output is transient, not equilibrium.
11. Every campaign case uses identical parameters; only packing structure
    changes.
12. The five-particle-diameter accessibility plane is an analysis definition;
    its depth must be reported and later tested for sensitivity.
13. Correlations from 20 beds are exploratory and require leave-one-bed-out and
    permutation checks before they are used as closures.
14. The power-law closure is internally cross-validated for the present
    geometry; it is not yet a universal packed-bed correlation.
