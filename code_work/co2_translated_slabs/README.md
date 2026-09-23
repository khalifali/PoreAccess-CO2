# Tortuosity and translated-slab assessment

## Result and research decision

Continue investigating the predictive transport closure, but retain an association claim.
The current evidence does not establish inlet-specific causation or literature novelty.
The shortest-path geometric tortuosity descriptor has substantially weaker uptake and
underutilization associations than the original inlet-connected conductance.
Matched translated slabs also fail to reproduce the original uptake correlation.
However, the matched lowest slab is weak too: source-boundary representation is a
competing explanation, not a resolved control.

All values below are Spearman correlations across the same 20 beds.

| Descriptor | q at 5000 s | t50 | Underutilized fraction |
|---|---:|---:|---:|
| Global geometric tortuosity | -0.459 | -0.110 | +0.428 |
| Original G_access (hull inlet to 5 dp) | +0.968 | -0.415 | -0.959 |
| Matched slab 0–5 dp | -0.065 | +0.197 | +0.080 |
| Matched slab 5–10 dp | -0.048 | +0.262 | +0.007 |
| Matched slab 10–15 dp | +0.432 | +0.152 | -0.408 |
| Matched slab 15–20 dp | -0.296 | +0.221 | +0.275 |

The slab rows use 1 dp boundary bands. Across bands of 0.5, 1 and 1.5 dp,
the maximum absolute uptake correlation is 0.432. Boundary sensitivity is plotted
rather than selecting the best band after seeing outcomes. These are exploratory
in-sample correlations, not predictive cross-validation. CSV p-values are nominal,
unadjusted two-sided Spearman p-values; no multiple-testing significance or formal
difference between dependent correlations is claimed. n=20 limits inference.

The earlier conversation reported tortuosity correlations of -0.507, -0.006 and
+0.444. Those values are not reproduced by the current repository. This analysis
recomputes tortuosity from raw networks and asserts agreement with the stored
campaign descriptor. The new slides use the verified current values.

## Why the old and new 0–5 dp results differ

An additional 80 full-network solves change the source and sink separately:

| Source held at 1 | Sink held at 0 | Uptake Spearman rho |
|---|---|---:|
| Original hull-labelled inlet | All pores at/above 5 dp | +0.968 |
| Original hull-labelled inlet | All pores at/above 4 dp | +0.968 |
| All pores in bottom 1 dp band | All pores at/above 5 dp | +0.023 |
| All pores in bottom 1 dp band | All pores at/above 4 dp | -0.065 |

Replacing the source-pore selection is the dominant reason for the change in uptake
correlation in these controls. Moving the sink alone preserves the strong association.
The full-band source fixes many previously internal pores directly; the original inlet
labels can also extend above 1 dp, so this is a replacement, not necessarily a superset.
The last row has the same uptake rank correlation as the cropped 0–5 dp slab. All four
rows use the full network, avoiding cropping as a confound in this comparison.
`boundary_bridge_conductances.csv` and `boundary_bridge_correlations.csv` retain all
values. This identifies sensitivity to the diagnostic boundary definition, not a
unique microscopic mechanism. A matched interior translation of the original inlet
requires a separate choice of source patches. The two 0–5 dp quantities should not be
presented as interchangeable measurements of the same conductance.

## Exact definitions

- Outcomes: the finite-inlet, 5000 s adsorption campaign, joined strictly by seed.
  t50 is time to half each bed's final uptake, not half equilibrium uptake.
  Underutilization is the fraction of particles below half the inlet-equilibrium loading.
- Tortuosity: minimum throat-length path from any inlet pore to any outlet pore,
  divided by max(outlet z) minus min(inlet z), exactly as in the campaign summarizer.
  This is a shortest-path geometric ratio, not a flux-weighted transport tortuosity.
- Throat conductance: D A/L, D = 1.5e-5 m2/s and dp = 0.0016 m.
- Original G_access: fix hull-labelled inlet pores at 1 and all pores at or above
  lowest-pore z + 5 dp at 0 on the original network. Recomputed, not copied.
- Translated slabs: retain pores within [z0+a dp, z0+(a+5) dp], with a = 0,5,10,15.
  Retain only throats whose two endpoints lie inside that window. Exterior paths
  cannot bypass the slab. In each window fix the bottom band to 1 and the top band
  to 0, using the same band thickness at every position. Solve the sparse network
  Laplacian at other pores and sum source flux. Nonspanning components contribute zero.
  These boundary bands have finite thickness inside the window, so their separation
  is less than 5 dp. They define a matched window diagnostic, not a full 5 dp
  plane-to-plane conductance or an effective diffusivity.
- The original hull-labelled source differs from a full cross-sectional band.
  It is therefore a separate benchmark. Comparing it to an interior slab changes
  both position and boundary representation. The matched series isolates position
  under the specified band convention, but cannot alone prove physical inlet causation.
- Full 20–25 dp windows do not fit any network (heights about 22.0–22.3 dp).
  `skipped_windows.csv` records all 20 exclusions. No shorter window is substituted.

## Reproduce

From `code_work` with Python, NumPy, SciPy, pandas and Matplotlib installed:

```bash
python analyze_co2_translated_slabs.py \
  --network-root co2_pore_networks_power22 \
  --window-dp 5 --max-start-dp 20 --output co2_translated_slabs
python -m unittest discover -s tests -p 'test_translated_slabs.py' -v
```

The analysis requires an exact 20-seed match and finite outcomes. It produces 240
slab rows, descriptor/outcome tables, all correlation values, two figures in PDF/PNG,
and input SHA-256 hashes in `provenance.json`. It does not modify raw inputs or rerun
transient adsorption. The original G_access is always the 5 dp benchmark even when
changing the optional slab width. The analytical tests check a known series network,
translation invariance, conductance scaling, exclusion of an exterior bypass,
disconnection and invalid/empty boundary bands.

Both Beamer folders contain copies of the generated PDFs for independent compilation.
If regenerating, copy both figure PDFs to each deck's `figures` directory, then compile
both decks twice as documented in the presentation README.

## Next scientific control

Use controlled changes to inlet connectivity or boundary-matched source patches
at different positions, with their area/count held fixed. Evaluate any resulting
closure on independent beds. Weak geometric-tortuosity correlations do not exclude
all tortuosity mechanisms, and a good conductance correlation alone does not prove
novelty relative to prior work.
