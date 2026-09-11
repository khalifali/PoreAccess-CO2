# Effective-diffusivity closure analysis

## Closure candidates

The simplest dimensionally consistent, one-parameter closure is

`D_eff = alpha D_access`, with `D_access = G_access H / A_tube`.

Here, `G_access` is the pore-network conductance from the inlet to an interior
plane at 5 particle diameters, `H` is bed height, and
`A_tube = pi R^2` is the full tube area. The fitted dimensionless factor is
`alpha = 0.152285` with a bed-bootstrap 95% interval of
`[0.142679, 0.161622]`.

A positive empirical alternative is `D_eff = C D_access^n`, for which this
ensemble gives `C = 1.26221e+07` and `n = 2.49691`. Because `C` carries
the units needed by the fitted exponent, its value is tied to SI units. The
model-comparison table should be used to decide whether its added exponent is
justified over the one-parameter closure.

## Held-out-bed validation

The physical closure gives leave-one-bed-out `R2 = 0.5549`
and `RMSE = 1.1406e-07 m2/s`. A porosity-only linear
model gives `R2 = -0.2340`. The best tested predictive
form is `raw_conductance_linear` with `R2 = 0.8977`.

These statistics describe this 20-bed ensemble. They support a structural
closure inside the studied geometry and parameter range; they do not yet prove
transfer to other tube-to-particle ratios, particle sizes, or sorbents.
