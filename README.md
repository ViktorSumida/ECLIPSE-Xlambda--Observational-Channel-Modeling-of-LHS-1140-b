# ECLIPSE-Xλ — LHS 1140 b worked example (observational channels)

ECLIPSE-Xλ is a forward model to simulate **stellar contamination** signatures in exoplanet transmission spectroscopy using the **same observational channel grid** as the data (channel centers + finite bandpasses). It supports **unspotted**, **spot-only**, **facula-only**, and **spot+facula (“both”)** scenarios and writes the modeled transit depth spectrum `D(λ)` on the observational channels.

This repository includes a worked example for **LHS 1140 b**, as used in our paper (A&A):  
**doi:10.1051/0004-6361/202556358**

> Design goal: do *not* invent an arbitrary wavelength grid. Instead, compute everything (SED contrasts, limb darkening, and depth spectrum) directly on the observational channels.

---

## What this model computes (conceptually)

Given:
- **Star** parameters (radius, mass, Teff, etc.),
- **Planet/orbit** parameters (Rp, period, inclination, semi-major axis, etc.),
- An **observational channel grid** (λ centers + Δλ bandpasses),
- **Active-region** parameters (filling factors + temperatures; optionally geometry),

ECLIPSE-Xλ computes:
- Channel-integrated contrasts `r_spot(λ)` and/or `r_facula(λ)` from SED providers (e.g., PHOENIX),
- Limb-darkening coefficients **per channel** (in this workflow: **grey atmosphere + Planck weighting**, i.e., no LDTK),
- A simulated transit depth spectrum `D(λ)` on the observational channels,
- Optional plotting and a geometry “preflight” check (to warn about occultations if you assume non-occulted baselines).

---

## Repository layout (minimum)

- `grid.py`  
  Driver script. Loads channels, warms SED cache, computes per-channel LDCs, and runs scenario grids (unspotted / spot / facula / both).
- `main.py`  
  Core engine: channel integration, instrument pre-convolution (constant R or auto R(λ)), contrast smoothing, and transit depth synthesis.
- `phoenix_sed.py`  
  PHOENIX interface + cache warm-up utilities (local grid or on-demand download, depending on your implementation).
- `star.py`, `Planeta.py`, `Eclipse.py` (names may vary)  
  Stellar map, orbital geometry, and eclipse/transit utilities.

---

## Installation

Recommended: **Python ≥ 3.10**

Install dependencies:
```bash
pip install numpy pandas matplotlib scipy astropy requests
Notes:

astropy is only needed if you read FITS products elsewhere; the LHS 1140 b example uses plain text I/O by default.

PHOENIX handling depends on how phoenix_sed.py is implemented (local grid vs downloader).

Observational channels input (LHS 1140 b)
ECLIPSE-Xλ is designed to run on the same channels as the observations. You can provide channels in two main ways:

A) MRT-like file (recommended)
Set in grid.py:

python
Copy code
MRT_INPUT_MODE = "file"
MRT_PATH = "apjlad5afat2_mrt.txt"
USE_ORDER = None  # optionally 1/2 if your file includes an order column
Expected numeric content per row (minimum):

center_um, width_um

Optional columns (if present):

depth_ppm, edepth_ppm, order, etc.

The loader is intentionally permissive: it extracts floats from each row and ignores non-numeric tokens. By default it maps:

vals[0] -> center_um

vals[1] -> width_um

vals[2] -> depth_ppm (optional)

vals[3] -> edepth_ppm (optional)

vals[6] -> order (optional)

If your file layout differs, update the mapping in the MRT loader inside grid.py.

B) Manual channels (CSV or arrays)
Set:

python
Copy code
MRT_INPUT_MODE = "manual"
MANUAL_CHANNELS_CSV = "channels.csv"
CSV expected columns (any of the following combinations):

lambda_um or lambda_nm

width_um or width_nm

OR R (width computed as Δλ = λ/R)

Example:

csv
Copy code
lambda_um,width_um
1.02,0.010
1.03,0.010
Practical advice: widths should represent an effective integration window per data point (FWHM-ish is fine if that’s what the product reports).

To avoid pathological Δλ → 0, the driver can enforce a floor:

python
Copy code
R_MIN_CHAN_FLOOR = 1_000  # example
width_um = max(width_um, lambda_um / R_MIN_CHAN_FLOOR)
Running the LHS 1140 b grid
From the repository root:

bash
Copy code
python grid.py
This will:

Load the observational channels (MRT or manual),

Warm the SED cache for required temperatures,

Compute grey/Planck limb-darkening coefficients per channel,

Run scenario grids:

unspotted

spot

facula

both (spot + facula)

Outputs
The main output is a CSV-like text file:

simulation_results_LHS1140b.txt (name may include TARGET)

Columns:

f_spot, tempSpot, f_facula, tempFacula, wavelength, D_lambda

Where:

wavelength is typically written in nm (verify the unit used in your main.py writer),

D_lambda is the modeled transit depth in ppm.

Input parameters (what you actually control)
Most configuration is at the top of grid.py. The key groups are:

1) Star parameters
Typical inputs:

R_star : stellar radius (solar radii or meters; be consistent with your code)

M_star : stellar mass

T_phot : photospheric effective temperature [K]

Optional (if used by your pipeline): logg, [Fe/H], distance, rotation, etc.

These control:

the stellar baseline flux (via SED provider),

limb darkening (per channel),

geometry scaling (if absolute radii are used).

2) Planet + orbit parameters
Typical inputs:

R_p : planet radius (RJ, RE, or meters)

P : orbital period [days]

a : semi-major axis [AU] or in stellar radii (depending on your internal convention)

i : inclination [deg]

e : eccentricity

ω / anom : argument of periastron / anomaly (if used)

These control:

the baseline transit depth,

transit chord geometry and impact parameter,

timing/phase sampling (if your engine simulates time-resolved curves).

3) Active regions (spots / faculae)
Core contamination parameters:

f_spot : spot covering fraction on the visible hemisphere (0–1)

T_spot : spot temperature [K]

f_facula : facula covering fraction (0–1)

T_facula : facula temperature [K]

both_mode : enables simultaneous spot+facula simulation

Geometry parameters (optional, only if you model localized regions):

number of regions (e.g., N_spots, N_faculae)

latitudes/longitudes (deg) for each region

region radii (either fixed or derived from covering fractions)

Important: if you assume the “non-occulted contamination baseline”, you typically want to avoid transit chord crossings (or treat them explicitly as occultations).

4) Observational channels / windows
lambda_center : channel centers (µm or nm)

width : channel widths (µm or nm)

optional: order for multi-order instruments

These control:

how SEDs are integrated in each bandpass,

how contrasts and LDCs are computed per channel.

5) Instrument pre-convolution (realism knob)
Two common modes:

(i) Constant resolving power

python
Copy code
AUTO_R_FROM_CHANNELS = False
CONSTANT_R = 25
(ii) Auto R(λ) from channel widths

python
Copy code
AUTO_R_FROM_CHANNELS = True
R_AUTO_FLOOR_R  = 300.0
R_AUTO_SMOOTH_R = 520.0
R_AUTO_BETA     = 1.0
Concept:

R(λ) ≈ β * λ / Δλ_channel

smoothed in log(λ) to avoid jagged resolution jumps

Interpretation:

treat β as a calibration factor if your reported Δλ is not a true LSF width.

6) SED providers / modes (PHOENIX vs alternatives)
Example configuration:

python
Copy code
SED_MODE_PHOT   = "phoenix"
SED_MODE_ACTIVE = "phoenix"
Depending on your implementation, other modes may include:

bb (blackbody)

phoenix_continuum

phoenix_continuum_hybrid_global

phoenix_continuum_hybrid_global_tamed

Hybrid/continuum modes can reduce line-driven artifacts and produce smoother contrast baselines.

7) Limb darkening mode (this repo’s default)
This workflow uses:

grey atmosphere + Planck weighting, per channel

Scientific caution:

this is a controlled approximation; benchmark against PHOENIX/LDTK LDCs at least once if your analysis relies on fine structure (<100 ppm).

8) Apodization inside each channel (integration stability)
If enabled:

python
Copy code
APODIZE_TUKEY = True
TUKEY_ALPHA   = 0.80
This reduces edge sensitivity when integrating SEDs across each bandpass.

Geometry “preflight” check (optional but recommended)
Before running large grids, the driver can evaluate whether the transit chord intersects the largest implied regions:

python
Copy code
will_hit, info = preflight_occultation_check(...)
If it predicts intersections, the driver can warn or abort (depending on your settings).

If you want to model occultations explicitly (spot-crossing bumps), you should disable “abort on intersection” logic and ensure the forward model includes occulted features.

Repro checklist (LHS 1140 b)
Place the channel file in the repo:

apjlad5afat2_mrt.txt (or your CSV)

Confirm units:

centers in µm, widths in µm (or adjust loader)

Choose convolution mode:

constant R for quick tests

auto R(λ) if resolution varies across channels

Run:

bash
Copy code
python grid.py
Inspect:

simulation_results_LHS1140b.txt

Security note (do not commit secrets)
If you use notifications, never commit tokens.
Read credentials from environment variables only (example):

bash
Copy code
export PUSHOVER_USER="..."
export PUSHOVER_TOKEN="..."
Citation
If you use this worked example in academic work, cite our paper:

doi:10.1051/0004-6361/202556358

When possible, also cite:

the original ECLIPSE lineage (if applicable in your codebase),

PHOENIX model atmospheres (per their citation guidelines),

and this repository (GitHub URL + commit hash / release tag).