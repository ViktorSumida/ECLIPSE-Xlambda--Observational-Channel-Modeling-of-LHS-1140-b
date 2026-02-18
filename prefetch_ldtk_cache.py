# prefetch_ldtk_cache.py — preview/cache builder with kernel ON + post_smoothing
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from ldtk_ldc import ldc_ldtk_for_channels
from phoenix_sed import phoenix_flux_interp_um

# ============= USER CONFIG (alinhar com o grid) =============
lambdaEff_um = np.array([0.75, 1.00, 1.50, 2.00, 2.40], float)
Rchan = 100.0
dx_half = lambdaEff_um / (2.0 * Rchan)
dx_low_um, dx_high_um = -dx_half, dx_half

teff, logg, feh = 3096.0, 5.0, 0.0
LDTK_SUBSAMPLES = 1201
EPS = 1e-15

# ===== Super-continuum only for *weighting* (optional) =====
lam_hr = np.linspace(0.4, 2.8, 20000)
F_hr   = phoenix_flux_interp_um(lam_hr, T=teff, feh=feh)
F_cont = F_hr / (np.nanmax(F_hr) or 1.0)  # aqui basta normalizar

# ============= Preview (único gráfico) =============
plt.figure(figsize=(10,4.2))
plt.plot(lam_hr, F_hr/np.nanmax(F_hr), lw=0.7, alpha=0.35, label="PHOENIX raw")
plt.plot(lam_hr, F_cont, lw=2.0, label="Weight for LDTK")
for j, lc in enumerate(lambdaEff_um):
    lo = lc + dx_low_um[j]; hi = lc + dx_high_um[j]
    plt.axvspan(lo, hi, color="k", alpha=0.05, zorder=0)
plt.xlabel("Wavelength [µm]"); plt.ylabel("Normalized scale")
plt.title("LDTK input preview — weighting")
plt.legend(loc="upper right"); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# ============= LDTK with smoothing policy =============
c1, c2, c3, c4 = ldc_ldtk_for_channels(
    lambdaEff_um=lambdaEff_um, dx_low_um=dx_low_um, dx_high_um=dx_high_um,
    teff=teff, logg=logg, feh=feh,
    throughput_array=(lam_hr, F_cont),     # pode ser plano também
    law="claret4", dataset="visir-lowres",
    # 1) mata linhas DENTRO do canal
    smooth_mode="R", R_smooth_lines=150.0, smooth_kind="tukey", tukey_alpha=0.60,
    jitter_n=7, jitter_spread=0.6, jitter_reduce="mean",
    n_sub=LDTK_SUBSAMPLES, verbose=True,
    # 2) suaviza os c_i ENTRE canais (const-R em log-λ)
    post_smooth_R=150.0
)

print("\n[LDTK] LDCs (kernel ON + post_smoothing):")
for i, lam in enumerate(lambdaEff_um):
    print(f"λ={lam:.3f} µm | c1={c1[i]:+.4f}, c2={c2[i]:+.4f}, c3={c3[i]:+.4f}, c4={c4[i]:+.4f}")