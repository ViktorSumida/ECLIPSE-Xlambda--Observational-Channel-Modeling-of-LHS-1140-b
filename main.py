#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — ECLIPSE‑Xλ core

Highlights
----------
• Instrument pre-convolution supports:
  - constant resolving power R
  - variable resolving power R(λ), driven automatically from the channel grid
    (center λ and widths) or by a user-supplied function/array.

• Channel integration: SEDs are integrated against the channel window
  (rectangular or Tukey), with optional throughput weighting.

• SED providers: PHOENIX full, continuum-like, hybrid (global / tamed), or BB.

• Optional: "anchor continuum" (shared envelope), FFT smoothing of r(λ),
  Planck-trend substitution, Zeeman placeholder, rotational/macro broadening.

Interface notes
---------------
- To enable automatic R(λ) from your channel widths, pass:
    R_preconv_override="auto"
    R_auto_floor_R=...   (enforces a minimum R; use your channel floor)
    R_auto_smooth_R=...  (smooth the R(λ) curve in log λ)
    R_auto_beta=...      (calibration factor between channel Δλ and LSF FWHM)
- If you pass a function in R_preconv_override, it must accept λ[µm] and return
  a scalar R or an array R(λ). An array is also accepted directly.

Nota importante (patch):
- Passando `cap_env=None` nos modos híbridos agora mantém **sem teto** (cap=∞).
  Antes o default caía para 1.00 e “cortava” o topo das linhas.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from star import Star
try:
    from eclipse_nv1 import Eclipse
except Exception:
    from Eclipse import Eclipse

from Planeta import Planeta
from keplerAux import keplerfunc

from phoenix_sed import (
    phoenix_flux_interp_um, warm_phoenix_cache,
    phoenix_continuum_hybrid_global,
    phoenix_continuum_hybrid_global_tamed,
    super_continuum_envelope,
    T_MIN_PHX, T_MAX_PHX
)

# ==========================================================
# Constants & modes
# ==========================================================
EPS   = 1e-12
C_KMS = 299_792.458

EFFECTIVE_DEPTH_MODE = "min"  # mid|min|area|percentile|parabola
AREA_FRAC       = 0.30
PCT_LOW         = 0.015
PARAB_WIN_FRAC  = 0.10

DEFAULT_FACULA_CONTRAST_SCALE = 1.0
DEFAULT_SPOT_CONTRAST_SCALE   = 1.0

# ==========================================================
# Preflight (geometry helpers)
# ==========================================================
def _calc_matrix_size_and_pixels_per_rstar(planet: Planeta,
                                           min_pixels: int, max_pixels: int,
                                           pixels_per_rp: float,
                                           margin: float = 0.1) -> tuple[int, float]:
    rp_rstar = float(planet.raioPlanetaRstar)
    pprs = float(pixels_per_rp) / max(rp_rstar, EPS)
    pprs = float(np.clip(pprs, min_pixels / 2.0, max_pixels / 2.0))
    matrix_size = int(2.0 * pprs * (1.0 + float(margin)))
    matrix_size = int(np.clip(matrix_size, min_pixels, max_pixels))
    return matrix_size, pprs

def _transit_chord_pixels(planet: Planeta, star_pixels: float, N: int, n_points_hint: int = 401):
    dtor = np.pi / 180.0
    a_pix = planet.semiEixoRaioStar * star_pixels
    n = max(101, int(n_points_hint))

    total_h = 3.0 * (2 * (90. - np.arccos(np.cos((-np.arcsin(planet.semiEixoRaioStar * np.cos(planet.anguloInclinacao * dtor)) / dtor) * dtor) / planet.semiEixoRaioStar) / dtor) * planet.periodo / 360. * 24.)
    t = np.linspace(-0.5*total_h, +0.5*total_h, n)

    nk = 2.0*np.pi / (planet.periodo * 24.0)
    Tp = planet.periodo * planet.anom / 360.0 * 24.0
    m = nk * (t - Tp)
    eccanom = keplerfunc(m, planet.ecc)

    xs = a_pix * (np.cos(eccanom) - planet.ecc)
    ys = a_pix * (np.sqrt(1 - planet.ecc**2) * np.sin(eccanom))

    ang = planet.anom * dtor - (np.pi/2.0)
    xp = xs*np.cos(ang) - ys*np.sin(ang)
    yp = xs*np.sin(ang) + ys*np.cos(ang)

    xplan = xp - xp[np.argmin(np.abs(t))]
    yplan = yp * np.cos(planet.anguloInclinacao * dtor)

    msk = (np.abs(xplan) < 0.6*N) & (np.abs(yplan) < 0.6*N)
    return (xplan[msk] + N/2.0, yplan[msk] + N/2.0)

def preflight_occultation_check(*,
    raioStar: float,
    semiEixoUA: float,
    raioPlanetaRj: float,
    periodo: float,
    anguloInclinacao: float,
    ecc: float,
    anom: float,
    min_pixels: int, max_pixels: int, pixels_per_rp: float,
    quantidade_spot: int, quantidade_fac: int,
    lat_spot: np.ndarray, long_spot: np.ndarray,
    lat_fac:  np.ndarray, long_fac:  np.ndarray,
    ff_spot_max: float, ff_fac_max: float,
) -> tuple[bool, dict]:
    raioStar_km = float(raioStar) * 696_340.0
    planet = Planeta(semiEixoUA, raioPlanetaRj, periodo, anguloInclinacao, ecc, anom, raioStar_km, 0)

    N, star_pixels = _calc_matrix_size_and_pixels_per_rstar(
        planet, min_pixels=min_pixels, max_pixels=max_pixels, pixels_per_rp=pixels_per_rp
    )

    star = Star(star_pixels, raioStar, 1.0, N)

    r_sp_pix = (np.sqrt(max(ff_spot_max, 0.0) / max(1, quantidade_spot)) if ff_spot_max > 0 else 0.0)
    r_fa_pix = (np.sqrt(max(ff_fac_max,  0.0) / max(1, quantidade_fac))  if ff_fac_max  > 0 else 0.0)

    if r_sp_pix > 0:
        for j in range(max(1, int(quantidade_spot))):
            star.addMancha(Star.Mancha(intensidade=1.0, raio=r_sp_pix,
                                       latitude=float(lat_spot[j % len(lat_spot)]),
                                       longitude=float(long_spot[j % len(long_spot)])))
    if r_fa_pix > 0:
        for j in range(max(1, int(quantidade_fac))):
            star.addFacula(Star.Facula(raio=r_fa_pix, intensidade=1.0,
                                       latitude=float(lat_fac[j % len(lat_fac)]),
                                       longitude=float(long_fac[j % len(long_fac)])))

    star.build_static_structures()

    xpix, ypix = _transit_chord_pixels(planet, star_pixels, N, n_points_hint=401)
    r_p_pix = float(planet.raioPlanetaRstar * star_pixels)
    chord_mask = np.zeros((N, N), dtype=bool)
    yy, xx = np.indices((N, N))
    r2 = r_p_pix*r_p_pix
    for x0, y0 in zip(xpix, ypix):
        dx = xx - x0; dy = yy - y0
        chord_mask |= (dx*dx + dy*dy) <= r2

    any_sp = any((chord_mask & m).any() for m in getattr(star, "_spot_masks", []))
    any_fa = any((chord_mask & m).any() for m in getattr(star, "_fac_masks", []))
    will_hit = bool(any_sp or any_fa)
    info = dict(N=N, star_pixels=star_pixels, r_p_pix=r_p_pix,
                r_sp_pix=r_sp_pix, r_fa_pix=r_fa_pix,
                any_sp=bool(any_sp), any_fa=bool(any_fa))
    if will_hit:
        print(f"⚠️ [preflight] transit chord intersects features: spot={any_sp}, facula={any_fa}")
    else:
        print("ℹ️ [preflight] no intersection detected.")
    return will_hit, info

# ==========================================================
# Throughput / windows / constant-R blur
# ==========================================================
def _load_throughput_csv(path: str):
    if path is None or (not os.path.exists(path)):
        return None
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
            except Exception:
                continue
            xs.append(x); ys.append(y)
    if not xs:
        return None
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if np.nanmax(ys) > 0:
        ys = ys/np.nanmax(ys)
    mx = float(np.nanmax(np.abs(xs)))
    if mx > 3000.0:
        lam_um = xs / 1e4
    elif mx > 40.0:
        lam_um = xs / 1e3
    else:
        lam_um = xs
    order = np.argsort(lam_um)
    return lam_um[order], ys[order]

def _throughput_at(lam_um: np.ndarray, thr_pair):
    if thr_pair is None:
        return np.ones_like(lam_um, float)
    lam_t, thr_t = thr_pair
    return np.interp(lam_um, lam_t, thr_t, left=0.0, right=0.0)

def _tukey_window(x, lo, hi, alpha=0.30):
    x = np.asarray(x, float); a = float(np.clip(alpha, 0.0, 1.0))
    L = hi - lo
    if L <= 0:
        return np.zeros_like(x)
    u = (x - lo)/L
    w = np.zeros_like(x, float)
    left  = (u < a/2)
    right = (u > 1 - a/2)
    mid   = (~left) & (~right) & (u>=0) & (u<=1)
    w[left]  = 0.5*(1 + np.cos(np.pi*(2*u[left]/a - 1)))
    w[right] = 0.5*(1 + np.cos(np.pi*(2*u[right]/a - 2/a + 1)))
    w[mid   ] = 1.0
    w[(u<0) | (u>1)] = 0.0
    s = float(np.trapz(w, x)); s = s if (np.isfinite(s) and s > 0) else 1.0
    w /= s
    return w

def _gauss_blur(x_nm: np.ndarray, y: np.ndarray, R_eff: float) -> np.ndarray:
    x = np.asarray(x_nm, float); y = np.asarray(y, float)
    out = np.full_like(y, np.nan, dtype=float)
    if (not np.any(np.isfinite(y))) or (not np.isfinite(R_eff)) or (R_eff <= 0):
        return y.copy()
    inv = 1.0/2.354820045
    for i, L in enumerate(x):
        if not np.isfinite(L): continue
        fwhm = L/float(R_eff); sigma = fwhm*inv
        w = np.exp(-0.5*((x-L)/max(sigma, EPS))**2)
        good = np.isfinite(w) & np.isfinite(y)
        if np.any(good):
            out[i] = np.nansum(w[good]*y[good]) / max(np.nansum(w[good]), EPS)
    return out

def _blur_constant_R_loglambda(lam_um, y, R):
    lam = np.asarray(lam_um, float); y = np.asarray(y, float)
    lam = np.clip(lam, 1e-12, None)
    z = np.log(lam)
    dz = 1.0 / (max(float(R), 1e-6) * 12.0)
    z_grid = np.arange(z.min(), z.max() + 1e-12, dz)
    y_grid = np.interp(z_grid, z, y)
    sigma_log = 1.0 / (2.355 * max(float(R), 1e-6))
    half = int(max(3, np.ceil(6.0 * sigma_log / dz)))
    k = np.arange(-half, half + 1) * dz
    ker = np.exp(-0.5 * (k / max(sigma_log, 1e-15))**2)
    ker /= max(float(ker.sum()), EPS)
    yg = np.convolve(np.pad(y_grid, (half, half), mode="edge"), ker, mode="same")[half:-half]
    return np.interp(lam, np.exp(z_grid), yg)

# ==========================================================
# Variable-R blur in log λ
# ==========================================================
def _blur_variable_R_loglambda(lam_um: np.ndarray, flux: np.ndarray, R_vec: np.ndarray) -> np.ndarray:
    """
    Pointwise Gaussian blur in log λ with σ_i = 1/(2.355*R_i).
    This is a simple, robust O(N * k) scheme (k ≈ ~10 σ neighborhood).
    """
    lam = np.asarray(lam_um, float)
    f   = np.asarray(flux,  float)
    R   = np.asarray(R_vec, float)

    m = np.isfinite(lam) & np.isfinite(f) & np.isfinite(R) & (R > 0)
    if np.count_nonzero(m) < 5:
        return flux.copy()

    z  = np.log(np.clip(lam[m], 1e-12, None))
    fm = f[m]; Rm = R[m]
    out = np.empty_like(fm)
    for i, zi in enumerate(z):
        sigma_i = 1.0 / (2.355 * max(Rm[i], 1e-6))
        span    = 5.0 * sigma_i
        dz      = z - zi
        mask    = (dz >= -span) & (dz <= +span)
        ker     = np.exp(-0.5 * (dz[mask] / max(sigma_i, 1e-15))**2)
        num     = np.nansum(ker * fm[mask])
        den     = np.nansum(ker)
        out[i]  = (num / den) if den > 0 else fm[i]

    g = flux.copy()
    g[m] = out
    return g

def _make_R_auto_from_channels(lambdaEff: np.ndarray,
                               dx_low_um: np.ndarray | None,
                               dx_high_um: np.ndarray | None,
                               *, floor_R: float | None = None,
                               smooth_R: float | None = 600.0,
                               beta: float = 1.0):
    """
    Build R(λ) directly from your channel widths:
        R(λ) ~ beta * λ / Δλ_channel(λ)
    where Δλ_channel = dx_high - dx_low (per channel).
    A small low-pass in log λ (smooth_R) keeps R(λ) smooth across channels.

    Returns: a function R_of_lam(lam_um) that works for scalars or arrays.
    """
    lam = np.asarray(lambdaEff, float)
    if (dx_low_um is None) or (dx_high_um is None):
        if floor_R is None or not np.isfinite(floor_R) or (floor_R <= 0):
            raise ValueError("R_auto requires channel borders (dx_low/high) or a valid floor_R.")
        dlam = lam / float(floor_R)
    else:
        dlo  = np.asarray(dx_low_um,  float)
        dhi  = np.asarray(dx_high_um, float)
        dlam = np.clip(dhi - dlo, 1e-15, np.inf)

    if (floor_R is not None) and np.isfinite(floor_R) and (floor_R > 0):
        dlam = np.maximum(dlam, lam / float(floor_R))

    R_cent = (float(beta) * lam) / np.maximum(dlam, 1e-15)

    if (smooth_R is not None) and (smooth_R > 0):
        R_cent = _blur_constant_R_loglambda(lam, R_cent, R=float(smooth_R))

    lam_ref = lam.copy()
    R_ref   = R_cent.copy()

    def R_of_lam(lu):
        z = np.asarray(lu, float)
        return np.interp(z, lam_ref, R_ref, left=R_ref[0], right=R_ref[-1])

    return R_of_lam

# ==========================================================
# SED providers
# ==========================================================
def _planck_lambda_um(lam_um: np.ndarray, T: float) -> np.ndarray:
    lam_m = np.asarray(lam_um, float) * 1e-6
    h = 6.62607015e-34; c = 2.99792458e8; k = 1.380649e-23
    a = 2*h*c*c; b = h*c/(k*T)
    with np.errstate(over='ignore', invalid='ignore'):
        num = a/(lam_m**5); den = np.expm1(b/lam_m)
        out = num/den
    return out

def _phoenix_continuum_like(lam_um: np.ndarray, T: float, R_continuum: float) -> np.ndarray:
    S_full = phoenix_flux_interp_um(lam_um, T)
    lam_nm = lam_um * 1000.0
    return _gauss_blur(lam_nm, S_full, R_continuum)

def _get_sed_array(lam_um: np.ndarray, T: float, mode: str, R_continuum: float,
                   hybrid_knobs: dict | None) -> np.ndarray:
    mode = str(mode).lower().strip()
    if mode == "phoenix":
        return phoenix_flux_interp_um(lam_um, T)
    elif mode == "phoenix_continuum":
        return _phoenix_continuum_like(lam_um, T, R_continuum)
    elif mode == "phoenix_continuum_hybrid_global":
        hk = hybrid_knobs or {}
        def _kn(k, d):
            v = hk.get(k, None)
            try: return d if (v is None or not np.isfinite(float(v))) else float(v)
            except Exception: return d
        try:
            # >>> Patch: cap_env default = None (sem teto) <<<
            return phoenix_continuum_hybrid_global(
                lam_um, T=T, feh=0.0,
                R_blur=_kn("R_blur", R_continuum),
                q=_kn("q", 0.985),
                Rwin_quant=_kn("Rwin_quant", 120.0),
                Rwin_close=_kn("Rwin_close", 300.0),
                bins_per_R=_kn("bins_per_R", 3.0),
                post_gauss_um=_kn("post_gauss_um", 0.040),
                beta_env=_kn("beta_env", 0.95),
                cap_env=_kn("cap_env", None),   # <<< aqui
                alpha_mix=_kn("alpha_mix", 0.60),
                R_post=_kn("R_post", None),
            )
        except Exception as e:
            print(f"[HYBRID WARN] fallback to continuum: {e}")
            return _phoenix_continuum_like(lam_um, T, R_continuum)
    elif mode == "phoenix_continuum_hybrid_global_tamed":
        hk = hybrid_knobs or {}
        def _kn(k, d):
            v = hk.get(k, None)
            try: return d if (v is None or not np.isfinite(float(v))) else float(v)
            except Exception: return d
        try:
            # >>> Patch: cap_env default = None (sem teto) <<<
            return phoenix_continuum_hybrid_global_tamed(
                lam_um, T=T, feh=0.0,
                ripple_alpha=_kn("ripple_alpha", 1.0),
                R_blur=_kn("R_blur", R_continuum),
                q=_kn("q", 0.985),
                Rwin_quant=_kn("Rwin_quant", 120.0),
                Rwin_close=_kn("Rwin_close", 300.0),
                bins_per_R=_kn("bins_per_R", 3.0),
                post_gauss_um=_kn("post_gauss_um", 0.040),
                beta_env=_kn("beta_env", 0.95),
                cap_env=_kn("cap_env", None),   # <<< aqui
                alpha_mix=_kn("alpha_mix", 0.60),
                R_post=_kn("R_post", None),
            )
        except Exception as e:
            print(f"[HYBRID-TAMED WARN] fallback to continuum: {e}")
            return _phoenix_continuum_like(lam_um, T, R_continuum)
    elif mode == "bb":
        return _planck_lambda_um(lam_um, T)
    else:
        print(f"[WARN] Unknown SED mode '{mode}', using PHOENIX full.")
        return phoenix_flux_interp_um(lam_um, T)

# ==========================================================
# Segment cache
# ==========================================================
_S_CACHE = {}
def _knobs_hash(hk: dict | None) -> tuple:
    if not hk: return ()
    keys = sorted(hk.keys())
    return tuple((k, float(hk[k]) if hk[k] is not None else None) for k in keys)

def _sed_segment(mode: str, T: float,
                 lam_lo_um: float, lam_hi_um: float, nsub: int,
                 R_continuum: float, hybrid_knobs: dict | None):
    key = (str(mode).lower().strip(),
           float(T), float(R_continuum),
           float(lam_lo_um), float(lam_hi_um), int(nsub),
           _knobs_hash(hybrid_knobs))
    if key in _S_CACHE:
        return _S_CACHE[key]
    lam_um = np.linspace(lam_lo_um, lam_hi_um, max(61, int(nsub)))
    S = _get_sed_array(lam_um, T, R_continuum=R_continuum, mode=mode, hybrid_knobs=hybrid_knobs)
    _S_CACHE[key] = (lam_um, S)
    return lam_um, S

# ==========================================================
# Broadening (rotation & macroturbulence)
# ==========================================================
def _rotational_kernel_gray(v_kms: np.ndarray, vsini_kms: float, epsilon: float) -> np.ndarray:
    v = np.asarray(v_kms, float)
    V = float(max(vsini_kms, 1e-9))
    x = np.clip(v / V, -1.0, 1.0)
    mu = np.sqrt(np.maximum(0.0, 1.0 - x*x))
    eps = float(np.clip(epsilon, 0.0, 1.0))
    num = (2.0 * (1.0 - eps) * mu) + (0.5 * np.pi * eps * (1.0 - x*x))
    denom = np.pi * V * (1.0 - eps/3.0)
    g = np.where(np.abs(v) <= V, num / max(denom, EPS), 0.0)
    s = float(np.trapz(g, v)); s = s if (np.isfinite(s) and s > 0) else 1.0
    return g / s

def _gaussian_kernel_v(v_kms: np.ndarray, sigma_kms: float) -> np.ndarray:
    v = np.asarray(v_kms, float)
    sig = float(max(sigma_kms, 1e-9))
    g = np.exp(-0.5 * (v/sig)**2)
    s = float(np.trapz(g, v)); s = s if (np.isfinite(s) and s > 0) else 1.0
    return g / s

def _build_velocity_kernel(vsini_kms: float, epsilon: float, macro_kms: float,
                           dv_kms: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    Vrot = float(max(vsini_kms, 0.0))
    Vmac = float(max(macro_kms, 0.0))
    half_span = max(8.0*max(Vmac, 1e-9), max(1.5*Vrot, 0.0)) + 5.0
    half_span = max(5.0, half_span)
    dv = float(max(0.05, min(dv_kms, half_span/400.0)))
    v = np.arange(-half_span, half_span + dv/2, dv, dtype=float)

    ker = np.ones_like(v)
    if Vrot > 0:
        ker = _rotational_kernel_gray(v, Vrot, epsilon)
    if Vmac > 0:
        g = _gaussian_kernel_v(v, Vmac)
        ker = np.convolve(ker, g, mode="same")
        s = float(np.trapz(ker, v)); s = s if (np.isfinite(s) and s > 0) else 1.0
        ker = ker / s
    return v, ker

def _doppler_convolve_loglambda(lam_um: np.ndarray, flux: np.ndarray,
                                vgrid_kms: np.ndarray, k_v: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam_um, float); f = np.asarray(flux, float)
    m = np.isfinite(lam) & np.isfinite(f)
    if np.count_nonzero(m) < 5:
        return flux.copy()

    z = np.log(np.clip(lam[m], 1e-12, None))
    z_min, z_max = float(z.min()), float(z.max())

    dv_target = max(0.05, min(0.25, (np.abs(vgrid_kms).max() or 5.0)/800.0))
    dz = dv_target / C_KMS
    N = int(np.ceil((z_max - z_min)/dz)) + 1
    z_grid = np.linspace(z_min, z_max, max(N, 256))
    dz = float(z_grid[1] - z_grid[0])

    f_grid = np.interp(z_grid, z, f[m])

    z_k = vgrid_kms / C_KMS
    z_k_min, z_k_max = float(z_k.min()), float(z_k.max())
    half = int(np.ceil(max(abs(z_k_min), abs(z_k_max)) / dz))
    zker = np.arange(-half, half+1) * dz
    k_idx = np.interp(zker, z_k, k_v, left=0.0, right=0.0)
    k_idx = np.clip(k_idx, 0.0, np.max(k_idx) if np.any(k_idx>0) else 1.0)
    if np.sum(k_idx) > 0:
        k_idx /= (np.sum(k_idx) or 1.0)

    y = np.convolve(np.pad(f_grid, (half, half), mode="edge"), k_idx, mode="same")[half:-half]
    return np.interp(np.log(lam), z_grid, y)

# ==========================================================
# Zeeman (placeholder)
# ==========================================================
def _zeeman_delta_lambda_A(lam_A: np.ndarray, B_kG: float, g_eff: float) -> np.ndarray:
    """Δλ_B (Å) ≈ 4.67e-13 * g_eff * λ^2(Å) * B(G);  1 kG = 1000 G."""
    if B_kG is None or g_eff is None:
        return np.zeros_like(lam_A, float)
    B_G = float(B_kG) * 1000.0
    return 4.67e-13 * float(g_eff) * (lam_A**2) * B_G

def _apply_zeeman_triplet(lam_um: np.ndarray, flux: np.ndarray,
                          B_kG: float, g_eff: float, pi_frac: float) -> np.ndarray:
    """
    Unpolarized, toy triplet: π, σ−, σ+. If B==0 → returns flux unchanged.
    """
    lam_um = np.asarray(lam_um, float); F = np.asarray(flux, float)
    if (B_kG is None) or (B_kG <= 0) or (g_eff is None) or (g_eff <= 0):
        return F.copy()
    lam_A = lam_um * 1e4
    dA = _zeeman_delta_lambda_A(lam_A, B_kG=float(B_kG), g_eff=float(g_eff))
    d_um = dA * 1e-4

    fpi = float(np.clip(pi_frac if pi_frac is not None else 0.5, 0.0, 1.0))
    fsig = 1.0 - fpi
    Fm = np.interp(lam_um, lam_um - d_um, F, left=F[0], right=F[-1])
    Fp = np.interp(lam_um, lam_um + d_um, F, left=F[0], right=F[-1])
    Fz = fpi * F + 0.5 * fsig * (Fm + Fp)
    return Fz

# ==========================================================
# Channel-wise r(λ)
# ==========================================================
def _ratio_channel_integrate_seds(center_um: float, dx_low_um, dx_high_um, R_fallback: float, nsub: int,
                                  T_active: float, T_phot: float, thr_pair, R_preconv_override,
                                  sed_mode_active: str, sed_mode_phot: str, R_continuum: float,
                                  channel_window_kind: str = "tukey", channel_tukey_alpha: float = 0.30,
                                  hybrid_knobs_active: dict | None = None,
                                  hybrid_knobs_phot: dict | None = None,
                                  vsini_kms: float = 0.0, rot_ld_epsilon: float = 0.6, macro_kms: float = 0.0,
                                  anchor_continuum: bool = False,
                                  anchor_q: float = 0.985,
                                  anchor_bins_per_R: float = 4.0,
                                  anchor_Rwin_quant: float = 120.0,
                                  anchor_Rwin_close: float = 400.0,
                                  anchor_post_gauss_um: float = 0.06,
                                  # Zeeman (optional)
                                  zeeman_enable: bool = False,
                                  zeeman_g_eff: float = 1.0,
                                  zeeman_pi_frac: float = 0.5,
                                  zeeman_B_phot_kG: float = 0.0,
                                  zeeman_B_active_kG: float = 0.0,
                                  ) -> float:
    have_borders = (dx_low_um is not None and dx_high_um is not None and
                    np.isfinite(dx_low_um) and np.isfinite(dx_high_um))
    if have_borders:
        lam_lo_um = float(center_um + dx_low_um)
        lam_hi_um = float(center_um + dx_high_um)
    else:
        half = float(center_um)/max(float(R_fallback), EPS)/2.0
        lam_lo_um = float(center_um) - half
        lam_hi_um = float(center_um) + half

    if not (np.isfinite(lam_lo_um) and np.isfinite(lam_hi_um) and lam_hi_um > lam_lo_um):
        Sph = _get_sed_array(np.array([center_um]), T_phot,   sed_mode_phot,   R_continuum, hybrid_knobs_phot)[0]
        Sac = _get_sed_array(np.array([center_um]), T_active, sed_mode_active, R_continuum, hybrid_knobs_active)[0]
        denom = Sph if (np.isfinite(Sph) and abs(Sph) > EPS) else EPS
        r = float(Sac/denom)
        return r if np.isfinite(r) and r > 0 else 1.0

    def _hk_get(hk, k, d):
        try:
            return float(hk.get(k, d)) if (hk is not None and hk.get(k, None) is not None) else float(d)
        except Exception:
            return float(d)

    Rwin_q_act  = _hk_get(hybrid_knobs_active, "Rwin_quant", 120.0)
    Rwin_q_phot = _hk_get(hybrid_knobs_phot,   "Rwin_quant", 120.0)
    binsR_act   = _hk_get(hybrid_knobs_active, "bins_per_R", 3.0)
    binsR_phot  = _hk_get(hybrid_knobs_phot,   "bins_per_R", 3.0)

    pad_um = 4.0 * float(center_um) / max(min(Rwin_q_act, Rwin_q_phot), 1e-6)
    seg_lo = max(1e-9, lam_lo_um - pad_um)
    seg_hi =           lam_hi_um + pad_um

    width_R = (seg_hi - seg_lo) / max(float(center_um), 1e-12) * max(Rwin_q_act, Rwin_q_phot)
    n_need  = int(np.ceil(width_R * max(binsR_act, binsR_phot) * 2.0))
    n_seg   = max(int(nsub), n_need, 1201)

    lam_um, S_ph_raw = _sed_segment(sed_mode_phot,   T_phot,   seg_lo, seg_hi, n_seg, R_continuum, hybrid_knobs_phot)
    _,      S_ac_raw = _sed_segment(sed_mode_active, T_active, seg_lo, seg_hi, n_seg, R_continuum, hybrid_knobs_active)
    S_ph = np.array(S_ph_raw, dtype=float, copy=True)
    S_ac = np.array(S_ac_raw, dtype=float, copy=True)

    # Optional Zeeman before anchor/blur
    if zeeman_enable and ( (zeeman_B_phot_kG and zeeman_B_phot_kG>0) or (zeeman_B_active_kG and zeeman_B_active_kG>0) ):
        S_ph = _apply_zeeman_triplet(lam_um, S_ph, B_kG=float(zeeman_B_phot_kG or 0.0),
                                     g_eff=float(zeeman_g_eff or 1.0), pi_frac=float(zeeman_pi_frac or 0.5))
        S_ac = _apply_zeeman_triplet(lam_um, S_ac, B_kG=float(zeeman_B_active_kG or 0.0),
                                     g_eff=float(zeeman_g_eff or 1.0), pi_frac=float(zeeman_pi_frac or 0.5))

    if anchor_continuum:
        mix = 0.5*(S_ph + S_ac)
        env_unit = super_continuum_envelope(
            lam_um, mix,
            Rwin_quant=anchor_Rwin_quant,
            bins_per_R=anchor_bins_per_R,
            q=anchor_q,
            Rwin_close=anchor_Rwin_close,
            post_gauss_um=anchor_post_gauss_um
        )
        env_unit = np.clip(env_unit, 1e-9, np.nanmax(env_unit) or 1.0)
        S_ph = S_ph / env_unit
        S_ac = S_ac / env_unit

    w_thr = _throughput_at(lam_um, thr_pair)

    if (vsini_kms and vsini_kms > 0) or (macro_kms and macro_kms > 0):
        vgrid, k_v = _build_velocity_kernel(vsini_kms, rot_ld_epsilon, macro_kms, dv_kms=0.2)
        S_ph = _doppler_convolve_loglambda(lam_um, S_ph, vgrid, k_v)
        S_ac = _doppler_convolve_loglambda(lam_um, S_ac, vgrid, k_v)

    # ---- Instrument pre-convolution ----
    R_override = R_preconv_override
    R_vec = None
    R_const = None

    if callable(R_override):
        out = R_override(lam_um)     # may return scalar or array
        if np.ndim(out) == 0:
            try:
                R_const = float(out)
            except Exception:
                R_const = None
        else:
            R_vec = np.asarray(out, float)
    elif isinstance(R_override, (list, tuple, np.ndarray)):
        arr = np.asarray(R_override, float)
        if arr.size == lam_um.size:
            R_vec = arr
        else:
            try:
                R_const = float(np.nanmedian(arr))
            except Exception:
                R_const = None
    else:
        try:
            if (R_override is not None) and np.isfinite(float(R_override)) and (float(R_override) > 0):
                R_const = float(R_override)
        except Exception:
            R_const = None

    if R_vec is not None:
        S_ph = _blur_variable_R_loglambda(lam_um, S_ph, R_vec)
        S_ac = _blur_variable_R_loglambda(lam_um, S_ac, R_vec)
        R_ph = R_ac = None
    elif (R_const is not None):
        S_ph = _blur_constant_R_loglambda(lam_um, S_ph, R_const)
        S_ac = _blur_constant_R_loglambda(lam_um, S_ac, R_const)
        R_ph = R_ac = None
    else:
        # infer R from the channel width only if using mixed SED modes
        mode_act = str(sed_mode_active).lower().strip()
        mode_pho = str(sed_mode_phot).lower().strip()
        both_phx_full = (mode_act == "phoenix") and (mode_pho == "phoenix")
        both_phx_cont = (mode_act == "phoenix_continuum") and (mode_pho == "phoenix_continuum")
        is_hybrid = lambda m: m in ("phoenix_continuum_hybrid_global",
                                    "phoenix_continuum_hybrid_global_tamed")
        both_hybrid = is_hybrid(mode_act) and is_hybrid(mode_pho)
        if both_phx_full or both_phx_cont or both_hybrid:
            R_ph = R_ac = None
        else:
            R_chan = float(center_um) / max(lam_hi_um - lam_lo_um, EPS)
            R_ph = R_ac = R_chan

    if (R_ph is not None) and np.isfinite(R_ph) and (R_ph > 0):
        lam_nm = lam_um * 1000.0
        S_ph = _gauss_blur(lam_nm, S_ph, R_ph)
    if (R_ac is not None) and np.isfinite(R_ac) and (R_ac > 0):
        lam_nm = lam_um * 1000.0
        S_ac = _gauss_blur(lam_nm, S_ac, R_ac)

    # Channel window (normalized)
    if channel_window_kind.lower().strip() == "tukey":
        w_ch = _tukey_window(lam_um, lam_lo_um, lam_hi_um, alpha=channel_tukey_alpha)
    else:
        w_ch = np.where((lam_um>=lam_lo_um) & (lam_um<=lam_hi_um), 1.0, 0.0)
        s = float(np.trapz(w_ch, lam_um)); s = s if (np.isfinite(s) and s > 0) else 1.0
        w_ch = w_ch / s

    w_eff = w_thr * w_ch
    good = np.isfinite(S_ph) & np.isfinite(S_ac) & np.isfinite(w_eff) & (w_eff > 0)
    if not np.any(good):
        return 1.0

    F_ph = float(np.trapz(S_ph[good] * w_eff[good], lam_um[good]))
    F_ac = float(np.trapz(S_ac[good] * w_eff[good], lam_um[good]))
    if (not np.isfinite(F_ph)) or (abs(F_ph) <= EPS):
        return 1.0
    r = float(F_ac / F_ph)
    return r if np.isfinite(r) and r > 0 else 1.0

# ==========================================================
# r(λ) smoothers
# ==========================================================
def _smooth_contrast_curve(lam_um, r_in, R_lp=220.0, gamma=1.0):
    r = np.asarray(r_in, float)
    r = np.where(np.isfinite(r) & (r > 0), r, 1.0)
    log_r = np.log(r)
    log_r_lp = _blur_constant_R_loglambda(lam_um, log_r, R=max(1.0, float(R_lp)))
    hf = np.exp(log_r - log_r_lp)
    r_out = np.exp(log_r_lp) * (np.clip(hf, 1e-6, 1e6) ** float(gamma))
    return r_out

def _soft_roll_mask(nbins: int, nyq: int, roll: float = 0.35, steep: float = 10.0) -> np.ndarray:
    k = np.arange(nbins, dtype=float)
    k0 = max(1.0, float(roll) * float(nyq))
    x = k / k0
    m = 1.0 / (1.0 + np.power(np.clip(x, 0.0, np.inf), float(steep)))
    return m

def _smooth_contrast_fft_energy(lam_um: np.ndarray, r_in: np.ndarray,
                                roll: float = 0.35,
                                gain_floor: float = 0.25,
                                energy: bool = True,
                                energy_exp: float = 0.70,
                                energy_mix: float = 0.65) -> np.ndarray:
    lam = np.asarray(lam_um, float)
    r   = np.asarray(r_in,   float)
    r   = np.where(np.isfinite(r) & (r > 0), r, 1.0)
    z   = np.log(np.clip(lam, 1e-12, None))

    N0 = max(128, len(lam) * 2)
    N  = 1 << int(np.ceil(np.log2(N0)))
    z_grid = np.linspace(float(z.min()), float(z.max()), N)
    logr   = np.log(r)
    logr_grid = np.interp(z_grid, z, logr)

    R = np.fft.rfft(logr_grid)
    nb = R.size
    nyq = nb - 1

    roll_mask = _soft_roll_mask(nb, nyq, roll=float(roll), steep=10.0)

    if energy:
        P = np.abs(R)**2
        w = max(5, int(0.015 * nb))
        ker = np.ones(w, float) / float(w)
        P_s = np.convolve(np.pad(P, (w//2, w//2), mode='edge'), ker, mode='same')[w//2:-w//2]
        if P_s.size != nb:
            xsrc = np.linspace(0.0, 1.0, P_s.size)
            xdst = np.linspace(0.0, 1.0, nb)
            P_s  = np.interp(xdst, xsrc, P_s)
        Pn = P_s / (np.nanmax(P_s) or 1.0)

        A_energy = np.power(1.0 - np.clip(Pn, 0.0, 1.0), float(energy_exp))
        A_energy = np.clip(A_energy, 0.0, 1.0)
        A_energy[0]  = 1.0
        if nb > 1:
            A_energy[-1] = max(A_energy[-1], 0.7)

        mix = float(np.clip(energy_mix, 0.0, 1.0))
        M = roll_mask * ((1.0 - mix) + mix * A_energy)
    else:
        M = roll_mask

    M = np.clip(M, float(gain_floor), 1.0)
    M[0] = 1.0

    Rf = R * M
    yf = np.fft.irfft(Rf, n=N)
    y_out = np.interp(np.log(lam), z_grid, yf)
    r_out = np.exp(y_out)
    r_out = np.where(np.isfinite(r_out) & (r_out > 0), r_out, 1.0)
    return r_out

# ==========================================================
# λ-dependent scalers (FIX for NameError)
# ==========================================================
def _build_scaler(spec, default_scale):
    """
    Returns a function s(lam_um) that yields a wavelength-dependent scale.
    - spec = None or scalar: constant scale
    - spec = {'kind':'powerlaw','amp':...,'pivot_um':...,'alpha':...}
    - spec = {'kind':'table','lam_um':[...],'scale':[...]}"""
    try:
        if spec is None or np.isscalar(spec):
            c = default_scale if spec is None else float(spec)
            return lambda lam_um: float(c)
    except Exception:
        pass
    if isinstance(spec, dict):
        kind = str(spec.get('kind', 'table')).lower().strip()
        if kind == 'powerlaw':
            amp = float(spec.get('amp', default_scale))
            pivot = float(spec.get('pivot_um', 1.0))
            alpha = float(spec.get('alpha', 0.0))
            pivot = pivot if np.isfinite(pivot) and pivot > 0 else 1.0
            return lambda lam_um: float(amp * (max(lam_um, 1e-9)/pivot)**alpha)
        elif kind == 'table':
            lam = np.asarray(spec.get('lam_um', []), float)
            scl = np.asarray(spec.get('scale', []), float)
            if lam.size >= 2 and scl.size == lam.size:
                lam_ord = np.argsort(lam)
                lam_s = lam[lam_ord]; scl_s = scl[lam_ord]
                def _f(lu):
                    return float(np.interp(lu, lam_s, scl_s, left=scl_s[0], right=scl_s[-1]))
                return _f
    c = default_scale
    return lambda lam_um: float(c)

# ==========================================================
# Planck-trend substitution (optional)
# ==========================================================
def _bb_trend_substitute(lam_um, r_in, T_active, T_phot, R_lp=260.0):
    lam = np.asarray(lam_um, float)
    r   = np.asarray(r_in,   float)
    r   = np.where(np.isfinite(r) & (r > 0), r, 1.0)

    log_r   = np.log(r)
    log_lpf = _blur_constant_R_loglambda(lam, log_r, R=max(1.0, float(R_lp)))

    Bb = _planck_lambda_um(lam, float(T_active)) / np.maximum(_planck_lambda_um(lam, float(T_phot)), EPS)
    Bb = np.clip(Bb, 1e-12, 1e12)
    log_bb = np.log(Bb)

    delta = float(np.nanmedian(log_lpf - log_bb))
    log_trend_ref = log_bb + delta

    log_r_new = log_trend_ref + (log_r - log_lpf)
    r_new = np.exp(log_r_new)
    r_new = np.where(np.isfinite(r_new) & (r_new > 0), r_new, 1.0)
    return r_new

# ==========================================================
# Main program
# ==========================================================
class MainProgram:
    def __init__(self, *,
                 target, num_elements, profile, c1, c2, c3, c4, lambdaEff,
                 intensidadeMaxima, raioStar, ecc, anom, tempStar,
                 lat_spot=None, long_spot=None, quantidade_spot=None,
                 lat_fac=None, long_fac=None, quantidade_fac=None,
                 starspots=True, quantidade=1, lat=None, longt=None,
                 r=None, r_spot=None, r_facula=None,
                 semiEixoUA=0.1, massStar=0.2, plot_anim=False, periodo=10.0, anguloInclinacao=89.0,
                 raioPlanetaRj=0.1, plot_graph=False, plot_star=False,
                 tempSpot=None, tempFacula=None, fillingFactor=(0.0,),
                 min_pixels=256, max_pixels=1024, pixels_per_rp=50,
                 simulation_mode="unspotted", both_mode=False,
                 dx_low_um=None, dx_high_um=None, R_fallback=20.0, nsub_obs=401,
                 use_throughput=False, throughput_csv=None,
                 sed_mode_active="phoenix", sed_mode_phot="phoenix",
                 R_continuum=120.0,
                 facula_contrast_scale: float = DEFAULT_FACULA_CONTRAST_SCALE,
                 spot_contrast_scale:   float = DEFAULT_SPOT_CONTRAST_SCALE,
                 facula_contrast_spec: dict | float | None = None,
                 spot_contrast_spec:   dict | float | None = None,
                 R_preconv_override=None,
                 # NEW: auto-R knobs
                 R_auto_floor_R: float | None = None,
                 R_auto_smooth_R: float | None = 600.0,
                 R_auto_beta: float = 1.0,
                 # Broadening
                 vsini_kms: float = 0.0, rot_ld_epsilon: float = 0.6, macro_kms: float = 0.0,
                 # CLV & window
                 beta_facula: float = 0.0, beta_spot: float = 0.0,
                 channel_window_kind: str = "tukey",
                 channel_tukey_alpha: float = 0.30,
                 # Debug
                 debug_plots: bool = True, debug_every: int = 1, debug_max: int | None = 100,
                 debug_save_png: bool = True, debug_outdir: str = "debug_plots",
                 # Hybrid knobs (global defaults)
                 hybrid_R_blur: float | None = None,
                 hybrid_q: float | None = None,
                 hybrid_Rwin_quant: float | None = None,
                 hybrid_Rwin_close: float | None = None,
                 hybrid_bins_per_R: float | None = None,
                 hybrid_post_gauss_um: float | None = None,
                 hybrid_beta_env: float | None = None,
                 hybrid_cap_env: float | None = None,
                 hybrid_alpha_mix: float | None = None,
                 hybrid_R_post: float | None = None,
                 hybrid_ripple_alpha: float | None = None,
                 # r(λ) smoothing
                 contrast_enable: bool = False,
                 contrast_lowpass_R: float | None = 220.0,
                 contrast_hf_gamma: float = 0.33,
                 contrast_fft: bool = False,
                 contrast_fft_roll: float = 0.35,
                 contrast_fft_gain_floor: float = 0.25,
                 contrast_fft_energy: bool = True,
                 contrast_fft_energy_exp: float = 0.70,
                 contrast_fft_energy_mix: float = 0.65,
                 plot_sanity: bool = False,
                 abort_on_occultation: bool = True,
                 # per-component overrides
                 hybrid_R_blur_spot: float | None = None,
                 hybrid_R_blur_facula: float | None = None,
                 hybrid_ripple_alpha_spot: float | None = None,
                 hybrid_ripple_alpha_facula: float | None = None,
                 hybrid_R_post_spot: float | None = None,
                 hybrid_R_post_facula: float | None = None,
                 contrast_lowpass_R_spot: float | None = None,
                 contrast_lowpass_R_facula: float | None = None,
                 contrast_hf_gamma_spot: float | None = None,
                 contrast_hf_gamma_facula: float | None = None,
                 contrast_fft_spot: bool | None = None,
                 contrast_fft_facula: bool | None = None,
                 contrast_fft_roll_spot: float | None = None,
                 contrast_fft_roll_facula: float | None = None,
                 contrast_fft_gain_floor_spot: float | None = None,
                 contrast_fft_gain_floor_facula: float | None = None,
                 contrast_fft_energy_spot: bool | None = None,
                 contrast_fft_energy_facula: bool | None = None,
                 contrast_fft_energy_exp_spot: float | None = None,
                 contrast_fft_energy_exp_facula: float | None = None,
                 contrast_fft_energy_mix_spot: float | None = None,
                 contrast_fft_energy_mix_facula: float | None = None,
                 contrast_r_floor_spot: float | None = None,
                 contrast_r_floor_facula: float | None = None,
                 # anchor continuum
                 anchor_continuum: bool = False,
                 anchor_q: float = 0.985,
                 anchor_bins_per_R: float = 4.0,
                 anchor_Rwin_quant: float = 120.0,
                 anchor_Rwin_close: float = 400.0,
                 anchor_post_gauss_um: float = 0.06,
                 # LDC zeroing
                 ldc_zero: bool = False,
                 # Planck-trend
                 trend_fix_enable: bool = True,
                 trend_R_lp: float = 50.0,
                 # HF boost
                 hf_boost_enable: bool = False,
                 hf_boost_alpha: float = 0.8,
                 hf_boost_max: float = 1.8,
                 hf_boost_lambda_ref: float = 1.0,
                 # Zeeman placeholders
                 zeeman_enable: bool = False,
                 zeeman_g_eff: float = 1.0,
                 zeeman_pi_frac: float = 0.5,
                 zeeman_B_quiet_kG: float = 0.0,
                 zeeman_B_spot_kG: float = 0.0,
                 zeeman_B_facula_kG: float = 0.0,
                 ):
        # --- basic init ---
        self.target = target
        self.num_elements = int(num_elements)
        self.profile = profile
        self.ldc_zero = bool(ldc_zero)
        c1 = np.asarray(c1, float); c2 = np.asarray(c2, float)
        c3 = np.asarray(c3, float); c4 = np.asarray(c4, float)
        if self.ldc_zero:
            self.c1 = np.zeros_like(c1); self.c2 = np.zeros_like(c2)
            self.c3 = np.zeros_like(c3); self.c4 = np.zeros_like(c4)
        else:
            self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4

        self.lambdaEff = np.asarray(lambdaEff, float)
        self.intensidadeMaxima= float(intensidadeMaxima)
        self.raioStar = float(raioStar)
        self.ecc = float(ecc)
        self.anom = float(anom)
        self.tempStar = float(np.clip(tempStar, T_MIN_PHX, T_MAX_PHX))

        self.lat_spot = np.asarray(lat_spot, float) if lat_spot is not None else (np.asarray(lat, float) if lat is not None else np.array([20.0]))
        self.long_spot = np.asarray(long_spot, float) if long_spot is not None else (np.asarray(longt, float) if longt is not None else np.array([+30.0]))
        self.lat_fac = np.asarray(lat_fac, float) if lat_fac is not None else (np.asarray(lat, float) if lat is not None else np.array([20.0]))
        self.long_fac = np.asarray(long_fac, float) if long_fac is not None else (np.asarray(longt, float) if longt is not None else np.array([-30.0]))
        self.quantidade_spot = int(quantidade_spot if quantidade_spot is not None else quantidade)
        self.quantidade_fac  = int(quantidade_fac  if quantidade_fac  is not None else quantidade)

        self.r = r
        self.r_spot   = r_spot   if r_spot   is not None else r
        self.r_facula = r_facula if r_facula is not None else r

        self.semiEixoUA = float(semiEixoUA)
        self.massStar   = float(massStar)
        self.plot_anim  = bool(plot_anim)
        self.periodo    = float(periodo)
        self.anguloInclinacao = float(anguloInclinacao)
        self.raioPlanetaRj = float(raioPlanetaRj)
        self.plot_graph = bool(plot_graph)
        self.plot_star  = bool(plot_star)

        def _clip_temp(x): return None if x is None else float(np.clip(x, T_MIN_PHX, T_MAX_PHX))
        self.tempSpot   = _clip_temp(tempSpot)
        self.tempFacula = _clip_temp(tempFacula)

        self.fillingFactor = (fillingFactor if isinstance(fillingFactor, (list, tuple, np.ndarray)) else (float(fillingFactor),))
        self.simulation_mode = str(simulation_mode).lower().strip()

        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.pixels_per_rp = float(pixels_per_rp)

        self.both_mode = bool(both_mode)

        self.dx_low_um  = (None if dx_low_um  is None else np.asarray(dx_low_um,  float))
        self.dx_high_um = (None if dx_high_um is None else np.asarray(dx_high_um, float))
        self.R_fallback = float(R_fallback)
        self.nsub_obs   = int(nsub_obs)
        self.thr_pair   = (_load_throughput_csv(throughput_csv) if use_throughput else None)

        self.facula_contrast_scale = float(facula_contrast_scale)
        self.spot_contrast_scale   = float(spot_contrast_scale)
        self._fac_scale_fn = _build_scaler(facula_contrast_spec, self.facula_contrast_scale)
        self._spot_scale_fn = _build_scaler(spot_contrast_spec,   self.spot_contrast_scale)

        self.R_preconv_override    = R_preconv_override
        self.R_auto_floor_R  = (None if R_auto_floor_R  is None else float(R_auto_floor_R))
        self.R_auto_smooth_R = (None if R_auto_smooth_R is None else float(R_auto_smooth_R))
        self.R_auto_beta     = float(R_auto_beta)

        self.vsini_kms       = float(max(vsini_kms, 0.0))
        self.rot_ld_epsilon  = float(np.clip(rot_ld_epsilon, 0.0, 1.0))
        self.macro_kms       = float(max(macro_kms, 0.0))

        self.sed_mode_active = str(sed_mode_active).lower().strip()
        self.sed_mode_phot   = str(sed_mode_phot).lower().strip()
        self.R_continuum     = float(R_continuum)

        self.beta_facula = float(beta_facula)
        self.beta_spot   = float(beta_spot)

        self.channel_window_kind = str(channel_window_kind).lower().strip()
        self.channel_tukey_alpha = float(channel_tukey_alpha)

        # --- Hybrid knobs (global) ---
        self.hybrid_knobs_active = {
            k: v for k, v in {
                "R_blur":        hybrid_R_blur,
                "q":             hybrid_q,
                "Rwin_quant":    hybrid_Rwin_quant,
                "Rwin_close":    hybrid_Rwin_close,
                "bins_per_R":    hybrid_bins_per_R,
                "post_gauss_um": hybrid_post_gauss_um,
                "beta_env":      hybrid_beta_env,
                "cap_env":       hybrid_cap_env,
                "alpha_mix":     hybrid_alpha_mix,
                "R_post":        hybrid_R_post,
                "ripple_alpha":  hybrid_ripple_alpha,
            }.items() if v is not None
        }
        self.hybrid_knobs_phot = dict(self.hybrid_knobs_active)

        def _merge_knobs(base: dict, **ovr):
            d = dict(base)
            for k, v in ovr.items():
                if v is not None: d[k] = v
            return {k: v for k, v in d.items() if v is not None}

        self.hybrid_knobs_spot = _merge_knobs(
            self.hybrid_knobs_active,
            R_blur=hybrid_R_blur_spot,
            ripple_alpha=hybrid_ripple_alpha_spot,
            R_post=hybrid_R_post_spot,
        )
        self.hybrid_knobs_facula = _merge_knobs(
            self.hybrid_knobs_active,
            R_blur=hybrid_R_blur_facula,
            ripple_alpha=hybrid_ripple_alpha_facula,
            R_post=hybrid_R_post_facula,
        )

        # --- r(λ) smoothing knobs ---
        self.contrast_enable    = bool(contrast_enable)
        self.contrast_lowpass_R = (None if contrast_lowpass_R is None else float(contrast_lowpass_R))
        self.contrast_hf_gamma  = float(contrast_hf_gamma)
        self.contrast_fft       = bool(contrast_fft)
        self.contrast_fft_roll  = float(contrast_fft_roll)
        self.contrast_fft_gain_floor = float(contrast_fft_gain_floor)
        self.contrast_fft_energy     = bool(contrast_fft_energy)
        self.contrast_fft_energy_exp = float(contrast_fft_energy_exp)
        self.contrast_fft_energy_mix = float(contrast_fft_energy_mix)

        def _pick(override, global_v): return global_v if override is None else override
        self.contrast_lowpass_R_spot = contrast_lowpass_R_spot if contrast_lowpass_R_spot is not None else self.contrast_lowpass_R
        self.contrast_lowpass_R_fac  = contrast_lowpass_R_facula if contrast_lowpass_R_facula is not None else self.contrast_lowpass_R
        self.contrast_hf_gamma_spot  = contrast_hf_gamma_spot if contrast_hf_gamma_spot is not None else self.contrast_hf_gamma
        self.contrast_hf_gamma_fac   = contrast_hf_gamma_facula if contrast_hf_gamma_facula is not None else self.contrast_hf_gamma
        self.contrast_fft_spot            = _pick(contrast_fft_spot, self.contrast_fft)
        self.contrast_fft_fac             = _pick(contrast_fft_facula, self.contrast_fft)
        self.contrast_fft_roll_spot       = _pick(contrast_fft_roll_spot, self.contrast_fft_roll)
        self.contrast_fft_roll_fac        = _pick(contrast_fft_roll_facula, self.contrast_fft_roll)
        self.contrast_fft_gain_floor_spot = _pick(contrast_fft_gain_floor_spot, self.contrast_fft_gain_floor)
        self.contrast_fft_gain_floor_fac  = _pick(contrast_fft_gain_floor_facula, self.contrast_fft_gain_floor)
        self.contrast_fft_energy_spot     = _pick(contrast_fft_energy_spot, self.contrast_fft_energy)
        self.contrast_fft_energy_fac      = _pick(contrast_fft_energy_facula, self.contrast_fft_energy)
        self.contrast_fft_energy_exp_spot = _pick(contrast_fft_energy_exp_spot, self.contrast_fft_energy_exp)
        self.contrast_fft_energy_exp_fac  = _pick(contrast_fft_energy_exp_facula, self.contrast_fft_energy_exp)
        self.contrast_fft_energy_mix_spot = _pick(contrast_fft_energy_mix_spot, self.contrast_fft_energy_mix)
        self.contrast_fft_energy_mix_fac  = _pick(contrast_fft_energy_mix_facula, self.contrast_fft_energy_mix)

        self.contrast_r_floor_spot = contrast_r_floor_spot
        self.contrast_r_floor_fac  = contrast_r_floor_facula

        self.plot_sanity = bool(plot_sanity)
        self.abort_on_occultation = bool(abort_on_occultation)

        # anchor continuum
        self.anchor_continuum      = bool(anchor_continuum)
        self.anchor_q              = float(anchor_q)
        self.anchor_bins_per_R     = float(anchor_bins_per_R)
        self.anchor_Rwin_quant     = float(anchor_Rwin_quant)
        self.anchor_Rwin_close     = float(anchor_Rwin_close)
        self.anchor_post_gauss_um  = float(anchor_post_gauss_um)

        # Planck-trend
        self.trend_fix_enable = bool(trend_fix_enable)
        self.trend_R_lp       = float(trend_R_lp)

        # Line-boost knobs (function kept ready even if disabled)
        self.hf_boost_enable      = bool(hf_boost_enable)
        self.hf_boost_alpha       = float(hf_boost_alpha)
        self.hf_boost_max         = float(hf_boost_max)
        self.hf_boost_lambda_ref  = float(hf_boost_lambda_ref)

        # Zeeman placeholders
        self.zeeman_enable     = bool(zeeman_enable)
        self.zeeman_g_eff      = float(zeeman_g_eff)
        self.zeeman_pi_frac    = float(zeeman_pi_frac)
        self.zeeman_B_quiet_kG = float(zeeman_B_quiet_kG)
        self.zeeman_B_spot_kG  = float(zeeman_B_spot_kG)
        self.zeeman_B_facula_kG= float(zeeman_B_facula_kG)

        # --- geometry + PHOENIX warm-up ---
        raioStar_km = self.raioStar * 696_340.0
        self.planeta_ = Planeta(self.semiEixoUA, self.raioPlanetaRj, self.periodo, self.anguloInclinacao, self.ecc, self.anom, raioStar_km, 0)
        (self.tamanhoMatriz, _, self.star_pixels) = self.calculateMatrixFromTransit(self.planeta_, self.min_pixels, self.max_pixels, self.pixels_per_rp)

        T_list = [self.tempStar]
        if self.tempSpot   is not None: T_list.append(self.tempSpot)
        if self.tempFacula is not None: T_list.append(self.tempFacula)
        warm_phoenix_cache(self.lambdaEff, T_list, feh=0.0)

        self.debug_plots    = bool(debug_plots)
        self.debug_every    = int(debug_every)
        self.debug_max      = (None if debug_max is None else int(debug_max))
        self.debug_save_png = bool(debug_save_png)
        self.debug_outdir   = str(debug_outdir)
        if self.debug_plots and self.debug_save_png and (not os.path.exists(self.debug_outdir)):
            os.makedirs(self.debug_outdir, exist_ok=True)

        self.star_mix = Star(self.star_pixels, self.raioStar, 1.0, self.tamanhoMatriz)

        f_fac, f_sp = self._ff_effective()
        r_spot_pix_frac = (np.sqrt(f_sp/max(1, self.quantidade_spot)) if f_sp>0 else 0.0)
        r_fac_pix_frac  = (np.sqrt(f_fac/max(1, self.quantidade_fac)) if f_fac>0 else 0.0)

        if (self.tempSpot is not None) and (r_spot_pix_frac > 0.0):
            for j in range(self.quantidade_spot):
                latj = float(self.lat_spot[j % len(self.lat_spot)])
                lonj = float(self.long_spot[j % len(self.long_spot)])
                self.star_mix.addMancha(Star.Mancha(intensidade=1.0, raio=r_spot_pix_frac, latitude=latj, longitude=lonj))
        if (self.tempFacula is not None) and (r_fac_pix_frac > 0.0):
            for j in range(self.quantidade_fac):
                latj = float(self.lat_fac[j % len(self.lat_fac)])
                lonj = float(self.long_fac[j % len(self.long_fac)])
                self.star_mix.addFacula(Star.Facula(raio=r_fac_pix_frac, intensidade=1.0, latitude=latj, longitude=lonj))

        self.star_mix.build_static_structures()

        will_occult = self._warn_if_occulted(self.star_mix)
        if will_occult and self.abort_on_occultation:
            self._occultation_flag = True
            print("🚩 [ABORT] Occultation predicted for the largest filling factors. Halting execution.")
            return

        # --- Build R(λ) callable if 'auto' / function / array were provided ---
        self._R_callable = None
        if isinstance(self.R_preconv_override, str) and \
           self.R_preconv_override.lower() in ("auto", "auto_from_mrt", "auto_from_channel"):
            self._R_callable = _make_R_auto_from_channels(
                self.lambdaEff, self.dx_low_um, self.dx_high_um,
                floor_R=self.R_auto_floor_R, smooth_R=self.R_auto_smooth_R, beta=self.R_auto_beta
            )
        elif callable(self.R_preconv_override):
            self._R_callable = self.R_preconv_override
        elif isinstance(self.R_preconv_override, (list, tuple, np.ndarray)):
            arr = np.asarray(self.R_preconv_override, float)
            lam_ref = self.lambdaEff.copy()
            R_ref   = np.interp(lam_ref, lam_ref, arr, left=arr[0], right=arr[-1]) if arr.size==lam_ref.size else np.full_like(lam_ref, float(np.nanmedian(arr)))
            self._R_callable = lambda lu: np.interp(np.asarray(lu, float), lam_ref, R_ref, left=R_ref[0], right=R_ref[-1])

        self.run_simulations_pixel_mixing()

    # utils
    def calculateMatrixFromTransit(self, planet, min_pixels, max_pixels, pixels_per_rp, margin=0.1):
        rp_rstar = planet.raioPlanetaRstar
        pixels_per_rstar = pixels_per_rp / max(rp_rstar, EPS)
        pixels_per_rstar = np.clip(pixels_per_rstar, min_pixels / 2, max_pixels / 2)
        matrix_size = int(2 * pixels_per_rstar * (1 + margin))
        matrix_size = int(np.clip(matrix_size, min_pixels, max_pixels))
        return matrix_size, matrix_size, pixels_per_rstar

    def _write_header_if_needed(self, out_file: str):
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            return
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write("# ECLIPSE-Xλ simulation grid (single-disk, non-occulted normalization)\n")
            f.write(f"# depth_mode={EFFECTIVE_DEPTH_MODE} LDTK={self.profile} ldc_zero={self.ldc_zero}\n")
            f.write(f"# sed_phot={self.sed_mode_phot} sed_active={self.sed_mode_active} R_continuum={self.R_continuum}\n")
            f.write(f"# throughput={'on' if (self.thr_pair is not None) else 'off'}\n")
            f.write(f"# beta_spot={self.beta_spot} beta_facula={self.beta_facula}\n")
            f.write("f_spot,tempSpot,f_facula,tempFacula,wavelength,D_lambda\n")

    def salvar_dados_simulacao(self, f_spot, tempSpot, lambdaEff_nm, D_lambda, f_facula=None, tempFacula=None):
        out_file = f"simulation_results_{self.target}.txt"
        self._write_header_if_needed(out_file)
        if self.both_mode:
            f_sp_col = f_spot;  t_sp_col = tempSpot
            f_fa_col = f_facula; t_fa_col = tempFacula
        else:
            if self.simulation_mode == "spot":
                f_sp_col = f_spot; t_sp_col = tempSpot
                f_fa_col = np.nan; t_fa_col = np.nan
            elif self.simulation_mode in ("faculae", "facula", "facula-only"):
                f_sp_col = np.nan; t_sp_col = np.nan
                f_fa_col = f_facula; t_fa_col = tempFacula
            else:
                f_sp_col = 0.0; t_sp_col = np.nan
                f_fa_col = np.nan; t_fa_col = np.nan
        n = len(lambdaEff_nm)
        data = np.column_stack([
            np.full(n, f_sp_col, dtype=float),
            np.full(n, t_sp_col, dtype=float),
            np.full(n, f_fa_col, dtype=float),
            np.full(n, t_fa_col, dtype=float),
            np.array(lambdaEff_nm, dtype=float),
            np.array(D_lambda, dtype=float)
        ])
        with open(out_file, 'a', encoding='utf-8') as f:
            np.savetxt(f, data, delimiter=",", fmt="%.6f")

    def _ff_effective(self):
        if self.both_mode:
            f_fac = self.fillingFactor[0] if len(self.fillingFactor) > 0 else 0.0
            f_sp  = self.fillingFactor[1] if len(self.fillingFactor) > 1 else 0.0
        else:
            f_fac = 0.0; f_sp  = 0.0
            if self.simulation_mode == "spot":
                f_sp  = float(self.fillingFactor[0]) if len(self.fillingFactor) else 0.0
            elif self.simulation_mode in ("faculae", "facula", "facula-only"):
                f_fac = float(self.fillingFactor[0]) if len(self.fillingFactor) else 0.0
        f_fac = max(0.0, min(0.95, float(f_fac)))
        f_sp  = max(0.0, min(0.95, float(f_sp)))
        return f_fac, f_sp

    def _transit_centers_pixels(self, n_points_hint: int = 401):
        dtor = np.pi / 180.0
        N = int(self.tamanhoMatriz)
        a_pix = self.planeta_.semiEixoRaioStar * self.star_pixels
        inc = self.planeta_.anguloInclinacao
        a_rs = self.planeta_.semiEixoRaioStar
        latitude_transit = -np.arcsin(a_rs * np.cos(inc * dtor)) / dtor
        dur_h = 2 * (90. - np.arccos(np.cos(latitude_transit * dtor) / a_rs) / dtor) * self.planeta_.periodo / 360. * 24.
        total_h = 3.0 * dur_h

        n = max(101, int(n_points_hint))
        t = np.linspace(-0.5*total_h, +0.5*total_h, n)

        nk = 2.0*np.pi / (self.planeta_.periodo * 24.0)
        Tp = self.planeta_.periodo * self.planeta_.anom / 360.0 * 24.0
        m = nk * (t - Tp)
        eccanom = keplerfunc(m, self.planeta_.ecc)

        xs = a_pix * (np.cos(eccanom) - self.planeta_.ecc)
        ys = a_pix * (np.sqrt(1 - self.planeta_.ecc**2) * np.sin(eccanom))
        ang = self.planeta_.anom * dtor - (np.pi/2.0)
        xp = xs*np.cos(ang) - ys*np.sin(ang)
        yp = xs*np.sin(ang) + ys*np.cos(ang)

        xplan = xp - xp[np.argmin(np.abs(t))]
        yplan = yp * np.cos(self.planeta_.anguloInclinacao * dtor)

        msk = (np.abs(xplan) < 0.6*N) & (np.abs(yplan) < 0.6*N)
        xpix = xplan[msk] + N/2.0
        ypix = yplan[msk] + N/2.0

        r_p_pix = float(self.planeta_.raioPlanetaRstar * self.star_pixels)
        return xpix, ypix, r_p_pix, N

    def _warn_if_occulted(self, star_obj: Star) -> bool:
        xpix, ypix, r_p_pix, N = self._transit_centers_pixels()
        if r_p_pix <= 0 or not np.isfinite(r_p_pix):
            print("ℹ️ [check] Invalid planetary radius; assuming no occultation.")
            return False
        chord_mask = np.zeros((N, N), dtype=bool)
        yy, xx = np.indices((N, N))
        r2 = r_p_pix*r_p_pix
        for x0, y0 in zip(xpix, ypix):
            dx = xx - x0; dy = yy - y0
            chord_mask |= (dx*dx + dy*dy) <= r2
        any_sp = any((chord_mask & m).any() for m in getattr(star_obj, "_spot_masks", []))
        any_fa = any((chord_mask & m).any() for m in getattr(star_obj, "_fac_masks", []))
        return bool(any_sp or any_fa)

    def run_simulations_pixel_mixing(self):
        lam_um = self.lambdaEff
        lam_nm = lam_um * 1000.0
        n = self.num_elements
        D_lambda = np.zeros(n)

        f_fac, f_sp = self._ff_effective()

        r_sp_raw = np.ones(n, float)
        r_fa_raw = np.ones(n, float)

        for i, lam_c_um in enumerate(lam_um):
            dx_lo = None if (self.dx_low_um  is None) else float(self.dx_low_um[i])
            dx_hi = None if (self.dx_high_um is None) else float(self.dx_high_um[i])

            R_call = (self._R_callable if (self._R_callable is not None) else self.R_preconv_override)

            if (self.tempSpot is not None) and (f_sp > 0.0):
                r_sp = _ratio_channel_integrate_seds(
                    lam_c_um, dx_lo, dx_hi, self.R_fallback, self.nsub_obs,
                    self.tempSpot, self.tempStar, self.thr_pair, R_call,
                    sed_mode_active=self.sed_mode_active, sed_mode_phot=self.sed_mode_phot, R_continuum=self.R_continuum,
                    channel_window_kind=self.channel_window_kind, channel_tukey_alpha=self.channel_tukey_alpha,
                    hybrid_knobs_active=self.hybrid_knobs_spot,
                    hybrid_knobs_phot=self.hybrid_knobs_phot,
                    vsini_kms=self.vsini_kms, rot_ld_epsilon=self.rot_ld_epsilon, macro_kms=self.macro_kms,
                    anchor_continuum=self.anchor_continuum,
                    anchor_q=self.anchor_q,
                    anchor_bins_per_R=self.anchor_bins_per_R,
                    anchor_Rwin_quant=self.anchor_Rwin_quant,
                    anchor_Rwin_close=self.anchor_Rwin_close,
                    anchor_post_gauss_um=self.anchor_post_gauss_um,
                    # Zeeman
                    zeeman_enable=self.zeeman_enable,
                    zeeman_g_eff=self.zeeman_g_eff,
                    zeeman_pi_frac=self.zeeman_pi_frac,
                    zeeman_B_phot_kG=self.zeeman_B_quiet_kG,
                    zeeman_B_active_kG=self.zeeman_B_spot_kG,
                )
                if (self.contrast_r_floor_spot is not None):
                    r_sp = max(r_sp, float(self.contrast_r_floor_spot))
                scale_sp = self._spot_scale_fn(lam_c_um)
                r_sp = 1.0 + scale_sp * (r_sp - 1.0)
                r_sp_raw[i] = float(r_sp)
            else:
                r_sp_raw[i] = 1.0

            if (self.tempFacula is not None) and (f_fac > 0.0):
                r_fa = _ratio_channel_integrate_seds(
                    lam_c_um, dx_lo, dx_hi, self.R_fallback, self.nsub_obs,
                    self.tempFacula, self.tempStar, self.thr_pair, R_call,
                    sed_mode_active=self.sed_mode_active, sed_mode_phot=self.sed_mode_phot, R_continuum=self.R_continuum,
                    channel_window_kind=self.channel_window_kind, channel_tukey_alpha=self.channel_tukey_alpha,
                    hybrid_knobs_active=self.hybrid_knobs_facula,
                    hybrid_knobs_phot=self.hybrid_knobs_phot,
                    vsini_kms=self.vsini_kms, rot_ld_epsilon=self.rot_ld_epsilon, macro_kms=self.macro_kms,
                    anchor_continuum=self.anchor_continuum,
                    anchor_q=self.anchor_q,
                    anchor_bins_per_R=self.anchor_bins_per_R,
                    anchor_Rwin_quant=self.anchor_Rwin_quant,
                    anchor_Rwin_close=self.anchor_Rwin_close,
                    anchor_post_gauss_um=self.anchor_post_gauss_um,
                    # Zeeman
                    zeeman_enable=self.zeeman_enable,
                    zeeman_g_eff=self.zeeman_g_eff,
                    zeeman_pi_frac=self.zeeman_pi_frac,
                    zeeman_B_phot_kG=self.zeeman_B_quiet_kG,
                    zeeman_B_active_kG=self.zeeman_B_facula_kG,
                )
                if (self.contrast_r_floor_fac is not None):
                    r_fa = max(r_fa, float(self.contrast_r_floor_fac))
                scale_fa = self._fac_scale_fn(lam_c_um)
                r_fa = 1.0 + scale_fa * (r_fa - 1.0)
                r_fa_raw[i] = float(r_fa)
            else:
                r_fa_raw[i] = 1.0

        # Planck-trend (optional)
        if self.trend_fix_enable:
            if (self.tempSpot is not None) and (f_sp > 0.0):
                r_sp_raw = _bb_trend_substitute(lam_um, r_sp_raw, self.tempSpot, self.tempStar, R_lp=self.trend_R_lp)
            if (self.tempFacula is not None) and (f_fac > 0.0):
                r_fa_raw = _bb_trend_substitute(lam_um, r_fa_raw, self.tempFacula, self.tempStar, R_lp=self.trend_R_lp)

        r_sp_smooth = r_sp_raw
        r_fa_smooth = r_fa_raw
        if self.contrast_enable:
            if (self.tempSpot is not None) and (f_sp > 0.0) and (self.contrast_lowpass_R_spot is not None):
                r_sp_smooth = _smooth_contrast_curve(lam_um, r_sp_raw,
                                                     R_lp=float(self.contrast_lowpass_R_spot),
                                                     gamma=float(self.contrast_hf_gamma_spot))
            if (self.tempFacula is not None) and (f_fac > 0.0) and (self.contrast_lowpass_R_fac is not None):
                r_fa_smooth = _smooth_contrast_curve(lam_um, r_fa_raw,
                                                     R_lp=float(self.contrast_lowpass_R_fac),
                                                     gamma=float(self.contrast_hf_gamma_fac))

        if self.plot_sanity:
            plt.figure(figsize=(12.8, 4.6))
            plt.plot(lam_um, r_sp_raw, lw=1.2, alpha=0.35, label="spot raw")
            plt.plot(lam_um, r_sp_smooth, lw=2.0, label="spot smoothed")
            plt.plot(lam_um, r_fa_raw, lw=1.2, alpha=0.20, label="facula raw")
            plt.plot(lam_um, r_fa_smooth, lw=2.0, label="facula smoothed")
            plt.axvspan(0.62, 1.00, alpha=0.15, label="TiO/VO")
            plt.ylim(0.75, 1.01)
            plt.xlabel("Wavelength [µm]"); plt.ylabel(r"$r(\lambda)=F_{active}/F_{phot}$")
            plt.title("Sanity — contrast curves (raw vs smoothed)")
            plt.legend(loc="lower right"); plt.grid(alpha=0.25); plt.tight_layout()
            plt.show()

        D_lambda = np.zeros_like(lam_nm, float)
        debug_count = 0
        for i, lam_c_um in enumerate(lam_um):
            u1i, u2i, u3i, u4i = self.c1[i], self.c2[i], self.c3[i], self.c4[i]
            r_sp = float(r_sp_smooth[i])
            r_fa = float(r_fa_smooth[i])

            img_mix = self.star_mix.render_canal(
                u1i, u2i, u3i, u4i,
                r_spot_k=r_sp, r_fac_k=r_fa,
                beta_spot=self.beta_spot, beta_facula=self.beta_facula,
                Sphot_k=1.0
            )
            eclipse_mix = Eclipse(self.tamanhoMatriz, self.tamanhoMatriz, self.star_pixels, self.star_mix, self.planeta_)
            eclipse_mix.setEstrela(img_mix)
            eclipse_mix.criarEclipse(False, False)

            y_rel = np.asarray(eclipse_mix.getCurvaLuz(), float)
            y_rel = _renorm_curve(y_rel)

            d_ppm = _depth_extract(y_rel)
            if (not np.isfinite(d_ppm)) or (d_ppm < 1.0):
                rprs = float(self.planeta_.raioPlanetaRstar)
                d_ppm = max((rprs*rprs)*1e6, 1.0)
            D_lambda[i] = d_ppm

            if self.debug_plots and ((i % max(1, self.debug_every)) == 0) and (self.debug_max is None or debug_count < self.debug_max):
                fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120)
                im = ax0.imshow(img_mix, origin='upper', cmap='viridis', vmin=0.0, vmax=np.nanmax(img_mix))
                ax0.set_title(f"Mixed disk — λ={lam_c_um*1000:.1f} nm")
                ax0.axis('off'); fig.colorbar(im, ax=ax0)
                ax1.plot(y_rel, lw=1.6); ax1.set_xlabel("frame"); ax1.set_ylabel("relative flux")
                ax1.set_title("Relative light curve (single-disk)")
                fig.tight_layout()
                if self.debug_save_png:
                    fbase = f"debug_{i:04d}_{lam_c_um*1000:.1f}nm.png"
                    fig.savefig(os.path.join(self.debug_outdir, fbase), bbox_inches="tight")
                else:
                    plt.show()
                plt.close(fig); debug_count += 1

        self.salvar_dados_simulacao(
            f_spot=f_sp, tempSpot=self.tempSpot,
            lambdaEff_nm=lam_nm, D_lambda=D_lambda,
            f_facula=f_fac, tempFacula=self.tempFacula
        )

# ==========================================================
# Light-curve renorm & depth extractor
# ==========================================================
def _renorm_curve(y: np.ndarray, frac_oot: float = 0.15) -> np.ndarray:
    y = np.asarray(y, float)
    n = len(y)
    k = max(3, int(round(frac_oot * n)))
    k = min(k, n//2) if n >= 6 else k
    oot = np.r_[y[:k], y[-k:]] if k > 0 else y
    c = np.nanmedian(oot[np.isfinite(oot)]) if np.any(np.isfinite(oot)) else 1.0
    c = c if np.isfinite(c) and abs(c) > EPS else 1.0
    out = y / c
    return out

def _depth_extract(y: np.ndarray) -> float:
    y = np.asarray(y, float)
    n = len(y)
    mode = EFFECTIVE_DEPTH_MODE.lower().strip()
    if mode == "mid":
        y_eff = float(y[n//2])
    elif mode == "min":
        y_eff = float(np.nanmin(y))
    elif mode == "area":
        w = max(3, int(round(AREA_FRAC*n)))
        i0 = max(0, (n - w)//2); i1 = min(n, i0 + w)
        y_eff = float(np.nanmean(y[i0:i1]))
    elif mode == "percentile":
        kN = max(3, int(round(PCT_LOW*n))); kN = min(kN, max(3, n))
        idx = np.argpartition(y, kN)[:kN]
        y_eff = float(np.nanmean(y[idx]))
    elif mode == "parabola":
        j = int(np.nanargmin(y))
        half = max(2, int(round(0.5*PARAB_WIN_FRAC*n)))
        a = max(0, j - half); b = min(n, j + half + 1)
        xs = np.arange(a, b, dtype=float); ys = y[a:b].astype(float)
        if len(xs) >= 3 and np.isfinite(ys).sum() >= 3:
            A, B, C = np.polyfit(xs, ys, 2)
            if np.isfinite(A) and A != 0 and np.isfinite(B):
                x_v = -B/(2*A); y_v = (A*x_v + B)*x_v + C
                y_eff = float(y_v) if np.isfinite(y_v) else float(np.nanmin(ys))
            else:
                y_eff = float(np.nanmin(ys))
        else:
            y_eff = float(np.nanmin(ys))
    else:
        y_eff = float(np.nanmin(y))
    d_ppm = (1.0 - y_eff) * 1e6
    return d_ppm
