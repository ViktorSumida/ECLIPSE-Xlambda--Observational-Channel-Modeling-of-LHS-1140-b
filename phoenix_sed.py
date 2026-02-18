#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHOENIX SED utilities (HiRes FITS; PHOENIX-ACES-AGSS-COND-2011)

- Cache em disco + memória
- Interpolação robusta para λ arbitrário (em µm)
- 'allow_nearest=True' → cai no nó de Teff mais próximo quando necessário
- Auto-download de FITS (CACHE_DIR) com proxies opcionais
- Continuum/envelope:
    * phoenix_continuum_global        → blur a R constante (log-λ)
    * phoenix_envelope_global         → envelope (quantil + closing + blur λ)
    * phoenix_continuum_hybrid_global → mix + clamp bilateral + pós-blur opcional
    * super_continuum_envelope        → envelope adaptativo (robusto)
- "Tamed": amplitude-only line tamer com α(λ) (escalar, callable ou dict)
  α=1 → preserva linhas; α=0 → achata linhas; α(λ) contínuo evita piecewise.

Compatível com chamadas existentes no seu código.
"""

from __future__ import annotations
import os, io
from typing import Dict, Tuple, Sequence, List, Callable, Any
import numpy as np
import requests
from astropy.io import fits

# ---------------------------- Remote dataset ----------------------------
BASE = "https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS"
SET  = "PHOENIX-ACES-AGSS-COND-2011"
LOGG_TRY = [5.00, 4.50]  # dwarfs primeiro

# Temperatura (K)
T_MIN_PHX = 2300.0
T_MAX_PHX = 7000.0
T_STEP    = 100

# ---------------------------- Local cache ----------------------------
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".phoenix_hires_cache")
FORCE_OFFLINE = False
PROXIES = {k: v for k, v in {
    "http":  os.environ.get("HTTP_PROXY")  or os.environ.get("http_proxy"),
    "https": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
}.items() if v}

# ---------------------------- Helpers ----------------------------
def _feh_str(z: float) -> str:
    s = f"{z:+.1f}"
    return s.replace("+0.0", "-0.0")  # servidor usa 'Z-0.0'

def _phoenix_filename(teff: int, logg: float, feh: float) -> str:
    z = _feh_str(feh)
    return f"lte{int(teff):05d}-{logg:.2f}{z}.{SET}-HiRes.fits"

def _phoenix_url(teff: int, logg: float, feh: float) -> str:
    z = _feh_str(feh)
    fname = _phoenix_filename(teff, logg, feh)
    return f"{BASE}/{SET}/Z{z}/{fname}"

def _local_cache_path(teff: int, logg: float, feh: float) -> str:
    z = _feh_str(feh)
    d = os.path.join(CACHE_DIR, SET, f"Z{z}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, _phoenix_filename(teff, logg, feh))

def _get_bytes(url: str, timeout: int = 120) -> io.BytesIO:
    r = requests.get(
        url, timeout=timeout, proxies=PROXIES,
        headers={"User-Agent": "python-requests/phoenix"},
        allow_redirects=True,
        stream=False,
    )
    r.raise_for_status()
    return io.BytesIO(r.content)

def _read_wave_from_wavefits() -> np.ndarray:
    """Fallback: WAVE_*.fits em Å → retorna µm."""
    url = f"{BASE}/WAVE_{SET}.fits"
    with fits.open(_get_bytes(url), memmap=False) as hdul:
        for h in hdul:
            if getattr(h, "data", None) is not None:
                w = np.array(h.data).ravel().astype(float)
                if w.size > 1000:
                    return w * 1e-4
    raise RuntimeError("WAVE_* FITS missing wavelength grid.")

def _sanitize_wave_flux(w_um: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Garante: 1D, finito, ordenado, estritamente crescente em λ.
    Se houver λ repetidos, faz 'reduce' por média.
    """
    w = np.asarray(w_um, float).ravel()
    y = np.asarray(f, float).ravel()
    m = np.isfinite(w) & np.isfinite(y)
    w, y = w[m], y[m]
    if w.size < 10:
        return w, y
    order = np.argsort(w, kind="mergesort")
    w, y = w[order], y[order]
    dw = np.diff(w)
    if np.any(dw <= 0):
        u, inv = np.unique(w, return_inverse=True)
        sums = np.bincount(inv, weights=y)
        counts = np.bincount(inv)
        y = sums / np.clip(counts, 1, None)
        w = u
    return w, y

def _read_flux_and_wave_from_hdul(hdul) -> Tuple[np.ndarray, np.ndarray]:
    # 1) Primário com WCS linear
    if hdul[0].data is not None and hdul[0].header.get("CRVAL1") is not None:
        h = hdul[0].header
        flux = np.asarray(hdul[0].data, float).ravel()
        n = flux.size
        crv, cd = float(h["CRVAL1"]), float(h["CDELT1"])
        cp = float(h.get("CRPIX1", 1.0))
        idx = np.arange(n, dtype=float)
        wavA = crv + (idx + 1.0 - cp) * cd
        w_um, f = wavA * 1e-4, flux
        return _sanitize_wave_flux(w_um, f)

    # 2) Tabela com colunas nomeadas
    for hdu in hdul[1:]:
        if hasattr(hdu, "columns") and hdu.columns is not None:
            names = [n.upper() for n in hdu.columns.names]
            if ("WAVE" in names) and ("FLUX" in names):
                tab = hdu.data
                w_um = np.array(tab["WAVE"], float).ravel() * 1e-4
                f    = np.array(tab["FLUX"], float).ravel()
                return _sanitize_wave_flux(w_um, f)

    # 3) Fallback: usar WAVE_* global
    flux = np.asarray(hdul[0].data, float).ravel()
    wav_um = _read_wave_from_wavefits()
    n = min(wav_um.size, flux.size)
    return _sanitize_wave_flux(wav_um[:n], flux[:n])

def _read_flux_and_wave_from_file_or_url(teff: int, logg: float, feh: float) -> Tuple[np.ndarray, np.ndarray]:
    path = _local_cache_path(teff, logg, feh)
    if os.path.isfile(path):
        with fits.open(path, memmap=False) as hdul:
            return _read_flux_and_wave_from_hdul(hdul)
    if FORCE_OFFLINE:
        raise FileNotFoundError(f"[OFFLINE] Missing local PHOENIX file: {path}")
    url = _phoenix_url(teff, logg, feh)
    bio = _get_bytes(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(bio.getbuffer())
    with fits.open(path, memmap=False) as hdul:
        return _read_flux_and_wave_from_hdul(hdul)

def _clip_T_to_phx_grid(T: float) -> float:
    Tgrid = int(round(T / T_STEP) * T_STEP)
    return float(np.clip(Tgrid, T_MIN_PHX, T_MAX_PHX))

def _candidate_Tgrids(Tgrid_center: int, step: int = T_STEP) -> List[int]:
    T_all = list(range(int(T_MIN_PHX), int(T_MAX_PHX) + 1, step))
    return sorted(T_all, key=lambda t: abs(t - int(Tgrid_center)))

# ---------------------------- Caches (in-memory) ----------------------------
_PHX_CACHE: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray]] = {}
_PHX_INTERP_CACHE: Dict[Tuple[Tuple[float, ...], int, float], np.ndarray] = {}
_PHX_MISSING: set[Tuple[int, float]] = set()

# ---------------------------- API: IO & Interp ----------------------------
def ensure_phoenix_file(teff: int, logg: float = 5.00, feh: float = 0.0) -> str:
    path = _local_cache_path(teff, logg, feh)
    if os.path.isfile(path):
        return path
    if FORCE_OFFLINE:
        raise FileNotFoundError(f"[OFFLINE] Missing local PHOENIX file: {path}")
    url = _phoenix_url(teff, logg, feh)
    bio = _get_bytes(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(bio.getbuffer())
    return path

def download_missing_teffs(T_list: Sequence[float],
                           feh: float = 0.0,
                           logg_list: Sequence[float] = tuple(LOGG_TRY)) -> List[Tuple[int, float]]:
    ensured: List[Tuple[int, float]] = []
    Ts = sorted(set(int(round(_clip_T_to_phx_grid(T) / T_STEP) * T_STEP) for T in T_list))
    for Tgrid in Ts:
        for lg in logg_list:
            try:
                p = ensure_phoenix_file(Tgrid, lg, feh=feh)
                ensured.append((Tgrid, lg))
                print(f"[PHX] ready: Teff={Tgrid} K, logg={lg:.2f} → {p}")
            except Exception as e:
                print(f"[PHX] failed to ensure Teff={Tgrid} K, logg={lg:.2f}: {e}")
    return ensured

def phoenix_flux_interp_um(lam_um_target: np.ndarray,
                           T: float,
                           feh: float = 0.0,
                           allow_nearest: bool = True) -> np.ndarray:
    lam_target = np.asarray(lam_um_target, float)
    lam_key = tuple(lam_target)
    Tgrid0 = int(round(_clip_T_to_phx_grid(T) / T_STEP) * T_STEP)

    T_candidates = _candidate_Tgrids(Tgrid0) if allow_nearest else [Tgrid0]
    last_errors = []
    for Tgrid in T_candidates:
        if all((Tgrid, lg) in _PHX_MISSING for lg in LOGG_TRY):
            continue
        for lg in LOGG_TRY:
            key_interp = (lam_key, Tgrid, lg)
            if key_interp in _PHX_INTERP_CACHE:
                return _PHX_INTERP_CACHE[key_interp]

            base_key = (Tgrid, lg)
            if base_key not in _PHX_CACHE:
                try:
                    lam_model_um, flux = _read_flux_and_wave_from_file_or_url(Tgrid, lg, feh)
                except Exception as e:
                    _PHX_MISSING.add((Tgrid, lg))
                    last_errors.append((Tgrid, lg, repr(e)))
                    continue
                _PHX_CACHE[base_key] = (lam_model_um, flux)

            lam_model_um, flux = _PHX_CACHE[base_key]
            lam_model_um, flux = _sanitize_wave_flux(lam_model_um, flux)
            m = np.isfinite(lam_model_um) & np.isfinite(flux)
            if np.count_nonzero(m) < 10:
                _PHX_MISSING.add((Tgrid, lg))
                last_errors.append((Tgrid, lg, "insufficient finite points"))
                continue

            f_interp = np.interp(lam_target, lam_model_um[m], flux[m], left=np.nan, right=np.nan)
            if not np.any(np.isfinite(f_interp)):
                _PHX_MISSING.add((Tgrid, lg))
                last_errors.append((Tgrid, lg, "target λ outside model range"))
                continue

            _PHX_INTERP_CACHE[key_interp] = f_interp
            if Tgrid != Tgrid0:
                print(f"[PHX] Teff≈{Tgrid0} K unavailable; using neighbor {Tgrid} K (logg={lg:.2f}).")
            return f_interp

        for lg in LOGG_TRY:
            _PHX_MISSING.add((Tgrid, lg))
        if not allow_nearest:
            break

    msg = f"No PHOENIX file for Teff≈{Tgrid0} K"
    if last_errors:
        msg += f" (tried neighbors; last errors: {last_errors[-3:]})"
    raise RuntimeError(msg + ".")

def warm_phoenix_cache(lam_um_target: np.ndarray,
                       T_list: Sequence[float],
                       feh: float = 0.0,
                       download_missing: bool = False) -> int:
    Ts = sorted(set(int(round(_clip_T_to_phx_grid(T) / T_STEP) * T_STEP) for T in T_list))
    if download_missing and not FORCE_OFFLINE:
        download_missing_teffs(Ts, feh=feh, logg_list=LOGG_TRY)
    ok = 0
    for Tgrid in Ts:
        try:
            _ = phoenix_flux_interp_um(lam_um_target, Tgrid, feh=feh, allow_nearest=True)
            ok += 1
        except Exception as e:
            print(f"[PHX] warm miss Teff={Tgrid} K: {e}")
    print(f"[PHOENIX] warmed cache for {ok}/{len(Ts)} Teff on {len(np.atleast_1d(lam_um_target))} λ.")
    return ok

# ============================ Continuum & Envelope ============================
EPS = 1e-15

def _blur_constant_R_loglambda(lam_um, flux, R=50.0):
    lam = np.asarray(lam_um, float); y = np.asarray(flux, float)
    lam = np.clip(lam, 1e-9, None)
    z = np.log(lam)
    dz = 1.0 / (max(R, 1e-6) * 12.0)
    z_grid = np.arange(z.min(), z.max()+1e-12, dz)
    y_grid = np.interp(z_grid, z, y)
    sigma_log = 1.0 / (2.355 * max(R, 1e-6))
    half = int(max(3, np.ceil(6.0 * sigma_log / dz)))
    k = np.arange(-half, half+1) * dz
    ker = np.exp(-0.5 * (k / max(sigma_log, 1e-15))**2)
    ker /= max(float(ker.sum()), EPS)
    yg = np.convolve(np.pad(y_grid, (half, half), mode="edge"), ker, mode="same")[half:-half]
    lam_grid = np.exp(z_grid)
    return np.interp(lam, lam_grid, yg)

# ---- utilitários envelope ----
def _rolling_max(arr, win):
    if win <= 1: return np.asarray(arr, float).copy()
    arr = np.asarray(arr, float); n = arr.size
    out = np.full(n, np.nan, float); w = int(max(1, win)); h = w // 2
    for i in range(n):
        j0 = max(0, i - h); j1 = min(n, i + h + 1)
        out[i] = np.nanmax(arr[j0:j1])
    return out

def _build_log_bins(lam_um, R_window=40.0, bins_per_R=2.0):
    lam = np.asarray(lam_um, float); z = np.log(np.clip(lam, 1e-12, None))
    dz = 1.0 / max(R_window * bins_per_R, 1.0)
    edges = np.arange(z.min(), z.max() + 1e-12, dz)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return z, edges, centers

def _binned_upper_quantile(z, y, edges, q=0.997):
    y = np.asarray(y, float); idx = np.digitize(z, edges) - 1
    nb = len(edges) - 1; qv = np.full(nb, np.nan, float)
    for b in range(nb):
        m = (idx == b)
        if np.any(m): qv[b] = np.quantile(y[m], float(q))
    return qv

def gaussian_blur_um(lam_um, flux, sigma_um):
    lam = np.asarray(lam_um, float); y = np.asarray(flux, float)
    if (sigma_um is None) or (sigma_um <= 0): return y.copy()
    dlam = np.median(np.diff(lam[np.isfinite(lam)]))
    if not np.isfinite(dlam) or dlam <= 0:
        dlam = (lam.max() - lam.min()) / 5000.0
    half = int(np.ceil(6.0 * sigma_um / dlam))
    half = max(1, min(half, max(1, len(lam)//3)))
    kx = np.arange(-half, half+1) * dlam
    ker = np.exp(-0.5 * (kx / max(sigma_um, 1e-15))**2); ker /= max(float(ker.sum()), EPS)
    ypad = np.pad(y, (half, half), mode="edge")
    return np.convolve(ypad, ker, mode="same")[half:-half]

# -------------------- Adaptive super-continuum --------------------
def super_continuum_envelope(lam_um, flux,
                             Rwin_quant=120.0, bins_per_R=3.0, q=0.985,
                             Rwin_close=300.0, post_gauss_um=0.060,
                             strategy: str = "adaptive",
                             floor_min: float = 0.0):
    """
    Envelope unitário S(λ)∈[0,1] que segue o topo das linhas sem 'vazar'.
    strategy='adaptive' é robusto (TiO/VO etc).
    floor_min: piso opcional do envelope (0.0 por padrão → sem ondulações artificiais).
    """
    lam = np.asarray(lam_um, float); F = np.asarray(flux, float)

    if strategy.lower().strip() != "adaptive":
        z, edges, centers = _build_log_bins(lam, R_window=Rwin_quant, bins_per_R=bins_per_R)
        qv = _binned_upper_quantile(z, F, edges, q=q)
        w_close = int(max(3, round((Rwin_close / Rwin_quant) * bins_per_R)))
        qv_cl = _rolling_max(qv, w_close)
        valid = np.isfinite(qv_cl)
        if not np.any(valid):
            raise RuntimeError("Super-continuum: empty quantile; adjust Rwin/q.")
        first, last = np.argmax(valid), len(valid) - 1 - np.argmax(valid[::-1])
        qv_cl[:first]  = qv_cl[first]
        qv_cl[last+1:] = qv_cl[last]
        lam_src = np.exp(centers)
        env = np.interp(lam, lam_src, qv_cl, left=qv_cl[first], right=qv_cl[last])
        env = env / (np.nanmax(env) or 1.0)
        if post_gauss_um and post_gauss_um > 0:
            env = gaussian_blur_um(lam, env, post_gauss_um)
        if floor_min is not None:
            env = np.maximum(env, float(floor_min))
        return np.clip(env, 0.0, 1.0)

    # Adaptativo
    knots_um   = np.array([0.60, 0.75, 1.00, 1.30, 1.70, 2.50], float)
    knots_Rwin = np.array([180.0, 150.0, 120.0, 110.0, 100.0,  90.0], float)
    knots_q    = np.array([0.995, 0.992, 0.988, 0.986, 0.984, 0.982], float)

    Rloc = np.interp(lam, knots_um, knots_Rwin, left=knots_Rwin[0], right=knots_Rwin[-1])
    qloc = np.interp(lam, knots_um, knots_q,    left=knots_q[0],    right=knots_q[-1])

    z = np.log(np.clip(lam, 1e-12, None))
    R_med = float(np.median(Rloc))
    z_edges_step = 1.0 / max(R_med * bins_per_R, 1.0)
    edges = np.arange(z.min(), z.max() + 1e-12, z_edges_step)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(z, edges) - 1
    nb = len(edges) - 1
    qv = np.full(nb, np.nan, float)

    for b in range(nb):
        m = (idx == b)
        if not np.any(m):
            continue
        q_b = float(np.nanmedian(qloc[m]))
        q_b = float(np.clip(q_b, 0.90, 0.999))
        qv[b] = np.quantile(F[m], q_b)

    R_close_ratio = float(Rwin_close / max(R_med, 1e-6))
    w_close = int(max(3, round(R_close_ratio * bins_per_R)))
    qv_cl = _rolling_max(qv, w_close)

    valid = np.isfinite(qv_cl)
    if not np.any(valid):
        raise RuntimeError("Super-continuum: empty quantile; adjust adaptive knots.")

    first, last = np.argmax(valid), len(valid) - 1 - np.argmax(valid[::-1])
    qv_cl[:first]  = qv_cl[first]
    qv_cl[last+1:] = qv_cl[last]

    lam_src = np.exp(centers)
    env = np.interp(lam, lam_src, qv_cl, left=qv_cl[first], right=qv_cl[last])
    env = env / (np.nanmax(env) or 1.0)
    if post_gauss_um and post_gauss_um > 0:
        env = gaussian_blur_um(lam, env, post_gauss_um)

    if floor_min is not None:
        env = np.maximum(env, float(floor_min))  # 0.0 por padrão → sem piso artificial

    return np.clip(env, 0.0, 1.0)

def phoenix_continuum_global(lam_um_global, T, feh=0.0, R=30.0):
    F_full = phoenix_flux_interp_um(lam_um_global, T=T, feh=feh)
    return _blur_constant_R_loglambda(lam_um_global, F_full, R=R)

def phoenix_envelope_global(lam_um_global, T, feh=0.0,
                            Rwin_quant=120.0, bins_per_R=3.0, q=0.980,
                            Rwin_close=300.0, post_gauss_um=0.060,
                            floor_min: float = 0.0):
    F_full = phoenix_flux_interp_um(lam_um_global, T=T, feh=feh)
    return super_continuum_envelope(
        lam_um_global, F_full,
        Rwin_quant=Rwin_quant, bins_per_R=bins_per_R, q=q,
        Rwin_close=Rwin_close, post_gauss_um=post_gauss_um,
        floor_min=floor_min
    )

def phoenix_pseudocontinuum_global(lam_um_global, T, feh=0.0,
                                   Rwin_quant=120.0, bins_per_R=3.0, q=0.980,
                                   Rwin_close=300.0, post_gauss_um=0.060,
                                   floor_min: float = 0.0):
    return phoenix_envelope_global(
        lam_um_global, T, feh=feh,
        Rwin_quant=Rwin_quant, bins_per_R=bins_per_R, q=q,
        Rwin_close=Rwin_close, post_gauss_um=post_gauss_um,
        floor_min=floor_min
    )

def phoenix_continuum_hybrid_global(lam_um_global, T, feh=0.0,
                                    R_blur=30.0,
                                    q=0.985, Rwin_quant=120.0, Rwin_close=300.0,
                                    bins_per_R=3.0, post_gauss_um=0.040,
                                    beta_env=0.95, cap_env=1.00,
                                    alpha_mix=0.60, R_post: float | None = None,
                                    env_floor_min: float = 0.0):
    """
    Híbrido (continuum) com clamp bilateral:
      1) F_full  = PHOENIX
      2) F_blur  = blur a R constante (log-λ)
      3) E_unit  = envelope unitário
      4) scale   = mediana(F_blur/E_unit) → F_env = scale * E_unit
      5) mix     = α*F_blur + (1-α)*F_env
      6) clamp   = max(β*F_env, min(mix, cap*F_env))
      7) pós-blur opcional
    """
    lam = np.asarray(lam_um_global, float)
    F_full = phoenix_flux_interp_um(lam, T=T, feh=feh)
    F_blur = _blur_constant_R_loglambda(lam, F_full, R=max(float(R_blur), 1e-3))
    E_unit = phoenix_envelope_global(lam, T=T, feh=feh,
                                     Rwin_quant=Rwin_quant, bins_per_R=bins_per_R, q=q,
                                     Rwin_close=Rwin_close, post_gauss_um=post_gauss_um,
                                     floor_min=env_floor_min)

    m = np.isfinite(F_blur) & np.isfinite(E_unit) & (E_unit > 1e-6)
    if not np.any(m):
        raise RuntimeError("Hybrid: no valid points to scale envelope.")
    scale = np.nanmedian(F_blur[m] / np.clip(E_unit[m], 1e-12, None))
    F_env = scale * E_unit

    alpha = float(np.clip(alpha_mix, 0.0, 1.0))
    beta  = float(beta_env)
    cap   = float(cap_env) if (cap_env is not None) else np.inf

    F_mix = alpha * F_blur + (1.0 - alpha) * F_env
    F_mix = np.maximum(F_mix, beta * F_env)
    F_mix = np.minimum(F_mix, cap  * F_env)

    if R_post is not None and np.isfinite(R_post) and R_post > 0:
        F_mix = _blur_constant_R_loglambda(lam, F_mix, R=float(R_post))

    return F_mix

# ============== Amplitude-only microstructure tamer (λ-dependente) ==============
HYB_RIPPLE_ALPHA_DEFAULT = 1.0  # 1.0 = preservar linhas; 0.0 = achatar.

def apply_amplitude_only_tamer(lam_um, flux_full, cont_smooth, alpha):
    """
    Reduz apenas a amplitude das linhas em torno de um contínuo suave:

        base  = cont_smooth
        micro = flux_full/base - 1
        out   = base * (1 + alpha * micro)

    α pode ser escalar ou array (mesmo shape de lam_um). α=1 preserva; α=0 zera.
    """
    base = np.maximum(np.asarray(cont_smooth, float), EPS)
    micro = (np.asarray(flux_full, float) / base) - 1.0
    if np.ndim(alpha) == 0:
        a = float(np.clip(alpha, 0.0, 1.0))
        return base * (1.0 + a * micro)
    a = np.clip(np.asarray(alpha, float), 0.0, 1.0)
    return base * (1.0 + a * micro)

def _alpha_lambda_from_spec(lam_um, ripple_alpha: float | dict | Callable[[np.ndarray], Any]):
    """
    Constrói α(λ) a partir de:
      - escalar            (0..1)
      - callable(lam_um)   → array/esc.
      - dict:
          kind='sigmoid'  → lam0_um, k, alpha_lo, alpha_hi
          kind='table'    → lam_um[], alpha[]
          kind='powerlaw' → pivot_um, alpha0, gamma
    """
    lam = np.asarray(lam_um, float)

    # Escalar
    try:
        if ripple_alpha is None:
            return 0.0
        if np.isscalar(ripple_alpha):
            return float(np.clip(ripple_alpha, 0.0, 1.0))
    except Exception:
        pass

    # Callable
    if callable(ripple_alpha):
        arr = np.asarray(ripple_alpha(lam), float)
        return np.clip(arr, 0.0, 1.0)

    # Dict
    if isinstance(ripple_alpha, dict):
        kind = str(ripple_alpha.get("kind", "sigmoid")).lower().strip()
        if kind == "sigmoid":
            lam0 = float(ripple_alpha.get("lam0_um", 1.0))
            k    = float(ripple_alpha.get("k", 10.0))
            alo  = float(ripple_alpha.get("alpha_lo", 0.2))
            ahi  = float(ripple_alpha.get("alpha_hi", 1.0))
            # α ~ ahi para λ << lam0; α ~ alo para λ >> lam0 (transição suave)
            z = 1.0 / (1.0 + np.exp(-k * (lam - lam0)))
            return np.clip(alo + (ahi - alo) * (1.0 - z), 0.0, 1.0)
        elif kind == "table":
            lt = np.asarray(ripple_alpha.get("lam_um", []), float)
            at = np.asarray(ripple_alpha.get("alpha", []), float)
            if lt.size >= 2 and at.size == lt.size:
                arr = np.interp(lam, lt, at, left=at[0], right=at[-1])
                return np.clip(arr, 0.0, 1.0)
        elif kind == "powerlaw":
            pivot = float(ripple_alpha.get("pivot_um", 1.0))
            a0    = float(ripple_alpha.get("alpha0", 1.0))
            gam   = float(ripple_alpha.get("gamma", -1.0))
            arr = a0 * np.power(np.clip(lam / max(pivot, 1e-6), 1e-6, 1e6), gam)
            return np.clip(arr, 0.0, 1.0)

    # Fallback
    try:
        return float(np.clip(ripple_alpha, 0.0, 1.0))
    except Exception:
        return 1.0

def phoenix_continuum_hybrid_global_tamed(
    lam_um, *, T, feh,
    ripple_alpha: float | dict | Callable[[np.ndarray], Any] = HYB_RIPPLE_ALPHA_DEFAULT,
    **kwargs
):
    """
    Constrói um "quase-contínuo" e reinjeta microestrutura com amplitude α(λ).
    α(λ)=1 → mantém linhas do PHOENIX; α(λ)=0 → contínuo puro.
    kwargs → mesmos knobs do phoenix_continuum_hybrid_global (R_blur, q, ...).
    """
    lam = np.asarray(lam_um, float)
    flux_full = phoenix_flux_interp_um(lam, T=T, feh=feh)
    cont = phoenix_continuum_hybrid_global(lam, T=T, feh=feh, **kwargs)

    alpha = _alpha_lambda_from_spec(lam, ripple_alpha)

    # Se α ~ 0 em todo λ, devolve o contínuo
    if (np.ndim(alpha) == 0 and alpha <= 1e-6) or (np.ndim(alpha) > 0 and np.all(alpha <= 1e-6)):
        return cont

    return apply_amplitude_only_tamer(lam, flux_full, cont, alpha)
