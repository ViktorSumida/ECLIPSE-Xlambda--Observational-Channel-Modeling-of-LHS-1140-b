# ldtk_ldc.py — LDTK limb-darkening with controlled smoothing (intra-band + cross-band)
from __future__ import annotations
import os, json, time, socket, hashlib
import numpy as np

# ================= Verbosity =================
_VERB = False
def _vprint(*a, **k):
    if _VERB:
        print(*a, **k)

EPS = 1e-15  # numerical floor

# ================= LDTK imports (robust across versions) =================
try:
    from ldtk import LDPSetCreator
except Exception as e:
    raise RuntimeError("LDTK not available. Please:  pip install -U ldtk") from e

_LimbDarkeningFitter = None
_NonlinearModel = None
_QuadraticModel  = None
_SquareRootModel = None

try:
    from ldtk.fitter import LimbDarkeningFitter as _LimbDarkeningFitter
except Exception:
    try:
        from ldtk import LimbDarkeningFitter as _LimbDarkeningFitter
    except Exception:
        _LimbDarkeningFitter = None

try:
    from ldtk.models import NonlinearModel as _NonlinearModel
except Exception:
    try:
        from ldtk import NonlinearModel as _NonlinearModel
    except Exception:
        _NonlinearModel = None

try:
    from ldtk.models import QuadraticModel as _QuadraticModel
except Exception:
    try:
        from ldtk import QuadraticModel as _QuadraticModel
    except Exception:
        _QuadraticModel = None

try:
    from ldtk.models import SquareRootModel as _SquareRootModel
except Exception:
    try:
        from ldtk import SquareRootModel as _SquareRootModel
    except Exception:
        _SquareRootModel = None

# Filters
_TabulatedFilter = None
_BoxcarFilter = None
try:
    from ldtk import TabulatedFilter as _TabulatedFilter
except Exception:
    pass
if _TabulatedFilter is None:
    try:
        from ldtk import BoxcarFilter as _BoxcarFilter
    except Exception:
        pass
if (_TabulatedFilter is None) and (_BoxcarFilter is None):
    raise RuntimeError("No compatible LDTK filter class found (TabulatedFilter/BoxcarFilter).")

# =========================================================
# Helpers (cache key, smoothing, throughput)
# =========================================================
def _lp_loglambda(lam_um, y, R=150.0):
    """Constant-R Gaussian low-pass in log-λ; mata microestrutura sem matar tendência."""
    lam = np.asarray(lam_um, float); z = np.log(np.clip(lam, 1e-12, None))
    yg  = np.asarray(y, float)
    dz  = 1.0 / (max(float(R), 1e-6) * 12.0)
    zg  = np.arange(z.min(), z.max() + 1e-12, dz)
    ygi = np.interp(zg, z, yg)
    sigma = 1.0 / (2.355 * max(float(R), 1e-6))
    half  = int(max(3, np.ceil(6.0 * sigma / dz)))
    k     = np.arange(-half, half+1) * dz
    ker   = np.exp(-0.5 * (k / max(sigma, 1e-15))**2)
    ker  /= (float(ker.sum()) or EPS)
    ygs   = np.convolve(np.pad(ygi, (half, half), mode="edge"), ker, mode="same")[half:-half]
    return np.interp(np.log(lam), zg, ygs)

def _mk_cache_key(teff, logg, feh, law, dataset, lam_um, dx_lo_um, dx_hi_um,
                  throughput_csv, throughput_hash, n_sub,
                  smooth_mode, R_smooth_lines, sigma_const_um, smooth_kind,
                  jitter_n, jitter_spread, jitter_reduce, tukey_alpha,
                  post_smooth_R, R_fit_min,
                  # NEW in cache key:
                  R_fit_min_blue, R_fit_min_red, R_fit_min_pivot_um,
                  jitter_strategy, half_split, aggregate):
    """Generate a cache key (includes smoothing policy)."""

    def _sigma_repr(s):
        if s is None:
            return None
        try:
            arr = np.asarray(s, float)
        except Exception:
            try:
                return float(s)
            except Exception:
                return str(s)
        if arr.ndim == 0:
            return float(arr)
        arr_r = np.round(arr, 9)
        h = hashlib.sha1(arr_r.tobytes()).hexdigest()[:12]
        return {"kind": "per_channel_sha", "sha": h, "len": int(arr.size)}

    payload = {
        "teff": float(teff), "logg": float(logg), "feh": float(feh),
        "law": str(law), "dataset": str(dataset),
        "lam_um": np.asarray(lam_um, float).round(9).tolist(),
        "dx_lo_um": np.asarray(dx_lo_um, float).round(9).tolist(),
        "dx_hi_um": np.asarray(dx_hi_um, float).round(9).tolist(),
        "throughput_csv": os.path.basename(throughput_csv or ""),
        "throughput_hash": throughput_hash or "none",
        "n_sub": int(n_sub),
        "smooth_mode": str(smooth_mode),
        "R_smooth_lines": (None if R_smooth_lines is None else float(R_smooth_lines)),
        "sigma_const_um": _sigma_repr(sigma_const_um),
        "smooth_kind": str(smooth_kind),
        "tukey_alpha": (None if tukey_alpha is None else float(tukey_alpha)),
        "jitter_n": int(jitter_n),
        "jitter_spread": float(jitter_spread),
        "jitter_reduce": str(jitter_reduce),
        "post_smooth_R": (None if post_smooth_R is None else float(post_smooth_R)),
        "R_fit_min": (None if R_fit_min is None else float(R_fit_min)),
        "R_fit_min_blue": (None if R_fit_min_blue is None else float(R_fit_min_blue)),
        "R_fit_min_red":  (None if R_fit_min_red  is None else float(R_fit_min_red)),
        "R_fit_min_pivot_um": (None if R_fit_min_pivot_um is None else float(R_fit_min_pivot_um)),
        "jitter_strategy": str(jitter_strategy),
        "half_split": bool(half_split),
        "aggregate": str(aggregate),
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _mk_ldtk_filter(name, lam_um, weights):
    lam_nm = np.asarray(lam_um, float) * 1000.0  # LDTK espera nm
    if _TabulatedFilter is not None:
        return _TabulatedFilter(name, lam_nm, weights)
    else:
        return _BoxcarFilter(name, float(lam_nm.min()), float(lam_nm.max()))

def _throughput_from_array(throughput_array):
    if throughput_array is None:
        return None, None, "none"
    lam, thr = throughput_array
    lam, thr = np.asarray(lam, float), np.asarray(thr, float)
    mx = float(np.nanmax(thr)) if np.any(np.isfinite(thr)) else 1.0
    if mx <= 0: mx = 1.0
    thr = np.clip(thr / mx, 0.0, 1.0)
    key = hashlib.sha1(np.round(thr, 6).tobytes()).hexdigest()[:10]
    return lam, thr, key

def _tukey_window(x, sigma, alpha=0.3):
    L = 3.0 * float(sigma)
    w = np.zeros_like(x)
    m = (x >= -L) & (x <= +L)
    if not np.any(m): return w
    u = (x[m] + L) / (2*L)
    a = float(max(0.0, min(1.0, alpha)))
    left  = u < a/2
    right = u > 1 - a/2
    mid   = (~left) & (~right)
    wl = np.zeros_like(u)
    wl[left]  = 0.5 * (1 + np.cos(np.pi*(2*u[left]/a - 1)))
    wl[right] = 0.5 * (1 + np.cos(np.pi*(2*u[right]/a - 2/a + 1)))
    wl[mid]   = 1.0
    w[m] = wl
    return w

def _kernel_weights(lam_sub, lam_center_eff,
                    smooth_mode="R", R_smooth_lines=10.0, sigma_const_um=None,
                    kind="tukey", tukey_alpha=0.30):
    lam_sub = np.asarray(lam_sub, float)
    kind_l = (kind or "tukey").lower().strip()
    if (str(smooth_mode).lower().strip() == "none") or (kind_l == "none"):
        w = np.ones_like(lam_sub, float)
        s = float(np.trapz(w, lam_sub)) or EPS
        return w / s

    if (smooth_mode == "sigma") and (sigma_const_um is not None) and (sigma_const_um > 0):
        sigma = float(sigma_const_um)
    else:
        R = float(R_smooth_lines) if (R_smooth_lines and R_smooth_lines > 0) else 8.0
        sigma = (float(lam_center_eff) / R) / 2.354820045

    x = lam_sub - float(lam_center_eff)
    if kind_l == "tukey":
        w = _tukey_window(x, sigma, alpha=(0.30 if tukey_alpha is None else float(tukey_alpha)))
    elif kind_l == "gauss":
        w = np.exp(-0.5 * (x / max(sigma, 1e-15))**2)
    else:
        w = np.exp(-0.5 * (x / max(sigma, 1e-15))**2)

    s = float(np.trapz(np.clip(w, 0.0, None), lam_sub)) or EPS
    return (w / s)

def _ldtk_creator_compat(teff, teff_unc, logg, logg_unc, feh, feh_unc, filters, dataset):
    try:
        return LDPSetCreator(
            teff=(float(teff), float(teff_unc)),
            logg=(float(logg), float(logg_unc)),
            z=(float(feh), float(feh_unc)),
            filters=filters,
            dataset=dataset
        )
    except TypeError:
        return LDPSetCreator(
            teff=float(teff), e_teff=float(teff_unc),
            logg=float(logg), e_logg=float(logg_unc),
            z=float(feh),   e_z=float(feh_unc),
            filters=filters, dataset=dataset
        )

def _fit_coeffs_from_profiles(ps, law: str, n_filters: int):
    law_l = (law or "").lower().strip()
    if _LimbDarkeningFitter is not None:
        try:
            if ("nonlinear" in law_l) or ("claret" in law_l) or law_l.startswith("4"):
                model = _NonlinearModel
            elif ("sqrt" in law_l) or ("square" in law_l):
                model = _SquareRootModel
            else:
                model = _QuadraticModel
            fitter = _LimbDarkeningFitter(ps, model)
            C = np.asarray(fitter.fit(), float)
            C = C.reshape(-1, C.shape[-1]) if C.ndim > 1 else np.tile(C, (n_filters, 1))
            out = np.zeros((n_filters, 4), float)
            out[:, :C.shape[1]] = C
            return out
        except Exception as e:
            _vprint(f"[LDTK] fitter path failed: {repr(e)}")

    try:
        if (("claret" in law_l) or ("nonlinear" in law_l) or law_l.startswith("4")) and hasattr(ps, "coeffs_nl"):
            C, _ = ps.coeffs_nl(do_mc=False)
            C = np.asarray(C, float).reshape(-1, 4)
            return C
        if hasattr(ps, "coeffs_qd"):
            C, _ = ps.coeffs_qd(do_mc=False)
            C = np.asarray(C, float).reshape(-1, 2)
            out = np.zeros((n_filters, 4), float); out[:, :2] = C
            return out
        if hasattr(ps, "coeffs_sqrt"):
            C, _ = ps.coeffs_sqrt(do_mc=False)
            C = np.asarray(C, float).reshape(-1, 3)
            out = np.zeros((n_filters, 4), float); out[:, :3] = C
            return out
    except Exception as e:
        _vprint(f"[LDTK] coeffs_* path failed: {repr(e)}")
    raise RuntimeError("Could not obtain LDTK coefficients for the requested 'law'.")

# =========================================================
# Public API
# =========================================================
def ldc_ldtk_for_channels(
    *,
    lambdaEff_um: np.ndarray,
    dx_low_um:    np.ndarray,
    dx_high_um:   np.ndarray,
    teff: float, logg: float, feh: float,
    teff_unc: float = 50.0, logg_unc: float = 0.1, feh_unc: float = 0.1,
    law: str = "claret4",
    dataset: str = "visir-lowres",
    # Throughput sources
    throughput_csv: str | None = None,
    throughput_array: tuple[np.ndarray, np.ndarray] | None = None,
    # Smoothing policy INSIDE each passband
    smooth_mode: str = "R",             # "R" | "sigma" | "none"
    R_smooth_lines: float | None = 120.0,  # if smooth_mode=="R"
    sigma_const_um: float | np.ndarray | None = None,  # if smooth_mode=="sigma"
    smooth_kind: str = "tukey",         # "tukey" | "gauss" | "none"
    tukey_alpha: float | None = 0.80,
    # Minimum resolving power to broaden each channel before LDTK fit
    R_fit_min: float | None = 140.0,    # None -> use actual channel width
    # Adaptive R_min (optional; overrides R_fit_min per side of pivot)
    R_fit_min_blue: float | None = None,
    R_fit_min_red:  float | None = None,
    R_fit_min_pivot_um: float = 1.00,
    # Half-split and jitter strategies
    half_split: bool = False,
    jitter_strategy: str = "avg_kernel",   # "avg_kernel" | "replicate"
    jitter_n: int = 7,
    jitter_spread: float = 0.8,            # in sigma units
    jitter_reduce: str = "mean",           # kept for backward compat (avg_kernel)
    aggregate: str = "median",             # combine replicate/halves: "median"|"mean"
    # Cross-band *post* smoothing of the final c_i(λ)
    post_smooth_R: float | None = 200.0,  # e.g. 150–200. None=off
    # Sampling, retries
    n_sub: int = 1601,
    timeout_s: float = 120.0,
    max_retries: int = 8,
    verbose: bool = False,
):
    """
    Build passbands and fit LDCs via LDTK com:
      • alargamento mínimo por R (R_fit_min ou peça-adaptativa VIS/NIR);
      • 'replicate jitter' -> múltiplos filtros por canal (mediana);
      • 'half_split' -> azul/vermelho por canal (média/mediana);
      • pós-suavização inter-canais (post_smooth_R em log-λ).
    """
    global _VERB
    _VERB = bool(verbose)

    lambdaEff_um = np.asarray(lambdaEff_um, float)
    dx_low_um    = np.asarray(dx_low_um, float)
    dx_high_um   = np.asarray(dx_high_um, float)
    n_ch = lambdaEff_um.size

    # Intra-banda: sigma (opcional)
    sigma_arr = None
    if (smooth_mode == "sigma") and (sigma_const_um is not None):
        arr = np.asarray(sigma_const_um, float)
        sigma_arr = np.full(n_ch, float(arr)) if arr.ndim == 0 else arr.astype(float, copy=False)
        if sigma_arr.size != n_ch:
            raise ValueError(f"sigma_const_um has size {sigma_arr.size}, but n_channels={n_ch}.")

    try:
        socket.setdefaulttimeout(float(timeout_s))
    except Exception:
        pass

    _ensure_dir("./ldc_cache")
    thr_lam, thr_vals, thr_hash = _throughput_from_array(throughput_array)

    cache_key = _mk_cache_key(
        teff, logg, feh, law, dataset, lambdaEff_um, dx_low_um, dx_high_um,
        throughput_csv, thr_hash, n_sub,
        smooth_mode, R_smooth_lines, sigma_const_um, smooth_kind,
        jitter_n, jitter_spread, jitter_reduce, tukey_alpha,
        post_smooth_R, R_fit_min,
        R_fit_min_blue, R_fit_min_red, R_fit_min_pivot_um,
        jitter_strategy, half_split, aggregate
    )
    cache_path = os.path.join("ldc_cache", f"ldc_{cache_key}.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        _vprint("[LDTK] SOURCE=cache")
        return z["c1"], z["c2"], z["c3"], z["c4"]

    # ----- helpers: R_min adaptativo -----
    def _Rmin_for_channel(lam_c):
        if (R_fit_min_blue is not None) and (R_fit_min_red is not None):
            return float(R_fit_min_blue if lam_c < float(R_fit_min_pivot_um) else R_fit_min_red)
        return (None if R_fit_min is None else float(R_fit_min))

    # ----- Build filters (replicates + half-split se ligado) -----
    filt_list = []
    meta = []  # (start_idx, count) por canal

    R_smooth_lines_eff = float(R_smooth_lines) if (R_smooth_lines is not None) else (float(R_fit_min) if R_fit_min else 120.0)
    jit_mode = (jitter_strategy or "avg_kernel").lower().strip()
    agg = (aggregate or "median").lower().strip()

    total_filters = 0

    for i, (lam_c, dlo, dhi) in enumerate(zip(lambdaEff_um, dx_low_um, dx_high_um)):
        lam_lo_ch = lam_c + float(dlo)
        lam_hi_ch = lam_c + float(dhi)
        width_ch  = lam_hi_ch - lam_lo_ch

        Rmin_i = _Rmin_for_channel(lam_c)
        if (Rmin_i is not None) and (Rmin_i > 0):
            width_target = float(lam_c) / float(Rmin_i)
            span = max(float(width_ch), float(width_target))
            lam_lo_fit = float(lam_c) - 0.5*span
            lam_hi_fit = float(lam_c) + 0.5*span
        else:
            lam_lo_fit = lam_lo_ch
            lam_hi_fit = lam_hi_ch

        lam_sub = np.linspace(lam_lo_fit, lam_hi_fit, int(max(61, n_sub)))

        # Throughput
        thr = np.ones_like(lam_sub) if (thr_lam is None or thr_vals is None) else np.interp(lam_sub, thr_lam, thr_vals, left=0.0, right=0.0)
        s_thr = float(np.trapz(thr, lam_sub)) or EPS
        lam_eff = float(np.trapz(lam_sub * thr, lam_sub) / s_thr)

        def _kernel(center):
            use_sigma = (float(sigma_arr[i]) if (smooth_mode == "sigma" and sigma_arr is not None) else None)
            return _kernel_weights(
                lam_sub, center,
                smooth_mode=smooth_mode,
                R_smooth_lines=R_smooth_lines_eff,
                sigma_const_um=use_sigma,
                kind=smooth_kind,
                tukey_alpha=tukey_alpha if (tukey_alpha is not None) else 0.30
            )

        start = len(filt_list)

        if jit_mode == "replicate":
            K = int(max(1, jitter_n))
            # σ do kernel (para espalhar offsets em unidades de σ)
            if (smooth_mode == "sigma") and (sigma_arr is not None):
                sigma_um = float(sigma_arr[i])
            else:
                Rloc = float(R_smooth_lines_eff if R_smooth_lines_eff else 10.0)
                sigma_um = (lam_eff / Rloc) / 2.354820045
            offsets = np.linspace(-float(jitter_spread), float(jitter_spread), K) * sigma_um
            for j, dk in enumerate(offsets):
                ker = _kernel(lam_eff + dk)
                W_full = thr * ker
                W_full /= (float(np.trapz(W_full, lam_sub)) or EPS)
                if half_split:
                    mask_b = (lam_sub <= lam_eff + 1e-15)
                    mask_r = (lam_sub >= lam_eff - 1e-15)
                    Wb = W_full * mask_b; sb = float(np.trapz(Wb, lam_sub)) or EPS; Wb /= sb
                    Wr = W_full * mask_r; sr = float(np.trapz(Wr, lam_sub)) or EPS; Wr /= sr
                    filt_list.append(_mk_ldtk_filter(f"ch{i:03d}_jit{j:02d}_B", lam_sub, Wb))
                    filt_list.append(_mk_ldtk_filter(f"ch{i:03d}_jit{j:02d}_R", lam_sub, Wr))
                else:
                    filt_list.append(_mk_ldtk_filter(f"ch{i:03d}_jit{j:02d}", lam_sub, W_full))
        else:
            # modo antigo: média de kernels antes do filtro
            K = int(max(1, jitter_n))
            if (smooth_mode == "sigma") and (sigma_arr is not None):
                sigma_um = float(sigma_arr[i])
            else:
                Rloc = float(R_smooth_lines_eff if R_smooth_lines_eff else 10.0)
                sigma_um = (lam_eff / Rloc) / 2.354820045
            offsets = np.linspace(-float(jitter_spread), float(jitter_spread), K) * sigma_um if K>1 else [0.0]
            ker_stack = np.vstack([_kernel(lam_eff + dk) for dk in offsets])
            ker = np.mean(ker_stack, axis=0)
            ker /= (float(np.trapz(ker, lam_sub)) or EPS)
            W_full = thr * ker
            W_full /= (float(np.trapz(W_full, lam_sub)) or EPS)
            if half_split:
                mask_b = (lam_sub <= lam_eff + 1e-15)
                mask_r = (lam_sub >= lam_eff - 1e-15)
                Wb = W_full * mask_b; sb = float(np.trapz(Wb, lam_sub)) or EPS; Wb /= sb
                Wr = W_full * mask_r; sr = float(np.trapz(Wr, lam_sub)) or EPS; Wr /= sr
                filt_list.append(_mk_ldtk_filter(f"ch{i:03d}_avg_B", lam_sub, Wb))
                filt_list.append(_mk_ldtk_filter(f"ch{i:03d}_avg_R", lam_sub, Wr))
            else:
                filt_list.append(_mk_ldtk_filter(f"ch{i:03d}_avg", lam_sub, W_full))

        count = (K * 2) if half_split else K
        meta.append((start, count))
        total_filters += count

    _vprint(f"[LDTK] building {total_filters} passbands for {n_ch} channels "
            f"(jitter={jit_mode}:{jitter_n}, half_split={half_split})")

    # ----- Create profiles and fit coefficients -----
    last_err = None
    for attempt in range(1, int(max_retries)+1):
        try:
            sc = _ldtk_creator_compat(
                teff, teff_unc, logg, logg_unc, feh, feh_unc,
                filters=filt_list, dataset=dataset
            )
            ps = sc.create_profiles()
            _vprint(f"[LDTK] profiles created for {len(filt_list)} passbands (dataset='{dataset}')")

            C_all = _fit_coeffs_from_profiles(ps, law, len(filt_list))  # shape: (total_filters, p<=4)

            # Agregar por canal (replicates x (metades?))
            out = np.zeros((n_ch, 4), float)
            for ch, (start, count) in enumerate(meta):
                block = C_all[start:start+count, :]            # (count, p)
                if agg == "mean":
                    vec = np.nanmean(block, axis=0)
                else:
                    vec = np.nanmedian(block, axis=0)
                out[ch, :block.shape[1]] = vec[:block.shape[1]]

            c1, c2, c3, c4 = out[:,0], out[:,1], out[:,2], out[:,3]

            # ===== cross-band smoothing (mata wiggles residuais) =====
            if (post_smooth_R is not None) and (post_smooth_R > 0):
                Rps = float(post_smooth_R)
                c1 = _lp_loglambda(lambdaEff_um, c1, R=Rps)
                c2 = _lp_loglambda(lambdaEff_um, c2, R=Rps)
                c3 = _lp_loglambda(lambdaEff_um, c3, R=Rps)
                c4 = _lp_loglambda(lambdaEff_um, c4, R=Rps)

            np.savez_compressed(cache_path, c1=c1, c2=c2, c3=c3, c4=c4)
            _vprint("[LDTK] SOURCE=online (computed & cached)")
            return c1, c2, c3, c4

        except Exception as e:
            last_err = e
            _vprint(f"[LDTK] attempt {attempt} failed: {repr(e)}")
            time.sleep(1.2 * attempt)

    raise RuntimeError(f"[LDTK] failed to obtain LDCs. Last error: {repr(last_err)}")
