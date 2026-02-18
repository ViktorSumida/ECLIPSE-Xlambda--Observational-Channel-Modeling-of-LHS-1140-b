# -*- coding: utf-8 -*-
# ldc_specint_continuum.py — LDCs a partir do PHOENIX SpecInt50 (só-contínuo)
from __future__ import annotations
import numpy as np

from phoenix_intensity_loader import load_specint_grid
# Vamos reutilizar somente utilidades "genéricas" do seu phoenix_sed:
from phoenix_sed import super_continuum_envelope

EPS = 1e-15

# --- blur a R constante (log-λ) ---
def _blur_constant_R_loglambda(lam_um, y, R=50.0):
    lam = np.asarray(lam_um, float); y = np.asarray(y, float)
    z = np.log(np.clip(lam, 1e-12, None))
    dz = 1.0 / (max(float(R), 1e-6) * 12.0)
    zg = np.arange(z.min(), z.max() + 1e-12, dz)
    yg = np.interp(zg, z, y)
    sigma = 1.0 / (2.355 * max(float(R), 1e-6))
    half = int(max(3, np.ceil(6.0 * sigma / dz)))
    k = np.arange(-half, half+1) * dz
    ker = np.exp(-0.5 * (k / max(sigma, 1e-15))**2)
    ker /= (np.sum(ker) or 1.0)
    yb = np.convolve(np.pad(yg, (half, half), mode="edge"),
                     ker, mode="same")[half:-half]
    return np.interp(z, zg, yb)

# --- híbrido do contínuo (idêntico ao dos fluxos, mas aplicado a I_μ(λ)) ---
def _hybrid_continuum(lam_um, spec,
                      *, R_blur, q, Rwin_quant, Rwin_close,
                      bins_per_R, post_gauss_um, alpha_mix,
                      beta_env, cap_env, R_post):
    lam = np.asarray(lam_um, float)
    full = np.asarray(spec, float)
    blur = _blur_constant_R_loglambda(lam, full, R=max(float(R_blur), 1e-3))
    env_unit = super_continuum_envelope(lam, full,
                                        Rwin_quant=Rwin_quant, bins_per_R=bins_per_R, q=q,
                                        Rwin_close=Rwin_close, post_gauss_um=post_gauss_um,
                                        floor_min=0.0)
    m = np.isfinite(blur) & np.isfinite(env_unit) & (env_unit > 1e-9)
    scale = np.nanmedian(blur[m] / np.clip(env_unit[m], 1e-12, None))
    env = scale * env_unit
    a = float(np.clip(alpha_mix, 0.0, 1.0))
    mix = a*blur + (1.0 - a)*env
    mix = np.maximum(mix, float(beta_env) * env)
    if cap_env is not None:
        mix = np.minimum(mix, float(cap_env) * env)
    if R_post is not None and np.isfinite(R_post) and R_post > 0:
        mix = _blur_constant_R_loglambda(lam, mix, R=float(R_post))
    return np.asarray(mix, float)

def _tukey_window(x, lo, hi, alpha=0.30):
    x = np.asarray(x, float); L = hi - lo
    if not np.isfinite(L) or L <= 0: return np.zeros_like(x)
    u = (x - lo)/L; a = float(np.clip(alpha, 0.0, 1.0))
    w = np.zeros_like(x)
    left = (u < a/2); right = (u > 1 - a/2)
    mid = (~left) & (~right) & (u>=0) & (u<=1)
    w[left]  = 0.5*(1 + np.cos(np.pi*(2*u[left]/a - 1)))
    w[right] = 0.5*(1 + np.cos(np.pi*(2*u[right]/a - 2/a + 1)))
    w[mid]   = 1.0
    s = float(np.trapz(w, x)) or EPS
    return w/s

def _lp_loglambda(lam_um, y, R=300.0):
    lam = np.asarray(lam_um, float); y = np.asarray(y, float)
    z = np.log(np.clip(lam, 1e-12, None))
    dz = 1.0/(max(float(R),1e-6)*12.0)
    zg = np.arange(z.min(), z.max()+1e-12, dz)
    yg = np.interp(zg, z, y)
    sigma = 1.0/(2.355*max(float(R),1e-6))
    half = int(np.ceil(6.0*sigma/dz))
    k = np.arange(-half, half+1)*dz
    ker = np.exp(-0.5*(k/max(sigma,1e-15))**2)
    ker /= (np.sum(ker) or 1.0)
    yf = np.convolve(np.pad(yg,(half,half),'edge'), ker, mode='same')[half:-half]
    return np.interp(z, zg, yf)

def _fit_claret4(mu, I_norm, nonneg=True, hi=1.5):
    mu = np.asarray(mu, float); y = 1.0 - np.asarray(I_norm, float)
    A = np.column_stack([1 - mu**0.5, 1 - mu, 1 - mu**1.5, 1 - mu**2])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)  # problema linear
    if nonneg: c = np.clip(c, 0.0, float(hi))
    return c  # c1..c4

def ldc_specint_continuum(
    *, lambdaEff_um, dx_low_um, dx_high_um,
    teff: int, logg: float = 5.00, feh: float = 0.0,
    # janela do canal
    channel_window_kind: str = "tukey", channel_tukey_alpha: float = 0.30,
    throughput_array=None,
    # knobs do contínuo (use os do seu preset)
    R_blur: float = 8.0, q: float = 0.990,
    Rwin_quant: float = 100.0, Rwin_close: float = 320.0,
    bins_per_R: float = 12.0, post_gauss_um: float = 0.05,
    alpha_mix: float = 0.40, beta_env: float = 1.00,
    cap_env: float | None = None, R_post: float | None = None,
    ripple_alpha: float = 0.0,      # 0 → “só-contínuo”
    # pós‑suavização entre canais
    post_smooth_R: float | None = 270.0,
    nsub_lambda: int = 801,
    mu_min: float = 0.02
):
    lam_c = np.asarray(lambdaEff_um, float)
    dlo   = np.asarray(dx_low_um, float)
    dhi   = np.asarray(dx_high_um, float)

    # (1) carrega SpecInt50
    lam_full, mu_grid, I_full = load_specint_grid(int(teff), logg=float(logg), feh=float(feh))

    # restringe domínio para acelerar
    lo_all = np.nanmin(lam_c + dlo) - 0.10
    hi_all = np.nanmax(lam_c + dhi) + 0.10
    mdom = (lam_full >= max(1e-3, lo_all)) & (lam_full <= hi_all)
    lam = lam_full[mdom]; I_full = I_full[:, mdom].astype(float, copy=False)

    # (2) contínuo híbrido por μ (linhas fora)
    nmu = len(mu_grid)
    I_cont = np.zeros_like(I_full)
    for j in range(nmu):
        spec = I_full[j, :]
        cont = _hybrid_continuum(
            lam, spec, R_blur=R_blur, q=q,
            Rwin_quant=Rwin_quant, Rwin_close=Rwin_close,
            bins_per_R=bins_per_R, post_gauss_um=post_gauss_um,
            alpha_mix=alpha_mix, beta_env=beta_env,
            cap_env=cap_env, R_post=R_post
        )
        if ripple_alpha <= 1e-3:
            I_cont[j, :] = cont
        else:
            base = np.maximum(cont, EPS)
            micro = spec/base - 1.0
            I_cont[j, :] = base * (1.0 + float(ripple_alpha)*micro)

    # (3) integra por canal, normaliza e ajusta Claret4
    C = np.zeros((lam_c.size, 4), float)
    mu_use = mu_grid[mu_grid >= float(mu_min)]
    idx_mu = np.searchsorted(mu_grid, mu_use)

    # throughput (opcional)
    if throughput_array is not None:
        lam_thr = np.asarray(throughput_array[0], float)
        thr_val = np.asarray(throughput_array[1], float)
    else:
        lam_thr = thr_val = None

    for i, lc in enumerate(lam_c):
        lo = lc + float(dlo[i]); hi = lc + float(dhi[i])
        x = np.linspace(lo, hi, int(max(101, nsub_lambda)))

        if channel_window_kind.lower().strip() == "tukey":
            w_ch = _tukey_window(x, lo, hi, alpha=float(channel_tukey_alpha))
        else:
            w_ch = np.where((x>=lo)&(x<=hi), 1.0, 0.0)
            s = float(np.trapz(w_ch, x)) or EPS; w_ch /= s

        if lam_thr is not None:
            thr = np.interp(x, lam_thr, thr_val, left=0.0, right=0.0)
            w_eff = w_ch * thr
            s = float(np.trapz(w_eff, x)) or EPS; w_eff /= s
        else:
            w_eff = w_ch

        I_mu = np.zeros_like(mu_use)
        for k, j in enumerate(idx_mu):
            spec = np.interp(x, lam, I_cont[j, :])
            I_mu[k] = float(np.trapz(spec * w_eff, x))

        I0 = float(np.interp(1.0, mu_use, I_mu, left=I_mu[0], right=I_mu[-1]))
        In = np.clip(I_mu / max(I0, EPS), 0.0, 2.0)

        C[i, :] = _fit_claret4(mu_use, In, nonneg=True, hi=1.5)

    if (post_smooth_R is not None) and (post_smooth_R > 0):
        for k in range(4):
            C[:, k] = _lp_loglambda(lam_c, C[:, k], R=float(post_smooth_R))

    return C[:,0], C[:,1], C[:,2], C[:,3]
