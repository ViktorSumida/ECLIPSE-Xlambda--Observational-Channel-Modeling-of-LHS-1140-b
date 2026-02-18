# phoenix_intensity_loader.py
from __future__ import annotations
import os, time
import numpy as np
import requests
from astropy.io import fits

# ===== CONFIG =====
WIN_BASE  = r"C:\Users\vikto\OneDrive\ECLIPSE-Xlambda\ECLIPSE-Xlambda - LHS 1140 b"
CACHE_DIR = os.path.join(WIN_BASE, "phoenix_specint_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

BASE = "https://phoenix.astro.physik.uni-goettingen.de/data/SpecInt50FITS"
SET  = "PHOENIX-ACES-AGSS-COND-SPECINT-2011"


def _feh_str(z: float) -> str:
    s = f"{z:+.1f}"
    # grade 2011: Z-0.0 é o solar
    return s.replace("+0.0", "-0.0")


def _fname(teff: int, logg: float, feh: float) -> str:
    # ex: lte03100-5.00-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits
    return f"lte{teff:05d}-{logg:.2f}{_feh_str(feh)}.{SET}.fits"


def _url(teff: int, logg: float, feh: float) -> str:
    return f"{BASE}/{SET}/Z{_feh_str(feh)}/{_fname(teff, logg, feh)}"


def _path(teff: int, logg: float, feh: float) -> str:
    d = os.path.join(CACHE_DIR, SET, f"Z{_feh_str(feh)}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, _fname(teff, logg, feh))


def ensure_specint(
    teff: int,
    logg: float = 5.00,
    feh: float = 0.0,
    retries: int = 3,
    timeout: int = 60,
) -> str:
    """Garante que o arquivo SpecInt50FITS exista localmente."""
    p = _path(teff, logg, feh)
    if os.path.isfile(p):
        return p

    u = _url(teff, logg, feh)
    last = None
    for k in range(retries):
        try:
            r = requests.get(
                u,
                timeout=timeout,
                headers={"User-Agent": "python-requests/phoenix-specint"},
            )
            r.raise_for_status()
            with open(p, "wb") as f:
                f.write(r.content)
            return p
        except Exception as e:
            last = e
            time.sleep(1.5 * (k + 1))

    raise RuntimeError(f"Falha ao baixar SpecInt: {u}  (último erro: {last})")


def _wave_from_header(hdu, axis: int):
    """
    Reconstrói λ (em microns) a partir do header de um HDU imagem.
    """
    hdr = hdu.header
    n = hdr.get(f"NAXIS{axis}")
    if not n:
        return None

    crval = hdr.get(f"CRVAL{axis}")
    cdelt = hdr.get(f"CDELT{axis}") or hdr.get(f"CD{axis}_{axis}")
    crpix = hdr.get(f"CRPIX{axis}", 1.0)
    cunit = (hdr.get(f"CUNIT{axis}", "Angstrom") or "").strip().lower()

    if crval is None or cdelt is None:
        return None

    pix = np.arange(int(n), dtype=float)
    x = crval + (pix + 1.0 - float(crpix)) * cdelt

    if "ang" in cunit:
        lam_um = x * 1e-4
    elif "nm" in cunit:
        lam_um = x * 1e-3
    elif "um" in cunit or "micron" in cunit:
        lam_um = x
    else:
        lam_um = x * 1e-4
    return lam_um


def load_specint_grid(teff: int,
                      logg: float = 5.00,
                      feh: float = 0.0,
                      verbose: bool = False):
    """
    Lê um arquivo SpecInt50FITS do conjunto
    PHOENIX-ACES-AGSS-COND-SPECINT-2011 e retorna:

      lam_um : (Nλ,)  em microns
      mu     : (Nµ,)  valores de cos(theta)
      I      : (Nµ,Nλ) intensidades específicas

    Layout específico do seu ficheiro:
      - PRIMARY HDU (0): imagem 2D com NAXIS1=510 (λ), NAXIS2=78 (µ)
      - Um HDU seguinte chamado 'MU' (ou equivalente) com vetor 1D de µ.
      - λ é reconstruído a partir de CRVAL1/CDELT1/CRPIX1 em Å.
    """
    p = ensure_specint(teff, logg, feh)
    if verbose:
        print(f"[SpecInt] abrindo '{p}'")

    with fits.open(p, memmap=False) as hdul:
        # 1) Intensidade no PRIMARY HDU
        I = np.array(hdul[0].data, float)
        hdr0 = hdul[0].header

        nlam = int(hdr0["NAXIS1"])  # 510
        nmu  = int(hdr0["NAXIS2"])  # 78

        # 2) Reconstrói λ (em Å) a partir de CRVAL1/CDELT1/CRPIX1
        crval1 = hdr0.get("CRVAL1")
        cdelt1 = hdr0.get("CDELT1")
        crpix1 = hdr0.get("CRPIX1", 1.0)

        if crval1 is None or cdelt1 is None:
            raise RuntimeError(
                f"Header do SpecInt sem CRVAL1/CDELT1 em '{p}'."
            )

        # índices FITS são 1-based
        j = np.arange(1, nlam + 1, dtype=float)
        wavA = crval1 + cdelt1 * (j - crpix1)  # em Angstrom

        # 3) Procura vetor MU em algum HDU seguinte
        mu = None
        for hdu in hdul[1:]:
            if hdu.data is None:
                continue
            arr = np.array(hdu.data, float).ravel()
            if arr.size == nmu and np.all(np.isfinite(arr)):
                if (arr.min() >= 0.0) and (arr.max() <= 1.0 + 1e-3):
                    mu = arr
                    break

        if mu is None:
            raise RuntimeError(
                f"Não encontrei vetor MU com {nmu} elementos em '{p}'."
            )

        # 4) Garante shape (Nµ, Nλ) para I
        if I.shape == (nmu, nlam):
            pass  # já está como (µ, λ)
        elif I.shape == (nlam, nmu):
            I = I.T
        else:
            raise RuntimeError(
                f"Shape inesperado de I em '{p}': {I.shape}, "
                f"esperado ({nmu},{nlam}) ou ({nlam},{nmu})."
            )

    lam_um = wavA * 1e-4  # Å → µm
    return lam_um, mu, I

