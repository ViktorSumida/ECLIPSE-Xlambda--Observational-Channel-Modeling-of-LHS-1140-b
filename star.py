# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from numba import njit, prange

class Star:
    """
    2D stellar map with Claret-4 limb darkening (per-channel), plus precomputed masks.
    IMPORTANT: render_canal espera fatores multiplicativos **r = F_active/F_phot**.
               Para diagnóstico há um kill-switch no main (EXPECTS_RATIO) que pode
               enviar 1/r se você quiser testar inversão de fase.
    """
    def __init__(self, raio, raioSun, intensidadeMaxima, tamanhoMatriz):
        self.raio = float(raio)
        self.raioSun = float(raioSun) * 696_340.0
        self.intensidadeMaxima = float(intensidadeMaxima)
        self.tamanhoMatriz = int(tamanhoMatriz)
        self.Nx = self.tamanhoMatriz
        self.Ny = self.tamanhoMatriz

        self.manchas: List[Star.Mancha] = []
        self.faculas: List[Star.Facula] = []

        self._mu_map: Optional[np.ndarray] = None
        self._spot_masks: List[np.ndarray] = []
        self._fac_masks: List[np.ndarray] = []

        self.coeficienteHum = 0.0
        self.coeficienteDois = 0.0
        self.coeficienteTres = 0.0
        self.coeficienteQuatro = 0.0
        self.estrelaMatriz: Optional[np.ndarray] = None

        self.starName: Optional[str] = None
        self.cadence: Optional[float] = None
        self.color = "hot"

    class Mancha:
        def __init__(self, intensidade, raio, latitude, longitude):
            self.intensidade = float(intensidade)
            self.raio = float(raio)
            self.latitude = float(latitude)
            self.longitude = float(longitude)

    class Facula:
        def __init__(self, raio, intensidade, latitude, longitude):
            self.intensidade = float(intensidade)
            self.raio = float(raio)
            self.latitude = float(latitude)
            self.longitude = float(longitude)

    @njit(parallel=True)
    def _ld_map_numba(lin, col, size, raio_pix, I1, c1, c2, c3, c4):
        I  = np.zeros((lin, col), dtype=np.float64)
        MU = np.zeros((lin, col), dtype=np.float64)
        center = size / 2.0
        inv_raio = 1.0 / max(raio_pix, 1e-12)
        for i in prange(lin):
            for j in range(col):
                dx = i - center
                dy = j - center
                r_over_R = math.hypot(dx, dy) * inv_raio
                if r_over_R <= 1.0:
                    mu = math.sqrt(max(0.0, 1.0 - r_over_R*r_over_R))
                    part = (1.0
                            - c1*(1.0 - math.sqrt(mu))
                            - c2*(1.0 - mu)
                            - c3*(1.0 - (mu**1.5))
                            - c4*(1.0 - (mu**2.0)))
                    if part < 0.0:
                        part = 0.0
                    I[i, j]  = I1 * part
                    MU[i, j] = mu
        return I, MU

    def set_ld_coeffs(self, u1, u2, u3, u4):
        self.coeficienteHum   = float(u1)
        self.coeficienteDois  = float(u2)
        self.coeficienteTres  = float(u3)
        self.coeficienteQuatro= float(u4)

    def addMancha(self, mancha): self.manchas.append(mancha)

    def addFacula(self, facula):
        if facula.intensidade < 1.0:
            print("Facula intensity must be > 1 (ignored).")
            return
        self.faculas.append(facula)

    def resetManchas(self):
        self.manchas = []
        self.faculas = []
        self._spot_masks = []
        self._fac_masks = []

    def build_static_structures(self):
        if self._mu_map is None:
            I, MU = Star._ld_map_numba(
                self.Nx, self.Ny, self.tamanhoMatriz, self.raio,
                1.0, 0.0, 0.0, 0.0, 0.0
            )
            self._mu_map = MU

        self._spot_masks = [self._make_elliptic_mask(m.latitude, m.longitude, m.raio) for m in self.manchas]
        self._fac_masks  = [self._make_elliptic_mask(f.latitude, f.longitude, f.raio) for f in self.faculas]

    def _make_elliptic_mask(self, latitude_deg, longitude_deg, raio_frac):
        raio_pix = self.raio * float(raio_frac)
        d2r = np.pi / 180.0
        lat = float(latitude_deg) * d2r
        lon = float(longitude_deg) * d2r
        ys = self.raio * np.sin(lat)
        xs = self.raio * np.cos(lat) * np.sin(lon)
        cos_hel = float(np.cos(lat) * np.cos(lon))
        cos_hel = float(np.clip(cos_hel, 1e-6, 1.0))
        yy = ys + self.Ny / 2.0
        xx = xs + self.Nx / 2.0
        if abs(xs) < 1e-12:
            ang_rot = 0.0
        else:
            ang_rot = math.atan(abs(ys / xs))
        if (lat * lon) > 0.0:
            ang_rot = -ang_rot
        elif (lat == 0.0) or (lon == 0.0):
            ang_rot = 0.0
        ca, sa = math.cos(ang_rot), math.sin(ang_rot)
        yy_idx, xx_idx = np.indices((self.Ny, self.Nx))
        vx = xx_idx.astype(np.float64) - xx
        vy = yy_idx.astype(np.float64) - yy
        xp = (vx*ca - vy*sa) / cos_hel
        yp = (vx*sa + vy*ca)
        mask = (xp*xp + yp*yp) < (raio_pix*raio_pix)
        if self._mu_map is not None:
            mask = mask & (self._mu_map > 0.0)
        return mask

    def render_canal(self, u1, u2, u3, u4,
                     r_spot_k=1.0, r_fac_k=1.0,
                     beta_spot=1.0, beta_facula=1.0,
                     Sphot_k=1.0):
        ld_map, mu_here = Star._ld_map_numba(
            self.Nx, self.Ny, self.tamanhoMatriz, self.raio,
            1.0, float(u1), float(u2), float(u3), float(u4)
        )
        self.estrelaMatriz = ld_map * float(self.intensidadeMaxima) * float(Sphot_k)

        if self._mu_map is None:
            self._mu_map = mu_here

        if self._spot_masks and (r_spot_k != 1.0):
            mu_pow = np.where(mu_here > 0.0, mu_here**float(beta_spot), 0.0)
            r_eff_map_sp = 1.0 + (float(r_spot_k) - 1.0) * mu_pow
            for mask in self._spot_masks:
                self.estrelaMatriz[mask] *= r_eff_map_sp[mask]

        if self._fac_masks and (r_fac_k != 1.0):
            mu_pow = np.where(mu_here > 0.0, mu_here**float(beta_facula), 0.0)
            r_eff_map_fa = 1.0 + (float(r_fac_k) - 1.0) * mu_pow
            for mask in self._fac_masks:
                self.estrelaMatriz[mask] *= r_eff_map_fa[mask]

        return self.estrelaMatriz

    # ---------- plotting helper ----------
    def Plotar(self, tamanhoMatriz: Optional[int] = None, estrela: Optional[np.ndarray] = None) -> None:
        if tamanhoMatriz is None:
            tamanhoMatriz = self.tamanhoMatriz
        if estrela is None:
            estrela = self.estrelaMatriz
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(estrela, cmap=self.color, origin='upper')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    # ---------- getters ----------
    def getNx(self): return self.Nx
    def getNy(self): return self.Ny
    def getRaioStar(self): return self.raio
    def getMatrizEstrela(self): return self.estrelaMatriz
    def getTamanhoMatriz(self): return self.tamanhoMatriz
    def getRaioSun(self): return self.raioSun
    def getIntensidadeMaxima(self): return self.intensidadeMaxima
    def setStarName(self, starName: str): self.starName = starName
    def getStarName(self): return self.starName
    def getCadence(self): return self.cadence
    def setCadence(self, cadence: float): self.cadence = cadence
