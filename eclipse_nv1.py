# (mantém todos os imports relevantes; removido o mitigation)
from IPython.display import HTML
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot
from star import Star
from Planeta import Planeta
from verify import calculaLat
from keplerAux import keplerfunc
import matplotlib.animation as animation
import os
import platform
import time
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def curvaLuz_jit_abs(x0, y0, tamanhoMatriz, raioPlanetaPixel, estrelaManchada_flat):
    """
    Retorna o FLUXO ABSOLUTO (soma dos pixels não ocultados) para a posição (x0,y0).
    Não faz nenhuma normalização.
    """
    N = tamanhoMatriz
    r2 = raioPlanetaPixel * raioPlanetaPixel
    total = 0.0
    for idx in prange(N * N):
        row = idx // N
        col = idx %  N
        dx = row - y0
        dy = col - x0
        if (dx*dx + dy*dy) > r2:
            total += estrelaManchada_flat[idx]
    return total

class Eclipse:
    def __init__(self, Nx, Ny, raio_estrela_pixel, estrela_manchada: Star, planeta_: Planeta):
        self.Nx = Nx
        self.Ny = Ny
        self.tamanhoMatriz = self.Nx
        self.raio_estrela_pixel = raio_estrela_pixel
        self.estrela_ = estrela_manchada
        self.estrela_matriz = estrela_manchada.getMatrizEstrela()
        self.planeta_ = planeta_
        # Curva relativa (compat) e absoluta (nova fonte de verdade)
        self.curvaLuz = None
        self.curvaAbs = None
        self.maxFlux = None
        self.error = None

    def setTempoHoras(self, intervalo, n_points):
        self.intervaloTempo = intervalo
        self.tempoHoras = (np.arange(int(n_points)) - float(n_points) / 2.0) * self.intervaloTempo / 60.0

    def criarEclipse(self, plot_anim, plot_graph):

        self.plot_anim = plot_anim
        self.plot_graph = plot_graph

        N_points, self.intervaloTempo, self.tempoTotal = self.calculateAutomaticNPoints()
        self.setTempoHoras(self.intervaloTempo, N_points)

        # Fallback
        if len(self.tempoHoras) == 0:
            print(">>> [WARNING] No time points found. Falling back to a single point.")
            self.tempoHoras = np.array([0.0])
            self.curvaAbs = np.array([float(np.sum(self.estrela_matriz))])
            self.maxFlux  = float(self.curvaAbs[0])
            self.curvaLuz = np.array([1.0])
            return

        tamanhoMatriz = self.tamanhoMatriz
        dtor = np.pi / 180.
        semiEixoPixel = self.planeta_.semiEixoRaioStar * self.raio_estrela_pixel

        nk = 2 * np.pi / (self.planeta_.periodo * 24)
        Tp = self.planeta_.periodo * self.planeta_.anom / 360. * 24.
        m = nk * (self.tempoHoras - Tp)

        eccanom = keplerfunc(m, self.planeta_.ecc)
        xs = semiEixoPixel * (np.cos(eccanom) - self.planeta_.ecc)
        ys = semiEixoPixel * (np.sqrt(1 - self.planeta_.ecc**2) * np.sin(eccanom))

        ang = self.planeta_.anom * dtor - (np.pi / 2)
        xp = xs * np.cos(ang) - ys * np.sin(ang)
        yp = xs * np.sin(ang) + ys * np.cos(ang)

        ie = np.where(np.abs(self.tempoHoras) == np.min(np.abs(self.tempoHoras)))[0]
        xplaneta = xp - xp[ie[0]]
        yplaneta = yp * np.cos(self.planeta_.anguloInclinacao * dtor)

        pp = np.where((np.abs(xplaneta) < 1.2 * tamanhoMatriz / 2) &
                      (np.abs(yplaneta) < tamanhoMatriz / 2))[0]

        # Fluxo OOT do disco (baseline absoluto)
        maxCurvaLuz = float(np.sum(self.estrela_matriz))
        self.maxFlux = maxCurvaLuz

        # Inicializa curvas
        self.curvaAbs = np.full(len(self.tempoHoras), maxCurvaLuz, dtype=np.float64)

        if len(pp) == 0:
            self.curvaLuz = np.ones(len(self.tempoHoras), dtype=np.float64)
            return

        xplan = xplaneta[pp] + tamanhoMatriz / 2
        yplan = yplaneta[pp] + tamanhoMatriz / 2

        raioPlanetaPixel = float(self.planeta_.raioPlanetaRstar * self.raio_estrela_pixel)
        estrela_matriz_flat = np.ascontiguousarray(self.estrela_matriz, dtype=np.float64).ravel()

        # preenche somente as amostras dentro da janela simulada
        for i in range(len(pp)):
            x0 = xplan[i]
            y0 = yplan[i]
            self.curvaAbs[pp[i]] = curvaLuz_jit_abs(
                x0, y0, tamanhoMatriz, raioPlanetaPixel,
                estrela_matriz_flat
            )

        # relativa (sem estabilização)
        rel = self.curvaAbs / max(self.maxFlux, 1e-30)
        self.curvaLuz = rel

        if plot_anim:
            ...
        elif plot_graph:
            plt.figure(figsize=(10, 5))
            plt.plot(self.tempoHoras, self.curvaLuz)
            plt.xlabel('Time from Transit Center [h]')
            plt.ylabel('Relative Flux')
            plt.title('Transit Light Curve')
            plt.show()

    # ---------- GETTERS ----------
    def getMaxFlux(self):
        return float(self.maxFlux if self.maxFlux is not None else np.sum(self.estrela_matriz))

    def getAbsFluxCurve(self):
        if self.curvaAbs is None:
            return None
        return np.asarray(self.curvaAbs, float)

    def calculaLatMancha(self):
        latsugerida = calculaLat(self.planeta_.semiEixoRaioStar, self.planeta_.anguloInclinacao)
        print("A latitude sugerida para que a mancha influencie na curva de luz da estrela é:", latsugerida)
        return latsugerida

    def calculaLongMancha(self, a, time, lat):
        latitude_rad = math.radians(lat)
        angle = math.radians(90) - (math.radians(360) * time) / (24 * self.planeta_.periodo)
        longitude = math.degrees(math.asin((a * math.cos(angle)) / math.cos(abs(latitude_rad))))
        return longitude

    def getMatrizTransformada(self, estrela):
        if isinstance(estrela, Star):
            estrela = estrela.getMatrizEstrela()
        if not isinstance(estrela, np.ndarray):
            raise TypeError("O parâmetro 'estrela' deve ser uma matriz NumPy.")
        return np.array(estrela, dtype=np.float64)

    def getTempoTransito(self):
        return self.tempoTotal

    def getTempoHoras(self):
        return self.tempoHoras

    def getCurvaLuz(self):
        return self.curvaLuz

    def setEstrela(self, estrela):
        self.estrela_matriz = estrela

    def getError(self):
        return self.error

    def calculateAutomaticNPoints(self,
        percent_resolution=0.01,
        min_interval_limit=0.1,
        max_interval_limit=10.0):
        dtor = np.pi / 180.
        a_rs = self.planeta_.semiEixoRaioStar
        inc = self.planeta_.anguloInclinacao
        periodo = self.planeta_.periodo

        latitude_transit = -np.arcsin(a_rs * np.cos(inc * dtor)) / dtor
        transit_duration = 2 * (90. - np.arccos(np.cos(latitude_transit * dtor) / a_rs) / dtor) * periodo / 360 * 24.
        total_time = 3 * transit_duration
        self.tempoTotal = total_time

        interval_adaptive = (transit_duration * 60) * percent_resolution
        interval_minutes = max(min_interval_limit, min(interval_adaptive, max_interval_limit))
        N_points = int(np.ceil((total_time * 60) / interval_minutes))

        return N_points, interval_minutes, total_time
