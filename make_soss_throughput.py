# make_soss_throughput.py  — gera CSVs de throughput do JWST/NIRISS SOSS (ordens 1 e 2)
# Funciona no Windows/macOS/Linux
import os, sys
import numpy as np
import matplotlib.pyplot as plt

###### Step 1: Download Required Data: James Webb Space Telescope: https://outerspace.stsci.edu/display/PEN/Pandeia+Engine+Installation #####

# ===================== CONFIGURÁVEL =====================
# Se você já sabe o caminho exato da sua pasta pandeia_data, coloque aqui:
PREFERRED_REF_PATH = r"pandeia_data"  # ex: r"C:\Users\SEU_USER\pandeia_data"  (deixe "" para auto-detectar)
# Grade de λ (µm) para amostrar o throughput:
LAM_MIN_UM, LAM_MAX_UM, NLAM = 0.60, 2.80, 3000
# Ponto de troca simples entre ordens (opção "switch"):
SWITCH_LAMBDA_UM = 1.0
# ========================================================

EXPECTED_REL = os.path.join("jwst", "telescope", "config.json")

def looks_like_refdata(path: str) -> bool:
    return os.path.exists(os.path.join(path, EXPECTED_REL))

def try_paths():
    """Gera candidatos comuns para pandeia_data."""
    here = os.path.abspath(os.getcwd())
    home = os.path.expanduser("~")
    candidates = []
    # 1) Valor preferido do usuário
    if PREFERRED_REF_PATH:
        candidates.append(PREFERRED_REF_PATH)
    # 2) Variável de ambiente já definida
    env = os.environ.get("pandeia_refdata", "")
    if env:
        candidates.append(env)
    # 3) Subpastas padrões perto do projeto
    for base in [here,
                 os.path.dirname(here),
                 os.path.join(here, "pandeia_data"),
                 os.path.join(os.path.dirname(here), "pandeia_data"),
                 os.path.join(home, "pandeia_data")]:
        candidates.append(base)
    # 4) Caso comum de zip duplicado: ...\pandeia_data\pandeia_data
    candidates.append(os.path.join(here, "pandeia_data", "pandeia_data"))
    candidates.append(os.path.join(home, "pandeia_data", "pandeia_data"))
    # Dedup mantendo ordem
    seen, out = set(), []
    for p in candidates:
        if p and p not in seen:
            seen.add(p); out.append(p)
    return out

def find_refdata_or_die():
    for p in try_paths():
        if looks_like_refdata(p):
            return p
    # não achou — mensagem amigável + instruções
    msg = (
        "\n[ERRO] Não encontrei a pasta 'pandeia_data'.\n"
        "Baixe o pacote de referência (pandeia_refdata) e aponte para ele.\n"
        "Passos:\n"
        "  1) Instale o engine:  pip install pandeia.engine\n"
        "  2) Baixe o 'pandeia_refdata' oficial (pasta 'pandeia_data') da STScI (link Box).\n"
        "  3) Descompacte; a pasta precisa conter subpastas como 'jwst/', 'niriss/', 'telescope/' etc.\n"
        "  4) Edite PREFERRED_REF_PATH neste script OU defina a env var pandeia_refdata.\n"
        "Checarei se existe: <pasta>/jwst/telescope/config.json\n"
    )
    raise SystemExit(msg)

# 1) Localiza e define pandeia_refdata
REF = find_refdata_or_die()
os.environ["pandeia_refdata"] = REF
print(f"[INFO] pandeia_refdata = {REF}")
ok_json = os.path.join(REF, EXPECTED_REL)
print(f"[INFO] Verificando: {ok_json} -> {'OK' if os.path.exists(ok_json) else 'FALHOU'}")

# 2) Importa Pandeia Engine
try:
    from pandeia.engine.instrument_factory import InstrumentFactory
except Exception as e:
    raise SystemExit(
        "Falha ao importar pandeia.engine.\n"
        "Instale com:  pip install pandeia.engine\n"
        f"Erro: {e}"
    )

# 3) Config do NIRISS/SOSS (GR700XD/CLEAR) — conforme a documentação do ETC/Pandeia
conf = {
    "detector": {
        "nexp": 1,
        "ngroup": 10,
        "nint": 1,
        "readout_pattern": "nisrapid",
        "subarray": "substrip96",
    },
    "instrument": {
        "aperture": "soss",
        "disperser": "gr700xd",
        "filter": "clear",
        "instrument": "niriss",
        "mode": "soss",
    },
}

# 4) Constrói instrumento e calcula eficiência total por ordem
inst = InstrumentFactory(config=conf)

lam_um = np.linspace(LAM_MIN_UM, LAM_MAX_UM, NLAM)

def get_eff(order: int, lam):
    inst.order = int(order)              # IMPORTANTE para SOSS
    eff = inst.get_total_eff(lam)        # throughput total (óptica+QE+telescope)
    eff = np.clip(np.asarray(eff, float), 0.0, None)
    m = np.nanmax(eff)
    return eff / m if np.isfinite(m) and m > 0 else eff

print("[INFO] Calculando throughput...")
eff1 = get_eff(1, lam_um)   # ordem 1 (domina >~1 µm)
eff2 = get_eff(2, lam_um)   # ordem 2 (domina <~1 µm)

# Combinações úteis
eff_switch = np.where(lam_um < SWITCH_LAMBDA_UM, eff2, eff1)      # “chave” em 1.0 µm
eff_max    = np.maximum(eff1, eff2)                               # envelope
mx = np.nanmax(eff_max)
if np.isfinite(mx) and mx > 0:
    eff_max = eff_max / mx

# 5) Salva CSVs (λ[µm], throughput_normalizado)
def save_csv(fname, lam, thr):
    import numpy as np
    np.savetxt(fname, np.column_stack([lam, thr]), delimiter=",",
               header="lambda_um,throughput_norm", comments="")
    print(f"[OK] salvo: {fname}")

save_csv("niriss_soss_thr_order1.csv", lam_um, eff1)
save_csv("niriss_soss_thr_order2.csv", lam_um, eff2)
save_csv("niriss_soss_thr_combined_switch.csv", lam_um, eff_switch)
save_csv("niriss_soss_thr_combined_max.csv", lam_um, eff_max)

# 6) Plot rápido
plt.figure(figsize=(9,5))
plt.plot(lam_um, eff1, label="Order 1 (norm.)", linewidth=2)
plt.plot(lam_um, eff2, label="Order 2 (norm.)", linewidth=2)
plt.plot(lam_um, eff_switch, label=f"Combined (switch @ {SWITCH_LAMBDA_UM:.2f} µm)", linewidth=2)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Relative throughput")
plt.title("JWST/NIRISS SOSS total throughput (Pandeia)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("\n[INFO] Use um dos CSVs acima no seu script (THR_CSV). "
      "Recomendo 'niriss_soss_thr_combined_switch.csv'.")
