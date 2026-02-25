"""
Global configuration constants and matplotlib settings.
"""

import os
import matplotlib.pyplot as plt

# ── Reproducibility ──
RANDOM_STATE = 42

# ── Cross-validation & statistical parameters ──
N_BOOTSTRAPS = 1000
N_CV_SPLITS = 10
LHS_N_ITER = 50
ALPHA = 0.05

# ── Memory-safe settings ──
# Reduce n_jobs if you hit MemoryError on the Combined model.
# Set to 1 for sequential (safest), or 2-4 for moderate parallelism.
N_JOBS_SAFE = 2
N_JOBS_FAST = 2
MAX_ESTIMATORS_COMBINED = 800
MAX_ESTIMATORS_DEFAULT = 1500

# ── Output ──
OUTPUT_DIR = "revision_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Matplotlib defaults ──
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── AMP-PD cohort conventions ──
# PD- prefix = PDBP (internal development)
# PP- prefix = PPMI (external validation)

# ── Proteomics data paths ──
PROTEOMICS_PANELS = {
    "oncology":        r"S:/AMP-PD/releases_2023_v4release_1027_proteomics-PLA-PPEA-D03_olink-explore_protein-expression_PLA-PPEA-D03_oncology.csv",
    "neurology":       r"S:/AMP-PD/releases_2023_v4release_1027_proteomics-PLA-PPEA-D03_olink-explore_protein-expression_PLA-PPEA-D03_neurology.csv",
    "inflammation":    r"S:/AMP-PD/releases_2023_v4release_1027_proteomics-PLA-PPEA-D03_olink-explore_protein-expression_PLA-PPEA-D03_inflammation.csv",
    "cardiometabolic": r"S:/AMP-PD/releases_2023_v4release_1027_proteomics-PLA-PPEA-D03_olink-explore_protein-expression_PLA-PPEA-D03_cardiometabolic.csv",
}

# ── RNA-seq data path ──
RNASEQ_PATH = "aligned_rnaseq_data_with_targets.csv"

# ── DEG file ──
DEG_PATH = r"S:/Genes/DEG_Analysis.csv"

# ── Clinical data ──
CLINICAL_PATH = r"C:\Users\NM\Documents\Python Scripts\results\tables\clinical_unified.csv"

# ── Prognostic analysis parameters ──
MILESTONE_DELTA = 5        # UPDRS-III increase for "clinically meaningful worsening"
TARGET_MONTH = 24           # target follow-up month
MONTH_WINDOW = 6            # +/- window around target month
PSI_QUANTILE_SPLIT = 3     # tertiles for KM curves

# ── PD gene sets (KEGG, symbol-based) ──
PD_GENE_SETS_SYMBOLS = {
    "KEGG_PARKINSONS_DISEASE": [
        "SNCA", "LRRK2", "PARK7", "PINK1", "PRKN", "ATP13A2",
        "FBXO7", "VPS35", "GBA", "UCHL1", "HTRA2", "DNAJC6",
    ],
    "KEGG_OXIDATIVE_PHOSPHORYLATION": [
        "NDUFA1", "NDUFA2", "NDUFA3", "NDUFB1", "NDUFB2",
        "SDHA", "SDHB", "UQCRB", "COX5A", "ATP5F1A",
        "NDUFS1", "NDUFS2", "COX7A2", "ATP5MC1",
    ],
    "KEGG_UBIQUITIN_PROTEASOME": [
        "UBA1", "UBE2D1", "UBE2D2", "UBE2L3", "PSMA1",
        "PSMB1", "PSMC1", "PSMD1", "USP14", "UCHL5",
        "UBB", "UBC", "PSMA2", "PSMB2",
    ],
    "KEGG_NEUROTROPHIN_SIGNALING": [
        "BDNF", "NTRK1", "NTRK2", "NGF", "MAPK1",
        "MAPK3", "AKT1", "PIK3CA", "SOS1", "GRB2",
        "BRAF", "RAF1", "MEK1", "MAP2K1",
    ],
    "KEGG_DOPAMINERGIC_SYNAPSE": [
        "TH", "DDC", "SLC6A3", "DRD1", "DRD2",
        "MAOA", "MAOB", "COMT", "SLC18A2", "PPP1CA",
        "GNAL", "GNG7", "PPP1R1B",
    ],
    "KEGG_LYSOSOME": [
        "GBA", "LAMP1", "LAMP2", "CTSD", "CTSB",
        "ATP6V0A1", "ATP6V1A", "SCARB2", "NPC1", "NPC2",
    ],
    "KEGG_APOPTOSIS": [
        "CASP3", "CASP9", "BAX", "BCL2", "CYCS",
        "APAF1", "BID", "XIAP", "DIABLO", "TP53",
    ],
}

# ── Model color palette ──
MODEL_COLORS = {
    "Combined": "#2196F3",
    "Proteomics only": "#4CAF50",
    "RNA-seq only": "#FF9800",
    "Balanced Early Fusion": "#9C27B0",
    "Late Fusion (avg)": "#E91E63",
    "Late Fusion (weighted)": "#00BCD4",
    "Stacking": "#795548",
    "Early Fusion": "#3F51B5",
}
