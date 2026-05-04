#!/usr/bin/env python3
"""Build canonical leaderboard CSV + LaTeX partial.

Combines results/leaderboard_params.csv (params from state_dict.numel())
with hardcoded test_l2 numbers from RESEARCH_STATE.md Session 9–11. Output:
  results/leaderboard.csv          — canonical leaderboard (model, L2, params, ...)
  results/leaderboard.tex          — LaTeX booktabs table
  results/pareto_data.csv          — (params, L2, model) for Pareto plot
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMS_CSV = ROOT / "tables" / "leaderboard_params.csv"
OUT_CSV    = ROOT / "tables" / "leaderboard.csv"
OUT_TEX    = ROOT / "tables" / "leaderboard.tex"
OUT_PARETO = ROOT / "tables" / "pareto_data.csv"

# (display_name, L2_best, params_key_in_audit_csv, provenance, on_pareto_frontier_excl_transformer)
# Provenance: published-baseline rows use \citet{<bibkey>}; our own runs use
# "this work" with an internal run identifier (\texttt{...}) appended for cross-reference
# against the supplementary code repository's runs.csv.
ROWS = [
    # ours, in order of empirical interest
    ("\\textbf{SpectraNet (ours, headline)}",                 0.0822, "AR-2D resi+ts linear-head", "this work (\\texttt{J23})", True),
    ("SpectraNet (ours), $w{=}32$ + residual, seed 0",        0.0836, "AR-2D w32 resi (J14)",      "this work (\\texttt{J14})",   True),
    ("SpectraNet (ours), $w{=}32$ + residual, seed 1",        0.0850, "AR-2D w32 resi seed1",      "this work (\\texttt{J3222})", False),
    ("SpectraNet (ours), $M{=}20$",                           0.0787, "AR-2D resi M20",            "this work (\\texttt{J19})",   True),
    ("SpectraNet (ours), $M{=}16$",                           0.0792, "AR-2D resi M16",            "this work (\\texttt{J18})",   True),
    ("SpectraNet (ours), $w{=}48$ plain",                     0.0836, "AR-2D w=48 plain",          "this work (\\texttt{3190})",  False),
    ("SpectraNet (ours), $w{=}64$ plain",                     0.0869, "AR-2D w=64 plain",          "this work (\\texttt{3199})",  False),
    ("SpectraNet (ours), $w{=}20$ plain",                     0.0941, "AR-2D w=20",                "this work (\\texttt{3187})",  True),
    ("SpectraNet (ours), $w{=}32$ + grouped channel-mixing MLP", 0.0854, "AR-2D resi mg4 (E4)",    "this work (\\texttt{J21})",   False),
    ("SpectraNet (ours), $w{=}32$ + spectral envelope",       0.0849, "AR-2D resi env (E5)",       "this work (\\texttt{J24})",   False),
    ("SpectraNet (ours), $w{=}32$ + disk truncation",         0.0893, "AR-2D resi disk (E2)",      "this work (\\texttt{J20})",   False),
    ("SpectraNet (ours), 3D-spectral variant",                0.1006, None,                         "this work (\\texttt{3188})",  False),
    # baselines (canonical PyTorch param counts from leaderboard_params.csv)
    ("Transformer (NSL, full softmax)",             0.0284, "Transformer",             "\\citet{wu2024nsl}",       None),
    ("Galerkin Transformer",                        0.1097, "Galerkin_Transformer",    "\\citet{cao2021galerkin}", False),
    ("FNO seed 0",                                  0.1024, "FNO",                     "\\citet{li2020fno}",        False),
    ("FNO seed 1",                                  0.1069, "FNO_seed1",               "\\citet{li2020fno} (seed 1)", False),
    ("U-FNO",                                       0.1218, "U_FNO",                   "\\citet{wen2022ufno}",     False),
    ("U-Net (NSL)",                                 0.1387, "U_Net",                   "\\citet{wu2024nsl}",       False),
    ("FactFormer",                                  0.1639, "FactFormer",              "\\citet{li2023factformer}",False),
    ("U-NO",                                        0.1697, "U_NO",                    "\\citet{rahman2023uno}",    False),
    ("SpectraNet (ours), single-shot variant",      0.1942, None,                       "this work (\\texttt{3123})", False),
    ("MWT",                                         0.1944, "MWT",                     "\\citet{gupta2021mwt}",     False),
    ("LSM",                                         0.1951, "LSM",                     "\\citet{wu2023lsm}",        False),
    ("OFormer",                                     0.2162, "OFormer",                 "\\citet{li2023oformer}",    False),
    ("Transolver",                                  0.2247, "Transolver",              "\\citet{wu2024transolver}", False),
    ("F-FNO",                                       0.2331, "F_FNO",                   "\\citet{tran2021ffno}",     False),
    ("GNOT",                                        0.2362, "GNOT",                    "\\citet{hao2023gnot}",      False),
    ("KNO2d",                                       0.3092, "KNO2d",                   "\\citet{xiong2024kno}",     False),
    ("CNO",                                         0.3259, "CNO",                     "\\citet{raonic2023cno}",    False),
    ("ONO",                                         0.3443, "ONO",                     "\\citet{xiao2024ono}",      False),
    ("Persistence (trivial floor)",                 0.7481, None,                       "this work",                  False),
]

# load params lookup
params_lookup = {}
with PARAMS_CSV.open() as f:
    for r in csv.DictReader(f):
        params_lookup[r["model"]] = int(r["sum_p_numel"])

# special params not in params_lookup yet — fill in known canonical counts (state_dict.numel basis)
_extra = {
    None: None,  # placeholder for "unknown / single-shot or AR-3D not yet in audit CSV"
}
# AR-3D was audited separately
params_lookup["AR-3D w=20 (audited 2026-04-30)"] = 8_574_657

def resolve_params(name, pk):
    if "3D-spectral" in name:
        return 8_574_657
    if "single-shot" in name:
        return 14_490_352
    if "Persistence" in name:
        return 0
    return params_lookup.get(pk) if pk else None


# emit canonical leaderboard CSV (drop LaTeX markup from model field for raw CSV)
import re
_LATEX_CMD = re.compile(r"\\(?:textbf|texttt|emph|citet)\{([^}]*)\}")
_BRACED    = re.compile(r"\{([^{}]*)\}")
def csv_safe(name):
    s = name
    # peel off \textbf{...}, \texttt{...}, \citet{...} (keep contents)
    while True:
        ns = _LATEX_CMD.sub(r"\1", s)
        if ns == s:
            break
        s = ns
    # collapse {=} → = and strip remaining curly groups (keep contents)
    s = s.replace("{=}", "=")
    while True:
        ns = _BRACED.sub(r"\1", s)
        if ns == s:
            break
        s = ns
    s = (s.replace("$", "")
         .replace("\\,", " ")
         .replace("\\\\", " "))
    return s.strip()

with OUT_CSV.open("w") as f:
    w = csv.writer(f)
    w.writerow(["model", "test_L2_best", "params", "provenance", "on_pareto_frontier"])
    for name, L2, pk, src, on_p in ROWS:
        params = resolve_params(name, pk)
        w.writerow([csv_safe(name), f"{L2:.4f}",
                    params if params is not None else "TBD",
                    csv_safe(src),
                    "" if on_p is None else ("Y" if on_p else "")])

# emit Pareto data (only models with known params)
with OUT_PARETO.open("w") as f:
    w = csv.writer(f)
    w.writerow(["model_short", "params", "test_L2", "is_ours", "is_pareto_frontier"])
    for name, L2, pk, src, on_p in ROWS:
        params = resolve_params(name, pk)
        if params is None:
            continue
        is_ours = "ours" in name.lower()
        short = csv_safe(name)
        w.writerow([short, params, f"{L2:.4f}", "Y" if is_ours else "",
                    "Y" if (on_p and on_p is not None) else ""])

# emit LaTeX booktabs table — sorted by test_L2 ascending
sorted_rows = sorted(ROWS, key=lambda r: r[1])
with OUT_TEX.open("w") as f:
    f.write("% Auto-generated by scripts/figures/make_leaderboard.py — do not edit by hand.\n")
    f.write("\\begin{table}[t]\n\\centering\n")
    f.write("\\caption{Unified-protocol leaderboard, NS $\\nu{=}10^{-5}$, $64{\\times}64$. ")
    f.write("Params reported as $\\sum_p p.\\text{numel}()$ (PyTorch convention). ")
    f.write("Pareto-frontier rows (excluding the full-attention Transformer) shaded. ")
    f.write("Row provenance is the published source for baselines and ``this work'' for ")
    f.write("our own runs; an internal run identifier (e.g.\\ \\texttt{J23}) is appended ")
    f.write("for our runs and is mapped to launch date, SLURM job, and trained-checkpoint ")
    f.write("path in the supplementary code repository (\\texttt{runs.csv}).}\n")
    f.write("\\label{tab:leaderboard}\n")
    f.write("\\begin{tabular}{lrrl}\n\\toprule\n")
    f.write("Model & Test rel.\\ $L^2$ & Params & Provenance \\\\\n\\midrule\n")
    for name, L2, pk, src, on_p in sorted_rows:
        params = resolve_params(name, pk)
        if params == 0:
            params_str = "0"
        elif params:
            params_str = f"{params/1e6:.2f}\\,M"
        else:
            params_str = "—"
        safe_name = name.replace("&", "\\&")
        prefix = "\\rowcolor{gray!10} " if on_p else ""
        f.write(f"{prefix}{safe_name} & {L2:.4f} & {params_str} & {src} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print("wrote", OUT_CSV)
print("wrote", OUT_TEX)
print("wrote", OUT_PARETO)
