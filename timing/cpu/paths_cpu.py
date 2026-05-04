"""Resolve local copies of the model code + checkpoints rsynced from the server.

Layout (created during setup):
  TIME_ANALYSIS/_remote_src/Baselines/...
  TIME_ANALYSIS/_remote_src/AttentionAblation/...
"""
import os

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(THIS_DIR)
TIME_ROOT   = os.path.dirname(SCRIPTS_DIR)
SRC_ROOT    = os.path.join(TIME_ROOT, "_remote_src")

NSL_DIR        = os.path.join(SRC_ROOT, "Baselines", "NSL")
OF_DIR         = os.path.join(SRC_ROOT, "Baselines", "OFormer", "uniform_grids")
KNO_DIR        = os.path.join(SRC_ROOT, "Baselines", "KoopmanLab")
FF_DIR         = os.path.join(SRC_ROOT, "Baselines", "FactFormer")
TS_DIR         = os.path.join(SRC_ROOT, "Baselines", "Transolver", "PDE-Solving-StandardBenchmark")
GN_DIR         = os.path.join(SRC_ROOT, "Baselines", "gnot")
ONO_DIR        = os.path.join(SRC_ROOT, "Baselines", "Orthogonal-Neural-operator")
ABL_DIR        = os.path.join(SRC_ROOT, "AttentionAblation")
SUNET_SRC      = os.path.join(ABL_DIR, "ns_sunet_goldstd.py")

RESULTS_DIR = os.path.join(TIME_ROOT, "results_cpu")
PLOTS_DIR   = os.path.join(TIME_ROOT, "plots_cpu")
