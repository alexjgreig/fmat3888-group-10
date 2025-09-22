# =============================================================================
# MC (Euler/Milstein + bridge) vs CN (PDE) — compact convergence comparison
# - Two figures:
# MC path convergence + CN time-step convergence
# MC step convergence + CN space-node convergence
# =============================================================================
import os, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Make local package importable (same folder)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---- MC side 
from euler_milstein_barrier_put import (
    prepare_pltr_market_inputs as prepare_market,  # market/IV selection
    sweep_vs_paths as mc_sweep_vs_paths,
    sweep_vs_steps as mc_sweep_vs_steps,
    _market_badge_text as market_badge_text,
    _add_badge as add_badge,
)

# ---- CN side
from CN_Method_v1 import (
    cn_sweep_vs_time,
    cn_sweep_vs_space,
)
from option_pricing.data.yahoo_data import OptionDataPuller

# ----Settings 
PATHS_LIST            = [100, 1_000, 10_000, 100_000, 1_000_000]
STEPS_LIST            = [50, 100, 250, 500, 1000]
FIXED_STEPS_FOR_PATHS = 250
FIXED_PATHS_FOR_STEPS = 100_000
CN_REF_GRID           = (1600, 1600)   # (M_ref, N_ref)
CN_FIXED_M            = 1200
CN_FIXED_N            = 1200
SEED_PATHS            = 4242
SEED_STEPS            = 2025
CHUNK_SIZE            = 200_000

# ----- Main 
if __name__ == "__main__":
    # Pull market once (consistent with your other scripts)
    puller = OptionDataPuller("PLTR")
    market = prepare_market(puller, barrier_pct=0.85, prefer_side="auto")
    S0, K, B, r, sigma, T = (market[k] for k in ("S0","K","B","r","sigma","T"))
    r = 0.035
    M_ref, N_ref = CN_REF_GRID

    # CN sweeps (time & space). cn_time_ref serves as our CN reference line.
    cn_time_df, cn_time_ref, _   = cn_sweep_vs_time(
        S0, K, B, r, sigma, T,
        steps_list=STEPS_LIST, M_fixed=CN_FIXED_M,
        ref_grid=CN_REF_GRID, rebate=0.0, use_rannacher=True
    )
    cn_space_df, cn_space_ref, _ = cn_sweep_vs_space(
        S0, K, B, r, sigma, T,
        M_list=[50, 100, 250, 500, 1000], N_fixed=CN_FIXED_N,
        ref_grid=CN_REF_GRID, rebate=0.0, use_rannacher=True
    )
    cn_ref = float(cn_time_ref)

    # MC sweeps (paths & steps/year)
    mc_paths = mc_sweep_vs_paths(
        market, PATHS_LIST, FIXED_STEPS_FOR_PATHS,
        seed=SEED_PATHS, chunk_size=CHUNK_SIZE
    )
    mc_steps = mc_sweep_vs_steps(
        market, STEPS_LIST, FIXED_PATHS_FOR_STEPS,
        seed=SEED_STEPS, chunk_size=CHUNK_SIZE
    )

    # ==== Figure 1 

    fig, axes = plt.subplots(1, 2, figsize=(12,5), sharey=False)

    # MC paths
    ax = axes[0]
    x = np.array(mc_paths["paths"])
    e, se = np.array(mc_paths["euler_price"]),    1.96*np.array(mc_paths["euler_se"])
    m, sm = np.array(mc_paths["milstein_price"]), 1.96*np.array(mc_paths["milstein_se"])
    ax.errorbar(x, e, yerr=se, fmt="o-", label="Euler (±95% CI)")
    ax.errorbar(x, m, yerr=sm, fmt="s-", label="Milstein (±95% CI)")
    ax.axhline(cn_ref, color="C1", linestyle="--",
               label=f"CN ref (M=N={M_ref}) = {cn_ref:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("MC: number of paths (log scale)")
    ax.set_ylabel("Down-and-Out PUT price")
    ax.set_title("MC Path Convergence (bridge KO)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    add_badge(ax, market_badge_text(market), loc="upper left")
    add_badge(ax, f"Fixed steps/year = {FIXED_STEPS_FOR_PATHS}", loc="lower left")

    # CN time
    ax = axes[1]
    ax.plot(cn_time_df["N_steps"], cn_time_df["Price"], marker="o",
            label=f"CN (M={CN_FIXED_M}, vary N)")
    ax.axhline(cn_time_ref, color="C1", linestyle="--",
               label=f"CN ref (M=N={M_ref}) = {cn_time_ref:.4f}")
    ax.set_xlabel("CN: time steps N")
    ax.set_ylabel("Down-and-Out PUT price")
    ax.set_title("CN Time-Step Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    add_badge(ax, market_badge_text(market), loc="upper left")

    fig.tight_layout()

    # ==== Figure 2 
    fig, axes = plt.subplots(1, 2, figsize=(12,5), sharey=False)

    # MC steps/year
    ax = axes[0]
    x = np.array(mc_steps["steps"])
    e, se = np.array(mc_steps["euler_price"]),    1.96*np.array(mc_steps["euler_se"])
    m, sm = np.array(mc_steps["milstein_price"]), 1.96*np.array(mc_steps["milstein_se"])
    ax.errorbar(x, e, yerr=se, fmt="o-", label="Euler (±95% CI)")
    ax.errorbar(x, m, yerr=sm, fmt="s-", label="Milstein (±95% CI)")
    ax.axhline(cn_ref, color="C1", linestyle="--",
               label=f"CN ref (M=N={M_ref}) = {cn_ref:.4f}")
    ax.set_xlabel("MC: time steps per year")
    ax.set_ylabel("Down-and-Out PUT price")
    ax.set_title("MC Time-Step Convergence (bridge KO)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    add_badge(ax, market_badge_text(market), loc="upper left")
    add_badge(ax, f"Fixed paths = {FIXED_PATHS_FOR_STEPS:,}", loc="lower left")

    # CN space nodes
    ax = axes[1]
    ax.plot(cn_space_df["M_nodes"], cn_space_df["Price"], marker="o",
            label=f"CN (N={CN_FIXED_N}, vary M)")
    ax.axhline(cn_space_ref, color="C1", linestyle="--",
               label=f"CN ref (M=N={M_ref}) = {cn_space_ref:.4f}")
    ax.set_xlabel("CN: space nodes M")
    ax.set_ylabel("Down-and-Out PUT price")
    ax.set_title("CN Space Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    add_badge(ax, market_badge_text(market), loc="upper left")

    fig.tight_layout()
    plt.show()

