# === Down-and-Out Put from OptionDataPuller data ==============================
import math
from typing import Tuple, List, Dict, Optional
import warnings, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from option_pricing.data.yahoo_data import OptionDataPuller

# ----- Tridiagonal solver -----
def thomas(l: np.ndarray, d: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(d)
    c = u.copy(); dd = d.copy(); bb = b.copy()
    for i in range(1, n):
        w = l[i] / dd[i-1]
        dd[i] -= w * c[i-1]
        bb[i] -= w * bb[i-1]
    x = np.zeros(n)
    x[-1] = bb[-1] / dd[-1]
    for i in range(n-2, -1, -1):
        x[i] = (bb[i] - c[i]*x[i+1]) / dd[i]
    return x

# ----- Vanilla BS put (q=0) -----
def bs_put_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0: return max(K - S0, 0.0)
    d1 = (math.log(S0/K) + (r + 0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    N = lambda x: 0.5*(1.0 + math.erf(x/math.sqrt(2)))
    return K*math.exp(-r*T)*N(-d2) - S0*N(-d1)

# ----- Crank–Nicolson Down-and-Out Put (continuous KO, log-space) -----
def price_do_put_cn(
    S0: float, K: float, B: float, r: float, sigma: float, T: float,
    Smax_factor: float = 6.0, M: int = 480, N: int = 480,
    rebate: float = 0.0, use_rannacher: bool = True, return_grid: bool = False
) -> Tuple[float, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    if not (B < S0): raise ValueError("Down-and-out requires B < S0.")
    Smax = Smax_factor * max(S0, K)
    x_min, x_max = math.log(B), math.log(Smax)
    dx = (x_max - x_min) / M
    dt = T / N
    x = x_min + dx*np.arange(M+1)
    S = np.exp(x)

    a = 0.5*sigma*sigma
    b = r - 0.5*sigma*sigma
    c = -r

    V = np.maximum(K - S, 0.0)
    V[S <= B] = rebate

    m = M - 1
    if m <= 0: raise ValueError("Increase M (need at least 2 interior nodes).")

    A_lower = (a/dx**2) - (b/(2*dx))
    A_main  = (-2*a/dx**2) + c
    A_upper = (a/dx**2) + (b/(2*dx))

    l_lhs = -0.5*dt*A_lower * np.ones(m)
    d_lhs = (1.0 - 0.5*dt*A_main) * np.ones(m)
    u_lhs = -0.5*dt*A_upper * np.ones(m)

    l_rhs = +0.5*dt*A_lower * np.ones(m)
    d_rhs = (1.0 + 0.5*dt*A_main) * np.ones(m)
    u_rhs = +0.5*dt*A_upper * np.ones(m)

    l_be = -dt*A_lower * np.ones(m)
    d_be = (1.0 - dt*A_main) * np.ones(m)
    u_be = -dt*A_upper * np.ones(m)

    def step_once(Vn: np.ndarray, scheme: str = "CN") -> np.ndarray:
        V_left_n = V_left_np1 = rebate
        V_right_n = V_right_np1 = 0.0
        Vn_int = Vn[1:M]

        if scheme == "BE":
            rhs = Vn_int.copy()
            rhs[0] += dt*A_lower*V_left_np1
            V_int_np1 = thomas(l_be, d_be, u_be, rhs)
        else:
            rhs = d_rhs*Vn_int
            rhs[1:]  += l_rhs[1:]*Vn_int[:-1]
            rhs[:-1] += u_rhs[:-1]*Vn_int[1:]
            rhs[0]  += l_rhs[0]*V_left_n
            rhs[-1] += u_rhs[-1]*V_right_n
            rhs[0]  += -l_lhs[0]*V_left_np1
            rhs[-1] += -u_lhs[-1]*V_right_np1
            V_int_np1 = thomas(l_lhs, d_lhs, u_lhs, rhs)

        Vnp1 = Vn.copy()
        Vnp1[0], Vnp1[1:M], Vnp1[M] = V_left_np1, V_int_np1, V_right_np1
        Vnp1[S <= B] = rebate
        return Vnp1

    if use_rannacher:
        V = step_once(V, "BE"); V = step_once(V, "BE")
        for _ in range(N-2): V = step_once(V, "CN")
    else:
        for _ in range(N): V = step_once(V, "CN")

    x0 = math.log(S0)
    i_float = (x0 - x_min)/dx
    i = int(np.clip(i_float, 1, M-1)); w = i_float - i
    price = (1-w)*V[i] + w*V[i+1]
    return (float(price), (S, V)) if return_grid else (float(price), (None, None))

# ----- Monte Carlo continuous barrier via Brownian bridge (exact GBM) -----
def mc_do_put_bb(
    S0: float, K: float, B: float, r: float, sigma: float, T: float,
    n_paths: int = 120_000, n_steps: int = 252, seed: int = 42
) -> Tuple[float, float]:
    if B >= S0: return 0.0, 0.0
    rng = np.random.default_rng(seed)
    dt = T/n_steps
    nudt = (r - 0.5*sigma*sigma)*dt
    sdt = sigma*math.sqrt(dt)

    half = n_paths // 2
    Z = rng.standard_normal((half, n_steps))
    Z = np.vstack([Z, -Z])
    n_paths = Z.shape[0]

    S = np.full(n_paths, S0, float)
    alive = np.ones(n_paths, bool)
    logB = math.log(B)
    disc = math.exp(-r*T)

    for t in range(n_steps):
        if not alive.any(): break
        z = Z[:, t]
        S_next = S * np.exp(nudt + sdt*z)

        idx = np.where(alive)[0]
        x = np.log(S[idx]) - logB
        y = np.log(S_next[idx]) - logB
        below = (S[idx] <= B) | (S_next[idx] <= B)

        hit_prob = np.zeros_like(x)
        mask = ~below
        if mask.any():
            xb = x[mask]; yb = y[mask]
            hit_prob[mask] = np.exp(-2.0*xb*yb/(sigma*sigma*dt))
            hit_prob = np.clip(hit_prob, 0.0, 1.0)

        u = rng.random(hit_prob.shape)
        hit = np.zeros_like(hit_prob, dtype=bool)
        hit[below] = True
        hit[mask] = (u[mask] < hit_prob[mask])

        # update
        alive[idx[hit]] = False
        S[idx[~hit]] = S_next[idx[~hit]]

    payoff = np.where(alive, np.maximum(K - S, 0.0), 0.0)
    disc_payoff = disc*payoff
    price = float(disc_payoff.mean())
    se = float(disc_payoff.std(ddof=1)/math.sqrt(n_paths))
    return price, se

# ----- Helpers to pick 1Y expiry, ATM strike, and IV from your puller -----
def pick_one_year_expiration(expirations: List[str], target_days: int = 365) -> Optional[str]:
    if not expirations: return None
    today = pd.Timestamp.now().normalize()
    def days_to(exp_str: str) -> int:
        return int((pd.to_datetime(exp_str) - today).days)
    # choose the date minimizing |days - target_days| but days>0
    exps_pos = [e for e in expirations if days_to(e) > 0]
    if not exps_pos: return None
    exps_pos.sort(key=lambda e: abs(days_to(e) - target_days))
    return exps_pos[0]

def get_atm_strike_and_iv(df: pd.DataFrame, S0: float, fallback_iv: Optional[float]=None) -> Tuple[float, Optional[float]]:
    if df is None or df.empty:
        return float('nan'), fallback_iv
    dff = df.copy()
    # ensure required cols
    for col in ["strike","mid","calculatedIV","impliedVolatility","yahooIV"]:
        if col not in dff.columns: dff[col] = np.nan
    # ATM = strike nearest to S0 with a valid mid; prefer rows with calculatedIV
    dff = dff[dff["mid"].notna()]
    if dff.empty: return float('nan'), fallback_iv
    dff["atm_gap"] = (dff["strike"] - S0).abs()
    dff.sort_values(["atm_gap","volume","openInterest"], ascending=[True, False, False], inplace=True)
    row = dff.iloc[0]
    K = float(row["strike"])
    iv = row.get("calculatedIV")
    if pd.isna(iv) or not (0.005 < iv < 5.0):
        # fallbacks: our impliedVolatility column (you overwrite with calculatedIV) then Yahoo's IV
        iv = row.get("impliedVolatility")
        if pd.isna(iv) or not (0.005 < iv < 5.0):
            iv = row.get("yahooIV")
        if pd.isna(iv) or not (0.005 < iv < 5.0):
            iv = fallback_iv
    return K, (None if pd.isna(iv) else float(iv))

# ----- Glue: pull live data with your OptionDataPuller and price DO put -----
def price_pltr_do_put_from_puller(
    puller,  # your OptionDataPuller instance
    target_days: int = 365,
    rannacher: bool = True,
    grid: Tuple[int,int] = (480,480),
    do_mc: bool = True
) -> Dict[str, float]:
    # 1) Underlying and expirations
    S0 = float(puller.get_current_price())
    expirations = puller.get_option_expirations()
    exp = pick_one_year_expiration(expirations, target_days=target_days)
    if exp is None:
        raise RuntimeError("Could not find a positive-dated expiration.")

    # 2) Option chain for chosen expiration
    chain = puller.get_option_chain(exp)
    puts = chain.get("puts", pd.DataFrame())
    calls = chain.get("calls", pd.DataFrame())
    exp_date = pd.to_datetime(exp)
    today = pd.Timestamp.now().normalize()
    T = max(0.0, (exp_date - today).days / 365.25)

    # 3) ATM strike + IV (prefer puts; fallback to calls)
    K_put, iv_put = get_atm_strike_and_iv(puts, S0)
    K_call, iv_call = get_atm_strike_and_iv(calls, S0)
    # choose K nearest to S0; pick IV from that side; average if both valid
    candidates = []
    if not math.isnan(K_put) and iv_put is not None: candidates.append(("put", K_put, iv_put))
    if not math.isnan(K_call) and iv_call is not None: candidates.append(("call", K_call, iv_call))
    if not candidates:
        raise RuntimeError("No valid ATM IV found on either side for the chosen expiration.")

    # pick the side with strike closest to S0
    side, K, sigma = sorted(candidates, key=lambda t: abs(t[1]-S0))[0]
    # small smoothing: if both sides available and close, average IVs
    if len(candidates) == 2 and abs(K_put - K_call) <= 0.5:
        sigma = 0.5*(iv_put + iv_call)

    # 4) Barrier at 85% of spot (at t=0), rebate=0
    B = 0.85 * S0
    r = float(puller.risk_free_rate)

    # 5) Price via CN
    M, N = grid
    do_price, (Sgrid, Vgrid) = price_do_put_cn(
        S0=S0, K=K, B=B, r=r, sigma=sigma, T=T,
        M=M, N=N, rebate=0.0, use_rannacher=rannacher, return_grid=True
    )

    # 6) Sanity: vanilla put + in/out parity
    vanilla = bs_put_price(S0, K, r, sigma, T)
    di_parity = vanilla - do_price

    # 7) Optional MC (continuous KO via Brownian bridge)
    mc_price, mc_ci = (float('nan'), float('nan'))
    if do_mc:
        mc_price, mc_se = mc_do_put_bb(S0, K, B, r, sigma, T, n_paths=120_000, n_steps=252, seed=7)
        mc_ci = 1.96*mc_se

    # 8) Save visuals
    mask = Sgrid >= B
    plt.figure()
    plt.plot(Sgrid[mask], Vgrid[mask], label=f"CN t=0 (M=N={M})")
    plt.axvline(B, linestyle="--", label=f"Barrier B={B:.2f}")
    plt.title(f"PLTR Down-and-Out Put at t=0 (exp={exp}, K≈ATM)")
    plt.xlabel("Spot S"); plt.ylabel("Option value")
    plt.legend(); plt.tight_layout()
    out_png = f"pltr_do_put_profile_{exp}.png"
    plt.savefig(out_png, dpi=160)

    # Simple convergence table (3 ladders)
    rows = []
    for (m,n) in [(120,120),(240,240),(480,480)]:
        p,_ = price_do_put_cn(S0, K, B, r, sigma, T, M=m, N=n, return_grid=False)
        rows.append({"M": m, "N": n, "Price": p})
    df = pd.DataFrame(rows)
    out_csv = f"pltr_do_put_convergence_{exp}.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== PLTR Down-and-Out Put (continuous barrier) ===")
    print(f"Expiration chosen: {exp}  (T ≈ {T:.4f}y, target 1y)")
    print(f"S0={S0:.4f}  K(ATM)={K:.4f}  B=0.85*S0={B:.4f}")
    print(f"r={r:.4%}  sigma(ATM)={sigma:.4%}  side_used={side}")
    print(df.to_string(index=False))
    print(f"Vanilla BS put:      {vanilla:.6f}")
    print(f"DO put (CN):         {do_price:.6f}")
    print(f"DI put (by parity):  {di_parity:.6f}")
    if do_mc:
        print(f"DO put (MC bridge):  {mc_price:.6f}  ± {mc_ci:.6f}  (95% CI)")
    print(f"Wrote: {out_csv}, {out_png}\n")

        # 9) Convergence plots
    steps_plot_png = f"pltr_cn_steps_convergence_{exp}.png"
    paths_plot_png = f"pltr_mc_paths_convergence_{exp}.png"

    df_steps = plot_cn_time_convergence(
        S0, K, B, r, sigma, T,
        steps_list=[50, 100, 250, 500, 1000],
        M_fixed=1200, ref_grid=(1600, 1600),
        rebate=0.0, use_rannacher=rannacher,
        title_suffix=f"(exp={exp}, K≈ATM, B=0.85*S0)",
        out_path=steps_plot_png
    )
    df_steps.to_csv(f"pltr_cn_steps_convergence_{exp}.csv", index=False)

    df_paths = plot_mc_path_convergence(
        S0, K, B, r, sigma, T,
        path_list=[100, 1_000, 10_000, 100_000, 1_000_000],
        n_steps=252, seed=7,
        cn_ref_grid=(1600, 1600),
        title_suffix=f"(exp={exp}, K≈ATM, B=0.85*S0)",
        out_path=paths_plot_png
    )
    df_paths.to_csv(f"pltr_mc_paths_convergence_{exp}.csv", index=False)

    print(f"Convergence plots written: {steps_plot_png}, {paths_plot_png}")

    space_plot_png = f"pltr_cn_space_convergence_{exp}.png"
    df_space = plot_cn_space_convergence(
        S0, K, B, r, sigma, T,
        M_list=[50, 100, 250, 500, 1000],
        N_fixed=1200, ref_grid=(1600, 1600),
        rebate=0.0, use_rannacher=rannacher,
        title_suffix=f"(exp={exp}, K≈ATM, B=0.85*S0)",
        out_path=space_plot_png
    )
    df_space.to_csv(f"pltr_cn_space_convergence_{exp}.csv", index=False)
    print(f"Space convergence plot written: {space_plot_png}")

    return {
        "S0": S0, "K": K, "B": B, "T": T, "r": r, "sigma": sigma,
        "do_put_cn": do_price,
        "vanilla_put": vanilla,
        "di_put_by_parity": di_parity,
        "do_put_mc": mc_price, "do_put_mc_95ci": mc_ci
    }

# ===== Convergence plots =====================================================

def plot_cn_time_convergence(
    S0: float, K: float, B: float, r: float, sigma: float, T: float,
    steps_list: List[int] = (50, 100, 250, 500, 1000),
    M_fixed: int = 1200,
    ref_grid: Tuple[int,int] = (1600, 1600),
    rebate: float = 0.0,
    use_rannacher: bool = True,
    title_suffix: str = "",
    out_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Vary only N (time steps) while fixing spatial resolution M to show CN convergence in time.
    Also compute a fine-grid CN reference on (M_ref, N_ref).
    Saves a PNG and returns a DataFrame of (N, price).
    """
    M_ref, N_ref = ref_grid
    ref_price, _ = price_do_put_cn(S0, K, B, r, sigma, T,
                                   M=M_ref, N=N_ref, rebate=rebate,
                                   use_rannacher=use_rannacher, return_grid=False)

    rows = []
    for N in steps_list:
        p, _ = price_do_put_cn(S0, K, B, r, sigma, T,
                               M=M_fixed, N=N, rebate=rebate,
                               use_rannacher=use_rannacher, return_grid=False)
        rows.append({"N_steps": N, "M_fixed": M_fixed, "Price": p, "AbsErr_vs_ref": abs(p - ref_price)})
    df = pd.DataFrame(rows)

    # Plot
    plt.figure()
    plt.plot(df["N_steps"], df["Price"], marker="o", label=f"CN (M={M_fixed}, vary N)")
    plt.axhline(ref_price, linestyle="--", label=f"CN reference (M=N={M_ref}) = {ref_price:.4f}")
    plt.title(f"CN Time-Step Convergence {title_suffix}".strip())
    plt.xlabel("Time steps N")
    plt.ylabel("Option price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if out_path is not None:
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    return df


def plot_mc_path_convergence(
    S0: float, K: float, B: float, r: float, sigma: float, T: float,
    path_list: List[int] = (100, 1_000, 10_000, 100_000, 1_000_000),
    n_steps: int = 252,
    seed: int = 777,
    cn_ref_grid: Tuple[int,int] = (1600, 1600),
    title_suffix: str = "",
    out_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Vary only the number of Monte Carlo paths, keep discrete time grid fixed.
    Uses Brownian-bridge KO inside mc_do_put_bb (your function) and overlays a CN reference.
    Saves a PNG and returns a DataFrame of (paths, price, SE).
    """
    # Fine CN reference for visual target
    M_ref, N_ref = cn_ref_grid
    cn_ref, _ = price_do_put_cn(S0, K, B, r, sigma, T,
                                M=M_ref, N=N_ref, rebate=0.0,
                                use_rannacher=True, return_grid=False)

    rows = []
    for i, P in enumerate(path_list):
        # Ensure even for antithetic pairing in mc_do_put_bb
        n_paths = int(2 * math.ceil(P / 2))
        price, se = mc_do_put_bb(S0, K, B, r, sigma, T,
                                 n_paths=n_paths, n_steps=n_steps, seed=seed + i)
        rows.append({"Paths": n_paths, "Steps_per_year": n_steps, "Price": price, "SE": se,
                     "AbsErr_vs_CNref": abs(price - cn_ref)})
    df = pd.DataFrame(rows)

    # Plot with 95% CI error bars; log-scale x for clarity
    ci95 = 1.96 * df["SE"].to_numpy()
    plt.figure()
    plt.errorbar(df["Paths"], df["Price"], yerr=ci95, fmt="-o", capsize=3, label="MC (±95% CI)")
    plt.axhline(cn_ref, linestyle="--", label=f"CN reference (M=N={M_ref}) = {cn_ref:.4f}")
    plt.xscale("log")
    plt.title(f"MC Path Convergence (Brownian-Bridge) {title_suffix}".strip())
    plt.xlabel("Number of simulated paths (log scale)")
    plt.ylabel("Option price")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    if out_path is not None:
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    return df

def plot_cn_space_convergence(
    S0: float, K: float, B: float, r: float, sigma: float, T: float,
    M_list: List[int] = (50, 100, 250, 500, 1000),
    N_fixed: int = 1200,
    ref_grid: Tuple[int,int] = (1600, 1600),
    rebate: float = 0.0,
    use_rannacher: bool = True,
    title_suffix: str = "",
    out_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Vary only M (space nodes) while fixing time discretisation N to show CN convergence in space.
    Also compute a fine-grid CN reference on (M_ref, N_ref).
    Saves a PNG and returns a DataFrame of (M, price).
    """
    M_ref, N_ref = ref_grid
    ref_price, _ = price_do_put_cn(S0, K, B, r, sigma, T,
                                   M=M_ref, N=N_ref, rebate=rebate,
                                   use_rannacher=use_rannacher, return_grid=False)

    rows = []
    for M in M_list:
        p, _ = price_do_put_cn(S0, K, B, r, sigma, T,
                               M=M, N=N_fixed, rebate=rebate,
                               use_rannacher=use_rannacher, return_grid=False)
        rows.append({"M_nodes": M, "N_fixed": N_fixed, "Price": p, "AbsErr_vs_ref": abs(p - ref_price)})
    df = pd.DataFrame(rows)

    # Plot
    plt.figure()
    plt.plot(df["M_nodes"], df["Price"], marker="o", label=f"CN (N={N_fixed}, vary M)")
    plt.axhline(ref_price, linestyle="--", label=f"CN reference (M=N={M_ref}) = {ref_price:.4f}")
    plt.title(f"CN Space Convergence {title_suffix}".strip())
    plt.xlabel("Number of space nodes M")
    plt.ylabel("Option price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if out_path is not None:
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    return df



# ---- Example runner ---------------------------------------------------------
if __name__ == "__main__":
    # If your class lives in this file already, just construct it; otherwise import it:
    # from option_data_puller import OptionDataPuller
    puller = OptionDataPuller("PLTR", risk_free_rate=0.05)
    results = price_pltr_do_put_from_puller(puller, target_days=365, rannacher=True, grid=(480,480), do_mc=True)
    print(results)
