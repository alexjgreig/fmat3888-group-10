# =============================================================================
# PLTR Down-and-Out Put — Crank–Nicolson (CN) PDE
# - ATM/IV selection consistent with MC scripts
# - Convergence vs time steps (N) and vs space nodes (M)
# - Plot formatting & badges aligned with the MC style
# =============================================================================
import os, sys, warnings, math
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
warnings.filterwarnings("ignore")

# Make local package importable (matches project layout)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from option_pricing.data.yahoo_data import OptionDataPuller


# =============================================================================
# [A] Numerics: small helpers (Thomas solver + Black–Scholes sanity)
# =============================================================================
def thomas(l: np.ndarray, d: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Classic tridiagonal solver; tiny, fast, and all we need for CN."""
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

def bs_put_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Vanilla BS put (q=0) for a quick parity/sanity check."""
    if T <= 0: 
        return max(K - S0, 0.0)
    d1 = (math.log(S0/K) + (r + 0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    N = lambda x: 0.5*(1.0 + math.erf(x/math.sqrt(2)))
    return K*math.exp(-r*T)*N(-d2) - S0*N(-d1)


# =============================================================================
# [B] Crank–Nicolson down-and-out put (log-space, continuous KO)
# =============================================================================
def price_do_put_cn(
    S0: float, K: float, B: float, r: float, sigma: float, T: float,
    *, Smax_factor: float = 6.0, M: int = 480, N: int = 480,
    rebate: float = 0.0, use_rannacher: bool = True, return_grid: bool = False
) -> Tuple[float, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """CN scheme with optional Rannacher start; barrier enforced each time level."""
    if not (B < S0):
        raise ValueError("Down-and-out requires B < S0.")
    Smax = Smax_factor * max(S0, K)

    # Grid in log(S)
    x_min, x_max = math.log(B), math.log(Smax)
    dx = (x_max - x_min) / M
    dt = T / N
    x = x_min + dx*np.arange(M+1)
    S = np.exp(x)

    # Coeffs for ∂V/∂t = a V_xx + b V_x + c V
    a = 0.5*sigma*sigma
    b = r - 0.5*sigma*sigma
    c = -r

    # Payoff at maturity (enforce knockout)
    V = np.maximum(K - S, 0.0)
    V[S <= B] = rebate

    # Interior points
    m = M - 1
    if m <= 0: 
        raise ValueError("Increase M (need at least 2 interior nodes).")

    A_lower = (a/dx**2) - (b/(2*dx))
    A_main  = (-2*a/dx**2) + c
    A_upper = (a/dx**2) + (b/(2*dx))

    # LHS/RHS (CN) and BE matrices (for Rannacher)
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
        V_left_n = V_left_np1 = rebate  # barrier boundary (knocked-out)
        V_right_n = V_right_np1 = 0.0   # far-field boundary for a put
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

    # Time-march (with/without Rannacher)
    if use_rannacher:
        V = step_once(V, "BE"); V = step_once(V, "BE")
        for _ in range(max(N-2, 0)): V = step_once(V, "CN")
    else:
        for _ in range(N): V = step_once(V, "CN")

    # Interpolate back to S0
    x0 = math.log(S0)
    i_float = (x0 - x_min)/dx
    i = int(np.clip(i_float, 1, M-1)); w = i_float - i
    price = (1-w)*V[i] + w*V[i+1]
    return (float(price), (S, V)) if return_grid else (float(price), (None, None))


# =============================================================================
# [C] Market prep — same ATM/IV policy as MC scripts (side nearest to S0)
# =============================================================================
def _pick_expiry_close_to_1y(puller: OptionDataPuller):
    exps = puller.get_option_expirations()
    if not exps: raise RuntimeError("No option expirations available.")
    ts = pd.to_datetime(exps, utc=True)
    today = pd.Timestamp.now(tz="UTC").normalize()
    days = (ts - today).days.to_numpy()
    idx = np.abs(days - 365).argmin()
    return exps[idx], int(days[idx])

def _pick_iv_from_side(df: pd.DataFrame, S0: float, puller: OptionDataPuller,
                       option_type: str, default_T_years: float):
    """ATM row on a single side (calls/puts). IV pref: calcIV → impliedVol → yahooIV → invert from mid."""
    if df is None or df.empty:
        return float("nan"), None
    dff = df.copy()

    # Ensure 'mid' exists (used both for filtering and as inversion fallback)
    if "mid" not in dff.columns:
        dff["mid"] = dff.apply(
            lambda row: puller.calculate_mid_price(row.get("bid", np.nan),
                                                   row.get("ask", np.nan),
                                                   row.get("lastPrice", np.nan)),
            axis=1
        )
    dff = dff[dff["mid"].notna()]
    if dff.empty: 
        return float("nan"), None

    dff["atm_gap"] = (dff["strike"] - S0).abs()
    dff.sort_values(["atm_gap", "volume", "openInterest"],
                    ascending=[True, False, False], inplace=True)
    row = dff.iloc[0]
    K = float(row["strike"])

    for col in ("calculatedIV", "impliedVolatility", "yahooIV"):
        iv = row.get(col, np.nan)
        if pd.notna(iv) and 0.005 < float(iv) < 5.0:
            return K, float(iv)

    # Last resort: invert from mid
    T_row = float(row.get("timeToExpiry", default_T_years))
    iv = puller.calculate_implied_volatility(
        option_price=float(row["mid"]), spot=float(S0), strike=K,
        time_to_expiry=T_row, option_type=option_type, dividend_yield=0.0
    )
    return K, (None if iv is None else float(iv))

def prepare_pltr_market_inputs(puller: OptionDataPuller, *, barrier_pct=0.85,
                               prefer_side: str = "auto") -> Dict[str, float]:
    """
    Returns {S0, K, B, r, sigma, T, expiry_str, days_to_exp, side_used}.
    prefer_side in {"auto","call","put","average"}.
    """
    S0 = float(puller.get_current_price())
    expiry_str, days_to_exp = _pick_expiry_close_to_1y(puller)
    chain = puller.get_option_chain(expiry_str)
    calls = chain.get("calls", pd.DataFrame()); puts = chain.get("puts", pd.DataFrame())

    T = max(days_to_exp, 1) / 365.25
    Kc, ivc = _pick_iv_from_side(calls, S0, puller, "call", T)
    Kp, ivp = _pick_iv_from_side(puts,  S0, puller, "put",  T)
    have_call = (not np.isnan(Kc)) and (ivc is not None)
    have_put  = (not np.isnan(Kp)) and (ivp is not None)

    def choose_auto():
        if have_call and have_put:
            if abs(Kc - Kp) <= 0.5:
                return 0.5*(Kc+Kp), 0.5*(ivc+ivp), "average"
            return (Kc, ivc, "call") if abs(Kc - S0) <= abs(Kp - S0) else (Kp, ivp, "put")
        if have_call: return Kc, ivc, "call"
        if have_put:  return Kp, ivp, "put"
        raise RuntimeError("No valid ATM IV found on either side.")

    if prefer_side == "call" and have_call:   K, sigma, side_used = Kc, ivc, "call"
    elif prefer_side == "put" and have_put:   K, sigma, side_used = Kp, ivp, "put"
    elif prefer_side == "average" and have_call and have_put and abs(Kc - Kp) <= 0.5:
        K, sigma, side_used = 0.5*(Kc+Kp), 0.5*(ivc+ivp), "average"
    else:
        K, sigma, side_used = choose_auto()

    if sigma > 2.0: sigma *= 0.01  # guard against percent-looking IVs

    return dict(
        S0=S0, K=float(K), B=float(barrier_pct*S0),
        r=float(puller.risk_free_rate), sigma=float(sigma), T=T,
        expiry_str=expiry_str, days_to_exp=days_to_exp, side_used=side_used
    )


# =============================================================================
# [D] Convergence sweeps: vs time steps N, vs space nodes M
# =============================================================================
def cn_sweep_vs_time(S0, K, B, r, sigma, T,
                     steps_list=(50, 100, 250, 500, 1000),
                     M_fixed=1200, ref_grid=(1600, 1600),
                     rebate=0.0, use_rannacher=True):
    """Hold space resolution fixed; vary N (time steps)."""
    M_ref, N_ref = ref_grid
    ref, _ = price_do_put_cn(S0, K, B, r, sigma, T,
                             M=M_ref, N=N_ref, rebate=rebate,
                             use_rannacher=use_rannacher, return_grid=False)
    rows = []
    for N in steps_list:
        p, _ = price_do_put_cn(S0, K, B, r, sigma, T,
                               M=M_fixed, N=N, rebate=rebate,
                               use_rannacher=use_rannacher, return_grid=False)
        rows.append({"N_steps": int(N), "M_fixed": int(M_fixed), "Price": float(p),
                     "AbsErr_vs_ref": abs(p - ref)})
    df = pd.DataFrame(rows)
    return df, ref, (M_ref, N_ref)

def cn_sweep_vs_space(S0, K, B, r, sigma, T,
                      M_list=(50, 100, 250, 500, 1000),
                      N_fixed=1200, ref_grid=(1600, 1600),
                      rebate=0.0, use_rannacher=True):
    """Hold time resolution fixed; vary M (space nodes)."""
    M_ref, N_ref = ref_grid
    ref, _ = price_do_put_cn(S0, K, B, r, sigma, T,
                             M=M_ref, N=N_ref, rebate=rebate,
                             use_rannacher=use_rannacher, return_grid=False)
    rows = []
    for M in M_list:
        p, _ = price_do_put_cn(S0, K, B, r, sigma, T,
                               M=M, N=N_fixed, rebate=rebate,
                               use_rannacher=use_rannacher, return_grid=False)
        rows.append({"M_nodes": int(M), "N_fixed": int(N_fixed), "Price": float(p),
                     "AbsErr_vs_ref": abs(p - ref)})
    df = pd.DataFrame(rows)
    return df, ref, (M_ref, N_ref)


# ==============================================
# [E] Plot utilities — badges + CN plots 
# ==============================================
def _market_badge_text(mkt):
    S0, K, B = mkt["S0"], mkt["K"], mkt["B"]
    r, sigma, T = mkt["r"], mkt["sigma"], mkt["T"]
    return f"S\u2080={S0:.2f}   K={K:.2f}   B={B:.2f}   r={r*100:.2f}%   σ={sigma*100:.2f}%   T={T:.2f}y"

def _add_badge(ax, text, loc="upper left", alpha=0.18, y_shift=0.0):
    at = AnchoredText(text, loc=loc, prop=dict(size=9, color="black"),
                      frameon=True, borderpad=0.6)
    at.patch.set_boxstyle("round,pad=0.4,rounding_size=0.8")
    at.patch.set_alpha(alpha)
    ax.add_artist(at)


def plot_cn_time_convergence(df_steps, ref_price, ref_grid, mkt, *, M_fixed):
    M_ref, N_ref = ref_grid
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(df_steps["N_steps"], df_steps["Price"], marker="o",
            label=f"CN (M={M_fixed}, vary N)")
    ax.axhline(ref_price, color = "C1", linestyle="--", label=f"CN reference (M=N={M_ref}) = {ref_price:.4f}")
    ax.set_xlabel("Time steps per year (N)")
    ax.set_ylabel("Option Price\n(PLTR; down-and-out; 85% barrier; ATM strike; 1Y expiry)")
    ax.set_title("Option Price Step Convergence (Crank–Nicolson PDE)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc = "center right")
    _add_badge(ax, _market_badge_text(mkt), loc="upper right")
    fig.tight_layout()

def plot_cn_space_convergence(df_space, ref_price, ref_grid, mkt, *, N_fixed):
    M_ref, N_ref = ref_grid
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(df_space["M_nodes"], df_space["Price"], marker="o",
            label=f"CN (N={N_fixed}, vary M)")
    ax.axhline(ref_price, color = "C1", linestyle="--", label=f"CN reference (M=N={M_ref}) = {ref_price:.4f}")
    ax.set_xlabel("Number of space nodes (M)")
    ax.set_ylabel("Option Price\n(PLTR; down-and-out; 85% barrier; ATM strike; 1Y expiry)")
    ax.set_title("Option Price Space Convergence (Crank–Nicolson PDE)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc = "center right")
    _add_badge(ax, _market_badge_text(mkt), loc="lower right", y_shift = -1)
    fig.tight_layout()


# =============================================================================
# [F] Main
# =============================================================================
if __name__ == "__main__":
    # Pull market and normalize inputs exactly like MC scripts
    puller = OptionDataPuller("PLTR", risk_free_rate=0.05)
    market = prepare_pltr_market_inputs(puller, barrier_pct=0.85, prefer_side="auto")

    S0, K, B = market["S0"], market["K"], market["B"]
    r, sigma, T = market["r"], market["sigma"], market["T"]

    print("\n=== PLTR Down-and-Out Put (CN) ===")
    print(f"Expiry chosen: {market['expiry_str']}  (T ≈ {T:.4f}y; ~{market['days_to_exp']} days)")
    print(f"S0={S0:.4f}  K(ATM)={K:.4f}  B=0.85*S0={B:.4f}")
    print(f"r={r:.4%}  sigma(ATM)={sigma:.4%}  side_used={market['side_used']}\n")

    # Reference price on a fine grid (handy for parity/logging)
    cn_ref, _ = price_do_put_cn(S0, K, B, r, sigma, T,
                                M=1600, N=1600, rebate=0.0, use_rannacher=True, return_grid=False)
    vanilla = bs_put_price(S0, K, r, sigma, T)
    print(f"CN reference (M=N=1600): {cn_ref:.6f}")
    print(f"Vanilla BS put:           {vanilla:.6f}")
    print(f"DI (by parity):           {vanilla - cn_ref:.6f}\n")

    # Convergence settings
    STEPS_LIST = [50, 100, 250, 500, 1000]
    M_FIXED    = 1200
    SPACE_LIST = [50, 100, 250, 500, 1000]
    N_FIXED    = 1200
    REF_GRID   = (1600, 1600)

    # Sweeps
    df_steps, ref_steps_price, _ = cn_sweep_vs_time(S0, K, B, r, sigma, T,
                                                    steps_list=STEPS_LIST,
                                                    M_fixed=M_FIXED, ref_grid=REF_GRID,
                                                    rebate=0.0, use_rannacher=True)

    df_space, ref_space_price, _ = cn_sweep_vs_space(S0, K, B, r, sigma, T,
                                                     M_list=SPACE_LIST,
                                                     N_fixed=N_FIXED, ref_grid=REF_GRID,
                                                     rebate=0.0, use_rannacher=True)

    # 5) Console tables
    print("Time-step convergence (CN; M fixed):")
    print(df_steps.to_string(index=False))
    print("\nSpace convergence (CN; N fixed):")
    print(df_space.to_string(index=False))

    # 6) Plots
    plot_cn_time_convergence(df_steps, ref_steps_price, REF_GRID, market, M_fixed=M_FIXED)
    plot_cn_space_convergence(df_space, ref_space_price, REF_GRID, market, N_fixed=N_FIXED)
    plt.show()
