# =============================================================================
# PLTR European Call — Euler/Milstein MC vs Black–Scholes
# - Pulls live market/chain from OptionDataPuller
# - Chooses ATM strike & IV consistently with the barrier scripts
# - Runs convergence vs. paths and vs. steps/year
# - Plots results with market badges and BS reference line
# =============================================================================
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt, exp, erf
from matplotlib.offsetbox import AnchoredText

# Make local package importable (matches project layout)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from option_pricing.data.yahoo_data import OptionDataPuller


# =============================================================================
# [A] Analytic reference (Black–Scholes, q=0)
# =============================================================================
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_call(S0: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    """Closed-form BS price for a European call (default: no dividend, q=0)."""
    if T <= 0 or sigma <= 0:
        return max(S0 * exp(-q * T) - K * exp(-r * T), 0.0)
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * exp(-q * T) * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)


# =============================================================================
# [B] Monte Carlo engines (Euler & Milstein)
# =============================================================================
def _euler_terminal(S0, r, sigma, T, steps, Z):
    """Euler–Maruyama terminal prices using provided normals Z (n_paths x steps)."""
    dt = T / steps
    S = np.full(Z.shape[0], S0, dtype=float)
    for n in range(steps):
        S += r * S * dt + sigma * S * np.sqrt(dt) * Z[:, n]
    return S

def _milstein_terminal(S0, r, sigma, T, steps, Z):
    """Milstein terminal prices using provided normals Z (n_paths x steps)."""
    dt = T / steps
    sqrt_dt = np.sqrt(dt)
    S = np.full(Z.shape[0], S0, dtype=float)
    for n in range(steps):
        dW = sqrt_dt * Z[:, n]
        S += r * S * dt + sigma * S * dW + 0.5 * (sigma * sigma) * S * (dW * dW - dt)
    return S

def mc_price_call_both_methods(S0, K, r, sigma, T, *, steps_per_year, n_paths,
                               seed=2025, chunk_size=200_000):
    """
    Chunked MC driver returning (price, se) for Euler & Milstein using common RNG.
    We chunk to manage memory when n_paths is large.
    """
    steps = max(1, int(np.ceil(steps_per_year * T)))
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    def _acc(): return {"sum": 0.0, "sumsq": 0.0, "n": 0}
    acc_eu, acc_mi = _acc(), _acc()

    def _upd(acc, vals):
        acc["sum"]   += vals.sum()
        acc["sumsq"] += (vals * vals).sum()
        acc["n"]     += vals.size

    done = 0
    while done < n_paths:
        m = min(chunk_size, n_paths - done)
        Z = rng.standard_normal((m, steps))            # common random numbers
        ST_eu = _euler_terminal  (S0, r, sigma, T, steps, Z)
        ST_mi = _milstein_terminal(S0, r, sigma, T, steps, Z)

        vals_eu = disc * np.maximum(ST_eu - K, 0.0)
        vals_mi = disc * np.maximum(ST_mi - K, 0.0)
        _upd(acc_eu, vals_eu); _upd(acc_mi, vals_mi)
        done += m

    def _final(acc):
        n = acc["n"]
        mean = acc["sum"] / n
        var  = max(acc["sumsq"] / n - mean * mean, 0.0)
        se   = np.sqrt(var / n)
        return mean, se

    return _final(acc_eu), _final(acc_mi)


# =============================================================================
# [C] Market plumbing — pick 1Y-ish expiry, ATM strike & IV (CN-consistent)
# =============================================================================
def _pick_expiry_close_to_1y(puller: OptionDataPuller):
    exps = puller.get_option_expirations()
    if not exps:
        raise RuntimeError("No option expirations available.")
    ts = pd.to_datetime(exps, utc=True)
    today = pd.Timestamp.now(tz="UTC").normalize()
    days = (ts - today).days.to_numpy()
    idx = np.abs(days - 365).argmin()
    return exps[idx], int(days[idx])

def _pick_iv_from_side(df: pd.DataFrame, S0: float, puller: OptionDataPuller,
                       option_type: str, default_T_years: float):
    """
    ATM selection on one side (calls OR puts).
    IV preference: calculatedIV -> impliedVolatility -> yahooIV -> invert from mid.
    Returns (K, iv or None).
    """
    if df is None or df.empty:
        return float("nan"), None

    dff = df.copy()
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

    # last resort: IV from mid
    T_row = float(row.get("timeToExpiry", default_T_years))
    iv = puller.calculate_implied_volatility(
        option_price=float(row["mid"]), spot=float(S0), strike=K,
        time_to_expiry=T_row, option_type=option_type, dividend_yield=0.0
    )
    return K, (None if iv is None else float(iv))

def prepare_pltr_market_inputs(prefer_side: str = "auto"):
    """
    Return a dict with S0, K, r, sigma, T, expiry_str, days_to_exp, side_used.
    prefer_side ∈ {"auto","call","put","average"} to control which IV to use.
    """
    puller = OptionDataPuller("PLTR")
    S0 = float(puller.get_current_price())
    expiry_str, days_to_exp = _pick_expiry_close_to_1y(puller)
    chain = puller.get_option_chain(expiry_str)

    T_default = max(days_to_exp, 1) / 365.25
    Kc, ivc = _pick_iv_from_side(chain.get("calls", pd.DataFrame()), S0, puller, "call", T_default)
    Kp, ivp = _pick_iv_from_side(chain.get("puts",  pd.DataFrame()), S0, puller, "put",  T_default)
    have_call = (not np.isnan(Kc)) and (ivc is not None)
    have_put  = (not np.isnan(Kp)) and (ivp is not None)

    # choose IV according to policy
    if prefer_side == "call" and have_call:
        K, sigma, side_used = Kc, ivc, "call"
    elif prefer_side == "put" and have_put:
        K, sigma, side_used = Kp, ivp, "put"
    elif prefer_side == "average" and have_call and have_put:
        # average if both are near-ATM (strikes close), else pick nearer strike
        if abs(Kc - Kp) <= 0.5:
            K = 0.5 * (Kc + Kp)
            sigma = 0.5 * (ivc + ivp)
            side_used = "average"
        else:
            if abs(Kc - S0) <= abs(Kp - S0): K, sigma, side_used = Kc, ivc, "call"
            else:                             K, sigma, side_used = Kp, ivp, "put"
    else:
        # auto (default): choose side whose strike is closest to S0; average if very close
        if have_call and have_put:
            if abs(Kc - Kp) <= 0.5:
                K = 0.5 * (Kc + Kp)
                sigma = 0.5 * (ivc + ivp)
                side_used = "average"
            else:
                if abs(Kc - S0) <= abs(Kp - S0): K, sigma, side_used = Kc, ivc, "call"
                else:                             K, sigma, side_used = Kp, ivp, "put"
        elif have_call:
            K, sigma, side_used = Kc, ivc, "call"
        elif have_put:
            K, sigma, side_used = Kp, ivp, "put"
        else:
            raise RuntimeError("No valid ATM IV found on either side.")

    if sigma > 2.0:    # guard against percent IV like 57 → 0.57
        sigma *= 0.01

    r = float(puller.risk_free_rate)
    T = T_default   # use *actual* days-to-expiry to be consistent with CN scripts
    return dict(S0=S0, K=K, r=r, sigma=sigma, T=T,
                expiry_str=expiry_str, days_to_exp=days_to_exp,
                side_used=side_used)


# =============================================================================
# [D] Convergence experiments
# =============================================================================
def sweep_vs_paths(mkt, paths=(100, 1_000, 10_000, 100_000, 1_000_000),
                   steps_per_year=250, seed=4242, chunk_size=200_000):
    out = {"paths": [], "euler_price": [], "euler_se": [],
           "milstein_price": [], "milstein_se": []}
    for n in paths:
        (pe, see), (pm, sem) = mc_price_call_both_methods(
            mkt["S0"], mkt["K"], mkt["r"], mkt["sigma"], mkt["T"],
            steps_per_year=steps_per_year, n_paths=int(n),
            seed=seed + int(n), chunk_size=chunk_size
        )
        out["paths"].append(int(n))
        out["euler_price"].append(pe);    out["euler_se"].append(see)
        out["milstein_price"].append(pm); out["milstein_se"].append(sem)
    return out

def sweep_vs_steps(mkt, steps=(50, 100, 250, 500, 1000),
                   n_paths=100_000, seed=2025, chunk_size=200_000):
    out = {"steps": [], "euler_price": [], "euler_se": [],
           "milstein_price": [], "milstein_se": []}
    for s in steps:
        (pe, see), (pm, sem) = mc_price_call_both_methods(
            mkt["S0"], mkt["K"], mkt["r"], mkt["sigma"], mkt["T"],
            steps_per_year=int(s), n_paths=int(n_paths),
            seed=seed + int(s), chunk_size=chunk_size
        )
        out["steps"].append(int(s))
        out["euler_price"].append(pe);    out["euler_se"].append(see)
        out["milstein_price"].append(pm); out["milstein_se"].append(sem)
    return out


# =============================================================================
# [E] Plot helpers (badges + convergence charts)
# =============================================================================
def _market_badge_text(S0, K, r, sigma, T):
    # plain text (no mathtext) with unicode S₀, σ for consistency
    return f"S\u2080={S0:.2f}   K={K:.2f}   r={r*100:.2f}%   σ={sigma*100:.2f}%   T={T:.2f}y"

def _add_badge(ax, text, loc="upper left", alpha=0.18):
    at = AnchoredText(text, loc=loc, prop=dict(size=9, color="black"),
                      frameon=True, borderpad=0.6)
    at.patch.set_boxstyle("round,pad=0.3,rounding_size=0.8")
    at.patch.set_alpha(alpha)
    ax.add_artist(at)

def plot_vs_paths(res, fixed_steps_per_year, bs_price, mkt):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array(res["paths"])
    e, se = np.array(res["euler_price"]),    1.96 * np.array(res["euler_se"])
    m, sm = np.array(res["milstein_price"]), 1.96 * np.array(res["milstein_se"])

    ax.errorbar(x, e, yerr=se, fmt="o-", label="Euler (±95% CI)")
    ax.errorbar(x, m, yerr=sm, fmt="s-", label="Milstein (±95% CI)")
    ax.axhline(bs_price, linestyle="--", linewidth=1.5,
               label=f"Black–Scholes = {bs_price:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("Option Price\n(PLTR Vanilla European Call 1Y Expiry)")
    ax.set_title(f"Option Price Path Convergence (MC)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    _add_badge(ax, _market_badge_text(mkt["S0"], mkt["K"], mkt["r"], mkt["sigma"], mkt["T"]),
               loc="upper left")
    _add_badge(ax, f"Fixed steps/year = {fixed_steps_per_year}", loc="lower left")
    fig.tight_layout()

def plot_vs_steps(res, fixed_n_paths, bs_price, mkt):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array(res["steps"])
    e, se = np.array(res["euler_price"]),    1.96 * np.array(res["euler_se"])
    m, sm = np.array(res["milstein_price"]), 1.96 * np.array(res["milstein_se"])

    ax.errorbar(x, e, yerr=se, fmt="o-", label="Euler (±95% CI)")
    ax.errorbar(x, m, yerr=sm, fmt="s-", label="Milstein (±95% CI)")
    ax.axhline(bs_price, linestyle="--", linewidth=1.5,
               label=f"Black–Scholes = {bs_price:.4f}")
    ax.set_xlabel("Time steps per year")
    ax.set_ylabel("Option Price\n(PLTR Vanilla European Call 1Y Expiry)")
    ax.set_title(f"Option Price Path Convergence (MC)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _add_badge(ax, _market_badge_text(mkt["S0"], mkt["K"], mkt["r"], mkt["sigma"], mkt["T"]),
               loc="upper left")
    _add_badge(ax, f"Fixed paths = {fixed_n_paths:,}", loc="lower left")
    fig.tight_layout()


# =============================================================================
# [F] Main
# =============================================================================
if __name__ == "__main__":
    # --- Pull market & align IV selection with barrier scripts
    market = prepare_pltr_market_inputs(prefer_side="auto")  # "auto" | "call" | "put" | "average"
    S0, K, r, sigma, T = market["S0"], market["K"], market["r"], market["sigma"], market["T"]

    print("PLTR vanilla call — market inputs:")
    print(f"  S0={S0:.4f}  K={K:.4f}  σ={sigma:.4%}  r={r:.4%}  T={T:.4f}")
    print(f"  Chain expiry chosen: {market['expiry_str']} (~{market['days_to_exp']} days) "
          f"| IV side used: {market['side_used']}\n")

    # --- Analytic reference
    bs_px = bs_call(S0, K, r, sigma, T)
    print(f"Black–Scholes reference: {bs_px:.6f}\n")

    # --- Convergence settings (consistent across scripts)
    PATHS_LIST            = [100, 1_000, 10_000, 100_000, 1_000_000]
    STEPS_LIST            = [50, 100, 250, 500, 1000]
    FIXED_STEPS_FOR_PATHS = 250
    FIXED_PATHS_FOR_STEPS = 100_000
    CHUNK_SIZE            = 200_000

    # --- Run sweeps
    res_paths = sweep_vs_paths(market, PATHS_LIST, steps_per_year=FIXED_STEPS_FOR_PATHS,
                               seed=4242, chunk_size=CHUNK_SIZE)
    res_steps = sweep_vs_steps(market, STEPS_LIST, n_paths=FIXED_PATHS_FOR_STEPS,
                               seed=2025, chunk_size=CHUNK_SIZE)

    # --- Console tables
    print("PATHS convergence (steps/year = {}):".format(FIXED_STEPS_FOR_PATHS))
    print("paths     | Euler_price   SE       | Milstein_price   SE")
    for n, pe, see, pm, sem in zip(res_paths["paths"], res_paths["euler_price"], res_paths["euler_se"],
                                   res_paths["milstein_price"], res_paths["milstein_se"]):
        print(f"{n:9,d} | {pe:12.6f} {see:9.6f} | {pm:13.6f} {sem:9.6f}")

    print("\nSTEPS convergence (paths = {:,}):".format(FIXED_PATHS_FOR_STEPS))
    print("steps/yr | Euler_price   SE       | Milstein_price   SE")
    for s, pe, see, pm, sem in zip(res_steps["steps"], res_steps["euler_price"], res_steps["euler_se"],
                                   res_steps["milstein_price"], res_steps["milstein_se"]):
        print(f"{s:8d} | {pe:12.6f} {see:9.6f} | {pm:13.6f} {sem:9.6f}")

    # --- Plots
    plot_vs_paths(res_paths, FIXED_STEPS_FOR_PATHS, bs_px, market)
    plot_vs_steps(res_steps, FIXED_PATHS_FOR_STEPS, bs_px, market)
    plt.show()







