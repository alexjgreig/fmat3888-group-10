# =============================================================================
# PLTR Down-and-Out Put — Euler/Milstein MC with Brownian-Bridge KO
# - Pulls live market/chain from OptionDataPuller
# - ATM strike & IV selection consistent with CN pipeline
# - Convergence vs. paths and vs. steps/year
# - Plots with market + meta badges
# =============================================================================
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from option_pricing.data.yahoo_data import OptionDataPuller


# =============================================================================
# [A] MC building blocks
# =============================================================================
def mc_discount_stats(payoffs, r, T):
    """Discount and return (price, standard error, 95% CI)."""
    vals = np.exp(-r * T) * payoffs
    price = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(max(vals.size, 1)))
    return price, se, (price - 1.96 * se, price + 1.96 * se)

def _euler_step(S, r, sigma, dt, Z):
    """Euler–Maruyama step in price space."""
    return S + r * S * dt + sigma * S * np.sqrt(dt) * Z

def _milstein_step(S, r, sigma, dt, Z):
    """Milstein step in price space."""
    dW = np.sqrt(dt) * Z
    return S + r * S * dt + sigma * S * dW + 0.5 * (sigma**2) * S * (dW**2 - dt)

def _bridge_knock_prob(S0, S1, B, sigma, dt):
    """
    Brownian-bridge crossing probability for a *down* barrier (log-space).
    Valid when both endpoints are > B:
        p_hit = exp(-2 * ln(S0/B) * ln(S1/B) / (sigma^2 * dt))
    """
    x = np.log(S0) - np.log(B)
    y = np.log(S1) - np.log(B)
    expo = -2.0 * (x * y) / (sigma**2 * dt)
    return np.clip(np.exp(np.clip(expo, -700.0, 50.0)), 0.0, 1.0)


# ======================================================
# [B] Market plumbing — pick 1Y expiry, ATM strike & IV 
# ======================================================
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
    """
    One-side ATM pick (calls OR puts).
    IV preference: calculatedIV → impliedVolatility → yahooIV → invert from mid.
    Returns (K, iv or None).
    """
    if df is None or df.empty:
        return float("nan"), None

    dff = df.copy()
    # Ensure a 'mid' exists for inversion fallback
    if "mid" not in dff.columns:
        dff["mid"] = dff.apply(
            lambda row: puller.calculate_mid_price(row.get("bid", np.nan),
                                                   row.get("ask", np.nan),
                                                   row.get("lastPrice", np.nan)),
            axis=1
        )
    dff = dff[dff["mid"].notna()]
    if dff.empty: return float("nan"), None

    # ATM row on this side (liquidity tie-breakers)
    dff["atm_gap"] = (dff["strike"] - S0).abs()
    dff.sort_values(["atm_gap", "volume", "openInterest"],
                    ascending=[True, False, False], inplace=True)
    row = dff.iloc[0]
    K = float(row["strike"])

    # Prefer our computed IVs over Yahoo’s raw one
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
                               prefer_side: str = "auto"):
    """
    Return a dict with S0, K, B, r, sigma, T, expiry_str, days_to_exp, side_used.
    prefer_side ∈ {"auto","call","put","average"} controls IV selection policy:
      - auto: pick side whose strike is closest to S0; average IVs if strikes ~equal
      - call/put: force that side if available
      - average: average if both sides available & strikes very close, else nearer strike
    """
    S0 = float(puller.get_current_price())
    expiry_str, days_to_exp = _pick_expiry_close_to_1y(puller)
    chain = puller.get_option_chain(expiry_str)
    calls = chain.get("calls", pd.DataFrame()); puts = chain.get("puts", pd.DataFrame())

    T_default = max(days_to_exp, 1) / 365.25
    Kc, ivc = _pick_iv_from_side(calls, S0, puller, "call", T_default)
    Kp, ivp = _pick_iv_from_side(puts,  S0, puller, "put",  T_default)
    have_call = (not np.isnan(Kc)) and (ivc is not None)
    have_put  = (not np.isnan(Kp)) and (ivp is not None)

    # Decide which IV to use
    if prefer_side == "call" and have_call:
        K, sigma, side_used = Kc, ivc, "call"
    elif prefer_side == "put" and have_put:
        K, sigma, side_used = Kp, ivp, "put"
    elif prefer_side == "average" and have_call and have_put:
        if abs(Kc - Kp) <= 0.5:
            K = 0.5 * (Kc + Kp); sigma = 0.5 * (ivc + ivp); side_used = "average"
        else:
            if abs(Kc - S0) <= abs(Kp - S0): K, sigma, side_used = Kc, ivc, "call"
            else:                             K, sigma, side_used = Kp, ivp, "put"
    else:
        # auto (default)
        if have_call and have_put:
            if abs(Kc - Kp) <= 0.5:
                K = 0.5 * (Kc + Kp); sigma = 0.5 * (ivc + ivp); side_used = "average"
            else:
                if abs(Kc - S0) <= abs(Kp - S0): K, sigma, side_used = Kc, ivc, "call"
                else:                             K, sigma, side_used = Kp, ivp, "put"
        elif have_call:
            K, sigma, side_used = Kc, ivc, "call"
        elif have_put:
            K, sigma, side_used = Kp, ivp, "put"
        else:
            raise RuntimeError("No valid ATM IV found on either side.")

    if sigma > 2.0: sigma *= 0.01

    return {
        "S0": S0,
        "K": float(K),
        "B": float(barrier_pct * S0),
        "r": float(puller.risk_free_rate),
        "sigma": float(sigma),
        "T": T_default,                            # use actual days-to-expiry
        "expiry_str": expiry_str,
        "days_to_exp": days_to_exp,
        "side_used": side_used
    }


# =============================================================================
# [C] MC pricer (Euler & Milstein) with Brownian-Bridge KO — chunked
# =============================================================================
def price_do_put_mc_both(S0, K, B, r, sigma, T, *, steps_per_year=365,
                         n_paths=200_000, seed=2025, chunk_size=200_000):
    """
    Chunked MC for a *down-and-out* put with continuous monitoring via
    Brownian-bridge correction. Returns ((pe,se,ci95), (pm,se,ci95)).
    """
    if B <= 0: raise ValueError("Barrier must be > 0.")
    if S0 <= B:
        zero = (0.0, 0.0, (0.0, 0.0))
        return zero, zero

    steps_total = max(1, int(np.ceil(steps_per_year * T)))
    dt = T / steps_total
    rng = np.random.default_rng(seed)

    # Running accumulators for discounted payoffs
    def _acc(): return {"sum": 0.0, "sumsq": 0.0, "n": 0}
    acc_eu, acc_mi = _acc(), _acc()

    def _upd(acc, vals):
        acc["sum"]   += vals.sum()
        acc["sumsq"] += (vals * vals).sum()
        acc["n"]     += vals.size

    disc = np.exp(-r * T)
    done = 0
    while done < n_paths:
        m = min(chunk_size, n_paths - done)

        # States and survival flags
        S_eu = np.full(m, S0, float)
        S_mi = np.full(m, S0, float)
        alive_eu = np.ones(m, bool)
        alive_mi = np.ones(m, bool)

        for _ in range(steps_total):
            Z = rng.standard_normal(m)  # CRNs for both schemes
            U = rng.random(m)           # uniforms for bridge KO

            # Euler step + bridge
            S_prev = S_eu
            S_eu = _euler_step(S_prev, r, sigma, dt, Z)

            end_ko = alive_eu & ((S_prev <= B) | (S_eu <= B) | (S_eu <= 0.0))
            alive_eu &= ~end_ko
            mask = alive_eu & (S_prev > B) & (S_eu > B)
            if mask.any():
                p_hit = _bridge_knock_prob(S_prev[mask], S_eu[mask], B, sigma, dt)
                alive_eu[mask] &= (U[mask] >= p_hit)

            # Milstein step + bridge (reuse Z, U)
            S_prev = S_mi
            S_mi = _milstein_step(S_prev, r, sigma, dt, Z)

            end_ko = alive_mi & ((S_prev <= B) | (S_mi <= B) | (S_mi <= 0.0))
            alive_mi &= ~end_ko
            mask = alive_mi & (S_prev > B) & (S_mi > B)
            if mask.any():
                p_hit = _bridge_knock_prob(S_prev[mask], S_mi[mask], B, sigma, dt)
                alive_mi[mask] &= (U[mask] >= p_hit)

            if not (alive_eu.any() or alive_mi.any()):
                break

        # Discounted terminal payoffs (zero if knocked out)
        pay_eu = np.maximum(K - S_eu, 0.0); pay_eu[~alive_eu] = 0.0
        pay_mi = np.maximum(K - S_mi, 0.0); pay_mi[~alive_mi] = 0.0
        _upd(acc_eu, disc * pay_eu)
        _upd(acc_mi, disc * pay_mi)
        done += m

    # Finalize stats
    def _final(acc):
        n = max(acc["n"], 1)
        mean = acc["sum"] / n
        var  = max(acc["sumsq"] / n - mean * mean, 0.0)
        se   = np.sqrt(var / n)
        return mean, se, (mean - 1.96 * se, mean + 1.96 * se)

    return _final(acc_eu), _final(acc_mi)


# =============================================================================
# [D] Convergence sweeps
# =============================================================================
def sweep_vs_paths(mkt, paths=(100, 1_000, 10_000, 100_000, 1_000_000),
                   steps_per_year=250, seed=4242, chunk_size=200_000):
    out = {"paths": [], "euler_price": [], "euler_se": [], "milstein_price": [], "milstein_se": []}
    for n in paths:
        (pe, see, _), (pm, sem, _) = price_do_put_mc_both(
            mkt["S0"], mkt["K"], mkt["B"], mkt["r"], mkt["sigma"], mkt["T"],
            steps_per_year=steps_per_year, n_paths=int(n),
            seed=seed + int(n), chunk_size=chunk_size
        )
        out["paths"].append(int(n))
        out["euler_price"].append(pe);    out["euler_se"].append(see)
        out["milstein_price"].append(pm); out["milstein_se"].append(sem)
    return out

def sweep_vs_steps(mkt, steps=(50, 100, 250, 500, 1000),
                   n_paths=100_000, seed=2025, chunk_size=200_000):
    out = {"steps": [], "euler_price": [], "euler_se": [], "milstein_price": [], "milstein_se": []}
    for s in steps:
        (pe, see, _), (pm, sem, _) = price_do_put_mc_both(
            mkt["S0"], mkt["K"], mkt["B"], mkt["r"], mkt["sigma"], mkt["T"],
            steps_per_year=int(s), n_paths=int(n_paths),
            seed=seed + int(s), chunk_size=chunk_size
        )
        out["steps"].append(int(s))
        out["euler_price"].append(pe);    out["euler_se"].append(see)
        out["milstein_price"].append(pm); out["milstein_se"].append(sem)
    return out


# =============================================================================
# [E] Plot helpers (badges + charts)
# =============================================================================
def _market_badge_text(mkt):
    S0, K, B = mkt["S0"], mkt["K"], mkt["B"]
    r, sigma, T = mkt["r"], mkt["sigma"], mkt["T"]
    return f"S\u2080={S0:.2f}   K={K:.2f}   B={B:.2f}   r={r*100:.2f}%   σ={sigma*100:.2f}%   T={T:.2f}y"

def _add_badge(ax, text, loc="upper left", alpha=0.18):
    at = AnchoredText(text, loc=loc, prop=dict(size=9, color="black"),
                      frameon=True, borderpad=0.6)
    at.patch.set_boxstyle("round,pad=0.3,rounding_size=0.8")
    at.patch.set_alpha(alpha)
    ax.add_artist(at)

def plot_vs_paths(res, fixed_steps_per_year, mkt):
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.array(res["paths"])
    e, se = np.array(res["euler_price"]),    1.96 * np.array(res["euler_se"])
    m, sm = np.array(res["milstein_price"]), 1.96 * np.array(res["milstein_se"])

    ax.errorbar(x, e, yerr=se, fmt="o-", label="Euler (±95% CI)")
    ax.errorbar(x, m, yerr=sm, fmt="s-", label="Milstein (±95% CI)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("Down-and-Out PUT price\n(85% barrier, ATM strike)")
    ax.set_title("Option Price Path Convergence (Brownian-Bridge MC)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    _add_badge(ax, _market_badge_text(mkt), loc="upper left")
    _add_badge(ax, f"Fixed steps/year = {fixed_steps_per_year}", loc="lower left")
    fig.tight_layout()

def plot_vs_steps(res, fixed_n_paths, mkt):
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.array(res["steps"])
    e, se = np.array(res["euler_price"]),    1.96 * np.array(res["euler_se"])
    m, sm = np.array(res["milstein_price"]), 1.96 * np.array(res["milstein_se"])

    ax.errorbar(x, e, yerr=se, fmt="o-", label="Euler (±95% CI)")
    ax.errorbar(x, m, yerr=sm, fmt="s-", label="Milstein (±95% CI)")
    ax.set_xlabel("Time steps per year")
    ax.set_ylabel("Down-and-Out PUT price\n(85% barrier, ATM strike)")
    ax.set_title("Option Price Step Convergence (Brownian-Bridge MC)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _add_badge(ax, _market_badge_text(mkt), loc="upper left")
    _add_badge(ax, f"Fixed paths = {fixed_n_paths:,}", loc="lower left")
    fig.tight_layout()


# =============================================================================
# [F] Main
# =============================================================================
if __name__ == "__main__":
    # Pull market with CN-consistent ATM/IV extraction
    puller = OptionDataPuller("PLTR")
    market = prepare_pltr_market_inputs(puller, barrier_pct=0.85, prefer_side="auto")

    print("\nMarket used for convergence tests:")
    print(f"  S0={market['S0']:.4f}  K={market['K']:.4f}  B={market['B']:.4f}  "
          f"σ={market['sigma']:.4f}  r={market['r']:.4f}  T={market['T']:.2f}")
    print(f"  Chosen chain expiry: {market['expiry_str']} (~{market['days_to_exp']} days)  "
          f"|  IV side used: {market['side_used']}  |  Pricing horizon: {market['T']:.2f}y\n")

    # Settings
    PATHS_LIST            = [100, 1_000, 10_000, 100_000, 1_000_000]
    STEPS_LIST            = [50, 100, 250, 500, 1000]
    FIXED_STEPS_FOR_PATHS = 250
    FIXED_PATHS_FOR_STEPS = 100_000
    CHUNK_SIZE            = 200_000

    # Sweeps
    res_paths = sweep_vs_paths(market, PATHS_LIST,
                               steps_per_year=FIXED_STEPS_FOR_PATHS,
                               seed=4242, chunk_size=CHUNK_SIZE)
    res_steps = sweep_vs_steps(market, STEPS_LIST,
                               n_paths=FIXED_PATHS_FOR_STEPS,
                               seed=2025, chunk_size=CHUNK_SIZE)

    # Console tables
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

    # Plots
    plot_vs_paths(res_paths, FIXED_STEPS_FOR_PATHS, market)
    plot_vs_steps(res_steps, FIXED_PATHS_FOR_STEPS, market)
    plt.show()
