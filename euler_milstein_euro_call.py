import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, os, sys
from math import log, sqrt, exp, erf
from matplotlib.offsetbox import AnchoredText
warnings.filterwarnings("ignore")

# Make local package importable (matches your project layout)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from option_pricing.data.yahoo_data import OptionDataPuller


# =============================== Black–Scholes ===============================

def norm_cdf(x):  # for completeness (matches your helper)
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_call(S0, K, r, sigma, T, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S0*exp(-q*T) - K*exp(-r*T), 0.0)
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * exp(-q*T) * norm_cdf(d1) - K * exp(-r*T) * norm_cdf(d2)


# ============================== MC primitives ===============================

def _euler_terminal(S0, r, sigma, T, steps, Z):
    dt = T / steps
    S = np.full(Z.shape[0], S0, dtype=float)
    for n in range(steps):
        S += r * S * dt + sigma * S * np.sqrt(dt) * Z[:, n]
    return S

def _milstein_terminal(S0, r, sigma, T, steps, Z):
    dt = T / steps
    sqrt_dt = np.sqrt(dt)
    S = np.full(Z.shape[0], S0, dtype=float)
    for n in range(steps):
        dW = sqrt_dt * Z[:, n]
        S += r * S * dt + sigma * S * dW + 0.5 * (sigma**2) * S * (dW**2 - dt)
    return S

def _price_chunked_call_both_methods(S0, K, r, sigma, T, *,
                                     steps_per_year, n_paths,
                                     seed=2025, chunk_size=200_000):
    """Chunked MC: returns (price, se) for Euler & Milstein using common random numbers."""
    steps_total = max(1, int(np.ceil(steps_per_year * T)))
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    def acc_init():
        return {"sum": 0.0, "sumsq": 0.0, "n": 0}

    acc_eu, acc_mi = acc_init(), acc_init()

    def acc_update(acc, vals):
        acc["sum"] += vals.sum()
        acc["sumsq"] += (vals**2).sum()
        acc["n"] += vals.size

    done = 0
    while done < n_paths:
        m = min(chunk_size, n_paths - done)
        Z = rng.standard_normal((m, steps_total))  # CRNs for both schemes

        ST_eu = _euler_terminal(S0, r, sigma, T, steps_total, Z)
        ST_mi = _milstein_terminal(S0, r, sigma, T, steps_total, Z)

        pay_eu = np.maximum(ST_eu - K, 0.0)
        pay_mi = np.maximum(ST_mi - K, 0.0)

        vals_eu = disc * pay_eu
        vals_mi = disc * pay_mi

        acc_update(acc_eu, vals_eu)
        acc_update(acc_mi, vals_mi)

        done += m

    def finalize(acc):
        n = acc["n"]
        mean = acc["sum"] / n
        var = max(acc["sumsq"] / n - mean * mean, 0.0)
        se = np.sqrt(var / n)
        return mean, se

    return finalize(acc_eu), finalize(acc_mi)


# ============================= Market preparation ============================

def _pick_expiry_close_to_1y(puller):
    exps = puller.get_option_expirations()
    if not exps:
        raise RuntimeError("No option expirations available.")
    ts = pd.to_datetime(exps, utc=True)
    today = pd.Timestamp.now(tz="UTC").normalize()
    days = (ts - today).days.to_numpy()
    idx = np.abs(days - 365).argmin()
    return exps[idx], int(days[idx])

def _atm_strike_and_iv_from_chain(chain, spot, r, puller):
    calls = chain.get("calls", pd.DataFrame())
    puts  = chain.get("puts",  pd.DataFrame())
    df = pd.concat([calls, puts], ignore_index=True) if (not calls.empty or not puts.empty) else pd.DataFrame()
    if df.empty:
        raise RuntimeError("Empty option chain for selected expiry.")

    if "strike" not in df.columns:
        raise RuntimeError("Option chain missing 'strike' column.")

    df = df.copy()
    df["abs_mis"] = (df["strike"] - float(spot)).abs()
    row = df.sort_values("abs_mis").iloc[0]

    K = float(row["strike"])
    sigma = float(row.get("impliedVolatility", np.nan))

    if not np.isfinite(sigma) or sigma <= 0:
        # backup: invert from mid using your puller
        mid = puller.calculate_mid_price(row.get("bid", np.nan),
                                         row.get("ask", np.nan),
                                         row.get("lastPrice", np.nan))
        T_chain = float(row.get("timeToExpiry", np.nan))
        opt_type = str(row.get("optionType", "call")).lower()
        if not (mid and np.isfinite(T_chain) and T_chain > 0):
            raise RuntimeError("Cannot back out IV: missing mid or timeToExpiry.")
        iv = puller.calculate_implied_volatility(mid, float(spot), K, T_chain, opt_type, dividend_yield=0.0)
        if iv is None or iv <= 0:
            raise RuntimeError("IV inversion failed for ATM option.")
        sigma = float(iv)

    if sigma > 2.0:  # guard if IV is in percent
        sigma *= 0.01

    return K, sigma

def prepare_pltr_market_inputs_1yATM():
    puller = OptionDataPuller("PLTR")
    S0 = float(puller.get_current_price())
    expiry_str, days_to_exp = _pick_expiry_close_to_1y(puller)
    chain = puller.get_option_chain(expiry_str)
    K, sigma = _atm_strike_and_iv_from_chain(chain, S0, puller.risk_free_rate, puller)
    r = float(puller.risk_free_rate)
    T = 1.0  # per your spec: price a 1.00-year option even if listed tenor differs
    return dict(S0=S0, K=K, r=r, sigma=sigma, T=T,
                expiry_str=expiry_str, days_to_exp=days_to_exp)


# ============================ Convergence experiments ========================

def run_paths_convergence(mkt, paths_list=(100, 1_000, 10_000, 100_000, 1_000_000),
                          fixed_steps_per_year=250, base_seed=4242, chunk_size=200_000):
    res = {"paths": [], "euler_price": [], "euler_se": [], "milstein_price": [], "milstein_se": []}
    for n in paths_list:
        (pe, see), (pm, sem) = _price_chunked_call_both_methods(
            mkt["S0"], mkt["K"], mkt["r"], mkt["sigma"], mkt["T"],
            steps_per_year=fixed_steps_per_year, n_paths=int(n),
            seed=base_seed + int(n), chunk_size=chunk_size
        )
        res["paths"].append(int(n))
        res["euler_price"].append(pe);    res["euler_se"].append(see)
        res["milstein_price"].append(pm); res["milstein_se"].append(sem)
    return res

def run_steps_convergence(mkt, steps_list=(50, 100, 250, 500, 1000),
                          fixed_n_paths=100_000, base_seed=2025, chunk_size=200_000):
    res = {"steps": [], "euler_price": [], "euler_se": [], "milstein_price": [], "milstein_se": []}
    for spy in steps_list:
        (pe, see), (pm, sem) = _price_chunked_call_both_methods(
            mkt["S0"], mkt["K"], mkt["r"], mkt["sigma"], mkt["T"],
            steps_per_year=int(spy), n_paths=int(fixed_n_paths),
            seed=base_seed + int(spy), chunk_size=chunk_size
        )
        res["steps"].append(int(spy))
        res["euler_price"].append(pe);    res["euler_se"].append(see)
        res["milstein_price"].append(pm); res["milstein_se"].append(sem)
    return res


# ================================== Plots ====================================

def _market_badge_text(S0, K, r, sigma, T):
    # Plain text (no mathtext) with unicode S₀ and σ
    return f"S\u2080={S0:.2f}   K={K:.2f}   r={r*100:.2f}%   σ={sigma*100:.2f}%   T={T:.2f}y"

def _add_market_badge(ax, S0, K, r, sigma, T, loc="upper left"):
    txt = _market_badge_text(S0, K, r, sigma, T)
    at = AnchoredText(txt, loc=loc, prop=dict(size=9, color="black"),
                      frameon=True, borderpad=0.6)
    at.patch.set_boxstyle("round,pad=0.3,rounding_size=0.8")
    at.patch.set_alpha(0.18)  # light translucent background
    ax.add_artist(at)

    

def plot_paths_convergence(res, fixed_steps_per_year, bs_price, market):
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.array(res["paths"])
    e = np.array(res["euler_price"]);   se = 1.96*np.array(res["euler_se"])
    m = np.array(res["milstein_price"]); sm = 1.96*np.array(res["milstein_se"])

    ax.errorbar(x, e, yerr=se, fmt="o-", label="Euler (±95% CI)")
    ax.errorbar(x, m, yerr=sm, fmt="s-", label="Milstein (±95% CI)")
    ax.axhline(bs_price, linestyle="--", linewidth=1.5, label=f"Black–Scholes = {bs_price:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("European CALL price")
    ax.set_title(f"Convergence vs paths (steps/year = {fixed_steps_per_year})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _add_market_badge(ax, market["S0"], market["K"], market["r"], market["sigma"], market["T"])
    fig.tight_layout()


def plot_steps_convergence(res, fixed_n_paths, bs_price, market):
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.array(res["steps"])
    e = np.array(res["euler_price"]);   se = 1.96*np.array(res["euler_se"])
    m = np.array(res["milstein_price"]); sm = 1.96*np.array(res["milstein_se"])

    ax.errorbar(x, e, yerr=se, fmt="o-", label="Euler (±95% CI)")
    ax.errorbar(x, m, yerr=sm, fmt="s-", label="Milstein (±95% CI)")
    ax.axhline(bs_price, linestyle="--", linewidth=1.5, label=f"Black–Scholes = {bs_price:.4f}")
    ax.set_xlabel("Time steps per year")
    ax.set_ylabel("European CALL price")
    ax.set_title(f"Convergence vs steps/year (paths = {fixed_n_paths:,})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _add_market_badge(ax, market["S0"], market["K"], market["r"], market["sigma"], market["T"])
    fig.tight_layout()



# ================================== Main =====================================

if __name__ == "__main__":
    # Pull market
    market = prepare_pltr_market_inputs_1yATM()
    S0, K, r, sigma, T = market["S0"], market["K"], market["r"], market["sigma"], market["T"]
    print("PLTR vanilla call — market inputs:")
    print(f"  S0={S0:.4f}  K={K:.4f}  σ={sigma:.4f}  r={r:.4f}  T={T:.2f}")
    print(f"  Chain expiry chosen: {market['expiry_str']} (~{market['days_to_exp']} days); pricing horizon fixed at 1.00y\n")

    # Analytic price for reference (q=0 by default)
    bs_px = bs_call(S0, K, r, sigma, T, q=0.0)
    print(f"Black–Scholes reference: {bs_px:.6f}\n")

    # Convergence settings
    PATHS_LIST = [100, 1_000, 10_000, 100_000, 1_000_000]
    STEPS_LIST = [50, 100, 250, 500, 1000]
    FIXED_STEPS_FOR_PATHS = 250
    FIXED_PATHS_FOR_STEPS = 100_000
    CHUNK = 200_000

    # Sweeps
    res_paths = run_paths_convergence(market, PATHS_LIST, FIXED_STEPS_FOR_PATHS, base_seed=4242, chunk_size=CHUNK)
    res_steps = run_steps_convergence(market, STEPS_LIST, FIXED_PATHS_FOR_STEPS, base_seed=2025, chunk_size=CHUNK)

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

    # Plots with BS line
    plot_paths_convergence(res_paths, FIXED_STEPS_FOR_PATHS, bs_px, market)
    plot_steps_convergence(res_steps, FIXED_PATHS_FOR_STEPS, bs_px, market)
    plt.show()







