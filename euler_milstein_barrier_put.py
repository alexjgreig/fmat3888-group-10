# euler_milstein_barrier_put_pltr.py
import numpy as np
import pandas as pd
import warnings, os, sys
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
warnings.filterwarnings("ignore")

# Make local package importable (matches your project layout)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from option_pricing.data.yahoo_data import OptionDataPuller


# =============================== MC PRIMITIVES ===============================

def mc_discount_stats(payoffs, r, T):
    """Return discounted price, standard error, 95% CI."""
    disc = np.exp(-r * T)
    vals = disc * payoffs
    price = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    ci95 = (price - 1.96 * se, price + 1.96 * se)
    return price, se, ci95

def _euler_step(S, r, sigma, dt, Z):
    # Euler–Maruyama in price space
    return S + r * S * dt + sigma * S * np.sqrt(dt) * Z

def _milstein_step(S, r, sigma, dt, Z):
    # Milstein in price space
    dW = np.sqrt(dt) * Z
    return S + r * S * dt + sigma * S * dW + 0.5 * (sigma**2) * S * (dW**2 - dt)

def _bridge_knock_prob(S0, S1, B, sigma, dt):
    """
    Brownian-bridge crossing probability for a *down* barrier B on log-price.
    Valid when both endpoints are > B. Vectorized.
    p_hit = exp(-2 * ln(S0/B) * ln(S1/B) / (sigma^2 * dt))
    """
    x = np.log(S0) - np.log(B)
    y = np.log(S1) - np.log(B)
    expo = -2.0 * (x * y) / (sigma**2 * dt)
    p = np.exp(np.clip(expo, -700.0, 50.0))
    return np.clip(p, 0.0, 1.0)

def price_down_and_out_put_euler_milstein_bb(
    S0, K, B, r, sigma, T, *, steps_per_year=365, n_paths=200_000, seed=2025
):
    """
    Price a European down-and-out PUT with continuous monitoring using
    Brownian-bridge correction. Returns dict with Euler/Milstein prices & stats.
    """
    if B <= 0:
        raise ValueError("Barrier must be > 0.")
    if S0 <= B:
        zero = (0.0, 0.0, (0.0, 0.0))
        return {
            "euler": {"price": 0.0, "se": 0.0, "ci95": (0.0, 0.0)},
            "milstein": {"price": 0.0, "se": 0.0, "ci95": (0.0, 0.0)}
        }

    steps_total = max(1, int(np.ceil(steps_per_year * T)))
    dt = T / steps_total
    rng = np.random.default_rng(seed)

    # State vectors
    S_eu = np.full(n_paths, S0, dtype=float)
    S_mi = np.full(n_paths, S0, dtype=float)
    alive_eu = np.ones(n_paths, dtype=bool)
    alive_mi = np.ones(n_paths, dtype=bool)

    for _ in range(steps_total):
        Z = rng.standard_normal(n_paths)  # common normals
        U = rng.random(n_paths)           # common uniforms for bridge KO

        # --- Euler step + bridge ---
        S_prev = S_eu.copy()
        S_eu = _euler_step(S_prev, r, sigma, dt, Z)

        end_ko = alive_eu & ((S_prev <= B) | (S_eu <= B) | (S_eu <= 0.0))
        alive_eu &= ~end_ko

        mask = alive_eu & (S_prev > B) & (S_eu > B)
        if mask.any():
            p_hit = _bridge_knock_prob(S_prev[mask], S_eu[mask], B, sigma, dt)
            ko = U[mask] < p_hit
            alive_eu[np.where(mask)[0][ko]] = False

        # --- Milstein step + bridge (reuse Z,U) ---
        S_prev = S_mi.copy()
        S_mi = _milstein_step(S_prev, r, sigma, dt, Z)

        end_ko = alive_mi & ((S_prev <= B) | (S_mi <= B) | (S_mi <= 0.0))
        alive_mi &= ~end_ko

        mask = alive_mi & (S_prev > B) & (S_mi > B)
        if mask.any():
            p_hit = _bridge_knock_prob(S_prev[mask], S_mi[mask], B, sigma, dt)
            ko = U[mask] < p_hit
            alive_mi[np.where(mask)[0][ko]] = False

        # Early exit if everything’s knocked out
        if not (alive_eu.any() or alive_mi.any()):
            break

    # Terminal put payoff if still alive
    pay_eu = np.maximum(K - S_eu, 0.0); pay_eu[~alive_eu] = 0.0
    pay_mi = np.maximum(K - S_mi, 0.0); pay_mi[~alive_mi] = 0.0

    pe, see, cie = mc_discount_stats(pay_eu, r, T)
    pm, sem, cim = mc_discount_stats(pay_mi, r, T)
    return {
        "euler": {"price": pe, "se": see, "ci95": cie},
        "milstein": {"price": pm, "se": sem, "ci95": cim},
    }


# ========================= MARKET INPUTS (PLTR, 1Y ATM) ======================

def _pick_expiry_close_to_1y(puller):
    exps = puller.get_option_expirations()
    ts = pd.to_datetime(exps, utc=True)                 # tz-aware UTC
    today = pd.Timestamp.now(tz="UTC").normalize()      # tz-aware UTC (no tz_localize)
    days = (ts - today).days.to_numpy()                 # TimedeltaIndex -> int days
    idx = np.abs(days - 365).argmin()
    return exps[idx], int(days[idx])

def _atm_strike_and_iv_from_chain(chain, spot, r, puller):
    """
    Combine calls/puts at chosen expiry; pick strike nearest to spot;
    return (K, sigma). Uses chain’s 'impliedVolatility' when available,
    otherwise inverts from mid price via OptionDataPuller.
    """
    calls = chain.get("calls", pd.DataFrame())
    puts  = chain.get("puts", pd.DataFrame())
    df = pd.concat([calls, puts], ignore_index=True) if (not calls.empty or not puts.empty) else pd.DataFrame()
    if df.empty:
        raise RuntimeError("Empty option chain for selected expiry.")

    if "strike" not in df or "impliedVolatility" not in df:
        raise RuntimeError("Option chain missing required columns.")

    df = df.copy()
    df["abs_mis"] = (df["strike"] - float(spot))**2
    row = df.sort_values("abs_mis").iloc[0]
    K = float(row["strike"])
    sigma = float(row["impliedVolatility"]) if pd.notna(row["impliedVolatility"]) else np.nan

    if not np.isfinite(sigma) or sigma <= 0:
        # Fallback: compute IV from mid price
        opt_type = str(row.get("optionType", "call")).lower()  # 'call'/'put'
        mid = float(row.get("mid", np.nan))
        T_chain = float(row.get("timeToExpiry", np.nan))  # years
        if not (np.isfinite(mid) and mid > 0 and np.isfinite(T_chain) and T_chain > 0):
            raise RuntimeError("Cannot back out IV: missing mid price or timeToExpiry.")
        sigma = puller.calculate_implied_volatility(
            option_price=mid, spot=float(spot), strike=K,
            time_to_expiry=T_chain, option_type=opt_type, dividend_yield=0.0
        )
        if sigma is None or sigma <= 0:
            raise RuntimeError("Implied volatility inversion failed.")

    # If IV looks like a percent (e.g., 35), convert to 0.35
    if sigma > 2.0:
        sigma *= 0.01
    return K, sigma

# ========================= Convergence helpers & plots =========================

def _prepare_pltr_market_inputs_1yATM(puller, barrier_pct=0.85):
    """Pick expiry closest to ~1y from chain, choose ATM strike, pull IV."""
    # expiry ~ 1y (make both tz-aware to avoid arithmetic errors)
    exps = puller.get_option_expirations()
    if not exps:
        raise RuntimeError("No option expirations available.")
    ts = pd.to_datetime(exps, utc=True)
    today = pd.Timestamp.now(tz="UTC").normalize()
    days = (ts - today).days.to_numpy()
    idx = np.abs(days - 365).argmin()
    expiry_str = exps[idx]
    days_to_exp = int(days[idx])

    # option chain at that expiry
    chain = puller.get_option_chain(expiry_str)
    calls = chain.get("calls", pd.DataFrame())
    puts  = chain.get("puts",  pd.DataFrame())
    df = pd.concat([calls, puts], ignore_index=True) if (not calls.empty or not puts.empty) else pd.DataFrame()
    if df.empty:
        raise RuntimeError("Empty option chain at selected expiry.")

    S0 = float(puller.get_current_price())
    df = df.copy()
    if "strike" not in df.columns:
        raise RuntimeError("Option chain missing 'strike' column.")
    # pick ATM by closest-to-spot strike
    df["abs_mis"] = (df["strike"] - S0).abs()
    row = df.sort_values("abs_mis").iloc[0]
    K = float(row["strike"])

    # IV: prefer chain's 'impliedVolatility'; if NaN, back out from mid using your inverter
    sigma = float(row.get("impliedVolatility", np.nan))
    if not np.isfinite(sigma) or sigma <= 0:
        # mid from bid/ask/last via your helper
        mid = puller.calculate_mid_price(row.get("bid", np.nan), row.get("ask", np.nan), row.get("lastPrice", np.nan))
        T_chain = float(row.get("timeToExpiry", max(days_to_exp,1)/365.25))
        opt_type = str(row.get("optionType", "call")).lower()  # 'call' or 'put'
        if not (mid and T_chain > 0):
            raise RuntimeError("Cannot back out IV: missing mid or timeToExpiry.")
        iv = puller.calculate_implied_volatility(mid, S0, K, T_chain, opt_type, dividend_yield=0.0)
        if iv is None or iv <= 0:
            raise RuntimeError("IV inversion failed for ATM option.")
        sigma = float(iv)
    if sigma > 2.0:  # guard in case IV is in percent
        sigma *= 0.01

    r = float(puller.risk_free_rate)
    B = float(barrier_pct * S0)
    # As per your spec, we price a *1.00 year* option, even if listed expiry is ~374d.
    T = 1.0

    return dict(S0=S0, K=K, B=B, r=r, sigma=sigma, T=T,
                expiry_str=expiry_str, days_to_exp=days_to_exp)

def _price_chunked_both_methods_bb(S0, K, B, r, sigma, T, *, steps_per_year, n_paths, seed=2025, chunk_size=200_000):
    """Chunked MC with Brownian-bridge continuous monitoring; returns (price, se, ci) for Euler & Milstein."""
    steps_total = max(1, int(np.ceil(steps_per_year * T)))
    dt = T / steps_total
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    # running sums for discounted payoffs
    def _acc_init():
        return {"sum": 0.0, "sumsq": 0.0, "n": 0}

    acc_eu, acc_mi = _acc_init(), _acc_init()

    def _acc_update(acc, vals):
        acc["sum"]   += vals.sum()
        acc["sumsq"] += (vals**2).sum()
        acc["n"]     += vals.size

    # loop over chunks
    done = 0
    while done < n_paths:
        m = min(chunk_size, n_paths - done)

        # states
        S_eu = np.full(m, S0, dtype=float)
        S_mi = np.full(m, S0, dtype=float)
        alive_eu = np.ones(m, dtype=bool)
        alive_mi = np.ones(m, dtype=bool)

        for _ in range(steps_total):
            Z = rng.standard_normal(m)
            U = rng.random(m)

            # Euler
            S_prev = S_eu
            S_eu = S_prev + r*S_prev*dt + sigma*S_prev*np.sqrt(dt)*Z
            end_ko = alive_eu & ((S_prev <= B) | (S_eu <= B) | (S_eu <= 0.0))
            alive_eu &= ~end_ko
            mask = alive_eu & (S_prev > B) & (S_eu > B)
            if mask.any():
                x = np.log(S_prev[mask]) - np.log(B)
                y = np.log(S_eu[mask])  - np.log(B)
                expo = -2.0 * (x*y) / (sigma**2 * dt)
                p_hit = np.exp(np.clip(expo, -700.0, 50.0))
                alive_eu[mask] &= (U[mask] >= np.clip(p_hit, 0.0, 1.0))

            # Milstein (reuse Z, U)
            S_prev = S_mi
            dW = np.sqrt(dt)*Z
            S_mi = S_prev + r*S_prev*dt + sigma*S_prev*dW + 0.5*(sigma**2)*S_prev*(dW**2 - dt)
            end_ko = alive_mi & ((S_prev <= B) | (S_mi <= B) | (S_mi <= 0.0))
            alive_mi &= ~end_ko
            mask = alive_mi & (S_prev > B) & (S_mi > B)
            if mask.any():
                x = np.log(S_prev[mask]) - np.log(B)
                y = np.log(S_mi[mask])  - np.log(B)
                expo = -2.0 * (x*y) / (sigma**2 * dt)
                p_hit = np.exp(np.clip(expo, -700.0, 50.0))
                alive_mi[mask] &= (U[mask] >= np.clip(p_hit, 0.0, 1.0))

            if not (alive_eu.any() or alive_mi.any()):
                break

        # discounted payoffs this chunk
        pay_eu = np.maximum(K - S_eu, 0.0); pay_eu[~alive_eu] = 0.0
        pay_mi = np.maximum(K - S_mi, 0.0); pay_mi[~alive_mi] = 0.0

        vals_eu = disc * pay_eu
        vals_mi = disc * pay_mi
        _acc_update(acc_eu, vals_eu)
        _acc_update(acc_mi, vals_mi)
        done += m

    def _finalize(acc):
        n = acc["n"]
        mean = acc["sum"] / n
        var  = (acc["sumsq"] - n*mean*mean) / max(n-1, 1)
        se   = np.sqrt(var / n)
        return mean, se, (mean - 1.96*se, mean + 1.96*se)

    return _finalize(acc_eu), _finalize(acc_mi)

def run_paths_convergence(market, paths_list=(100, 1_000, 10_000, 100_000, 1_000_000),
                          fixed_steps_per_year=250, base_seed=4242, chunk_size=200_000):
    res = {"paths": [], "euler_price": [], "euler_se": [], "milstein_price": [], "milstein_se": []}
    for n in paths_list:
        (pe, see, _), (pm, sem, _) = _price_chunked_both_methods_bb(
            market["S0"], market["K"], market["B"], market["r"], market["sigma"], market["T"],
            steps_per_year=fixed_steps_per_year, n_paths=int(n),
            seed=base_seed + int(n), chunk_size=chunk_size
        )
        res["paths"].append(int(n))
        res["euler_price"].append(pe);    res["euler_se"].append(see)
        res["milstein_price"].append(pm); res["milstein_se"].append(sem)
    return res

def run_steps_convergence(market, steps_list=(50, 100, 250, 500, 1000),
                          fixed_n_paths=100_000, base_seed=2025, chunk_size=200_000):
    res = {"steps": [], "euler_price": [], "euler_se": [], "milstein_price": [], "milstein_se": []}
    for spy in steps_list:
        (pe, see, _), (pm, sem, _) = _price_chunked_both_methods_bb(
            market["S0"], market["K"], market["B"], market["r"], market["sigma"], market["T"],
            steps_per_year=int(spy), n_paths=int(fixed_n_paths),
            seed=base_seed + int(spy), chunk_size=chunk_size
        )
        res["steps"].append(int(spy))
        res["euler_price"].append(pe);    res["euler_se"].append(see)
        res["milstein_price"].append(pm); res["milstein_se"].append(sem)
    return res

def _market_badge_text_barrier(market, percent=True):
    S0, K, B = market["S0"], market["K"], market["B"]
    r, sigma, T = market["r"], market["sigma"], market["T"]
    if percent:
        # Pretty badge with percentages (matches the vanilla plots)
        return f"S\u2080={S0:.2f}   K={K:.2f}   B={B:.2f}   r={r*100:.2f}%   σ={sigma*100:.2f}%   T={T:.2f}y"
    else:
        # Exact decimals, like your example line
        return f"S\u2080={S0:.4f}   K={K:.4f}   B={B:.4f}   r={r:.4f}   σ={sigma:.4f}   T={T:.2f}"

def _add_market_badge(ax, market, *, percent=True, loc="upper left"):
    txt = _market_badge_text_barrier(market, percent=percent)
    at = AnchoredText(txt, loc=loc, prop=dict(size=9, color="black"),
                      frameon=True, borderpad=0.6)
    at.patch.set_boxstyle("round,pad=0.3,rounding_size=0.8")
    at.patch.set_alpha(0.18)  # light translucent background
    ax.add_artist(at)

def plot_paths_convergence(res, fixed_steps_per_year, market, *, percent=True):
    fig, ax = plt.subplots(figsize=(8,5))
    e = np.array(res["euler_price"]); se = 1.96*np.array(res["euler_se"])
    m = np.array(res["milstein_price"]); sm = 1.96*np.array(res["milstein_se"])
    x = np.array(res["paths"])
    ax.errorbar(x, e, yerr=se, fmt="o-", label="Euler (±95% CI)")
    ax.errorbar(x, m, yerr=sm, fmt="s-", label="Milstein (±95% CI)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("Down-and-out PUT price")
    ax.set_title(f"Convergence vs paths (steps/year = {fixed_steps_per_year})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _add_market_badge(ax, market, percent=percent, loc="upper left")
    fig.tight_layout()


def plot_steps_convergence(res, fixed_n_paths, market, *, percent=True):
    fig, ax = plt.subplots(figsize=(8,5))
    e = np.array(res["euler_price"]); se = 1.96*np.array(res["euler_se"])
    m = np.array(res["milstein_price"]); sm = 1.96*np.array(res["milstein_se"])
    x = np.array(res["steps"])
    ax.errorbar(x, e, yerr=se, fmt="o-", label="Euler (±95% CI)")
    ax.errorbar(x, m, yerr=sm, fmt="s-", label="Milstein (±95% CI)")
    ax.set_xlabel("Time steps per year")
    ax.set_ylabel("Down-and-out PUT price")
    ax.set_title(f"Convergence vs steps/year (paths = {fixed_n_paths:,})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _add_market_badge(ax, market, percent=percent, loc="upper left")
    fig.tight_layout()


# ------------------------- Driver (call from __main__) -------------------------
if __name__ == "__main__":
    puller = OptionDataPuller("PLTR")
    market = _prepare_pltr_market_inputs_1yATM(puller, barrier_pct=0.85)

    print("\nMarket used for convergence tests:")
    print(f"  S0={market['S0']:.4f}  K={market['K']:.4f}  B={market['B']:.4f}  σ={market['sigma']:.4f}  r={market['r']:.4f}  T={market['T']:.2f}")
    print(f"  Chosen chain expiry: {market['expiry_str']} (~{market['days_to_exp']} days)  |  Pricing horizon: 1.00y\n")

    # Settings per your request
    PATHS_LIST = [100, 1_000, 10_000, 100_000, 1_000_000]
    STEPS_LIST = [50, 100, 250, 500, 1000]
    FIXED_STEPS_FOR_PATHS = 250
    FIXED_PATHS_FOR_STEPS = 100_000
    CHUNK = 200_000

    # Run sweeps
    res_paths = run_paths_convergence(market, paths_list=PATHS_LIST,
                                      fixed_steps_per_year=FIXED_STEPS_FOR_PATHS,
                                      base_seed=4242, chunk_size=CHUNK)
    res_steps = run_steps_convergence(market, steps_list=STEPS_LIST,
                                      fixed_n_paths=FIXED_PATHS_FOR_STEPS,
                                      base_seed=2025, chunk_size=CHUNK)

    # Plots
    plot_paths_convergence(res_paths, FIXED_STEPS_FOR_PATHS, market, percent=True)
    plot_steps_convergence(res_steps, FIXED_PATHS_FOR_STEPS, market, percent=True)

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

    plt.show()


