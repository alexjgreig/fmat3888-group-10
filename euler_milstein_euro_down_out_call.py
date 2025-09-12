import numpy as np
import matplotlib.pyplot as plt


# Monte Carlo pricing mechanism for barrier options
def mc_barrier_price_from_payoffs(payoffs, r, T):
    """
    Given per-path payoffs (0 if knocked-out), return:
      - discounted Monte Carlo price
      - standard error (SE)
      - 95% confidence interval (CI)
    """
    disc = np.exp(-r * T)
    vals = disc * payoffs
    price = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(vals.size)
    ci95 = (price - 1.96 * se, price + 1.96 * se)
    return price, se, ci95


# One-step SDE updates
def _euler_step(S, r, sigma, dt, Z):
    """
    Euler step for GBM under Q:
      S_{n+1} = S_n + r S_n dt + sigma S_n sqrt(dt) * Z
    """
    return S + r * S * dt + sigma * S * np.sqrt(dt) * Z

def _milstein_step(S, r, sigma, dt, Z):
    """
    Milstein step for GBM under Q:
      dW = sqrt(dt) * Z
      S_{n+1} = S_n + r S_n dt + sigma S_n dW + 0.5 sigma^2 S_n (dW^2 - dt)
    """
    dW = np.sqrt(dt) * Z
    return S + r * S * dt + sigma * S * dW + 0.5 * (sigma**2) * S * (dW**2 - dt)



# Barrier simulators with FIXED monitoring frequency
#       Barrier is checked ONLY at monitor ends.
#       Time grid between monitors is refined via substeps.
def simulate_down_and_out_euler_fixed_monitoring(
    S0, K, B, r, sigma, T,
    monitors_per_year, steps_per_year, Z
):
    """
    Down-and-out call with DISCRETE monitoring at fixed frequency.
    Euler stepping inside each monitoring interval.

    Parameters
    ----------
    S0, K, B, r, sigma, T : floats
        Usual contract/market inputs.
    monitors_per_year : int
        How many times per year the barrier is checked (e.g., 12 for monthly).
    steps_per_year : int
        Total Euler steps per year (must be a MULTIPLE of monitors_per_year).
    Z : ndarray, shape (n_paths, steps_total)
        Standard normals for all time steps. We reuse the same Z for Euler/Milstein comparisons.

    Returns
    -------
    payoffs : ndarray (n_paths,)
        Zero for knocked-out paths; otherwise max(S_T - K, 0).
    """
    n_paths = Z.shape[0]
    if B >= S0:
        # Assumes an initial check at t=0 -> immediate KO. Remove this guard if initial S0 < B should be allowed.
        return np.zeros(n_paths, dtype=float)

    # Validate divisibility so each monitor has the same number of substeps
    if steps_per_year % monitors_per_year != 0:
        raise ValueError("steps_per_year must be a multiple of monitors_per_year "
                         f"(got {steps_per_year=} and {monitors_per_year=}).")

    n_monitors = int(round(monitors_per_year * T))
    steps_total = int(round(steps_per_year * T))
    substeps_per_monitor = steps_per_year // monitors_per_year  # integer by constraint

    dt_monitor = T / n_monitors
    dt = dt_monitor / substeps_per_monitor

    S = np.full(n_paths, S0, dtype=float)
    alive = np.ones(n_paths, dtype=bool)

    col = 0
    for m in range(n_monitors):
        # substeps inside the m-th monitoring period
        for s in range(substeps_per_monitor):
            S = _euler_step(S, r, sigma, dt, Z[:, col])
            col += 1
        # monitor at END of the period
        knocked = S <= B
        alive &= (~knocked)
        if not np.any(alive):
            break

    payoffs = np.zeros(n_paths, dtype=float)
    if np.any(alive):
        payoffs[alive] = np.maximum(S[alive] - K, 0.0)
    return payoffs


def simulate_down_and_out_milstein_fixed_monitoring(
    S0, K, B, r, sigma, T,
    monitors_per_year, steps_per_year, Z
):
    """
    Down-and-out call with DISCRETE monitoring at fixed frequency.
    Milstein stepping inside each monitoring interval.

    Same interface/behavior as Euler version above.
    """
    n_paths = Z.shape[0]
    if B >= S0:
        return np.zeros(n_paths, dtype=float)

    if steps_per_year % monitors_per_year != 0:
        raise ValueError("steps_per_year must be a multiple of monitors_per_year "
                         f"(got {steps_per_year=} and {monitors_per_year=}).")

    n_monitors = int(round(monitors_per_year * T))
    steps_total = int(round(steps_per_year * T))
    substeps_per_monitor = steps_per_year // monitors_per_year

    dt_monitor = T / n_monitors
    dt = dt_monitor / substeps_per_monitor

    S = np.full(n_paths, S0, dtype=float)
    alive = np.ones(n_paths, dtype=bool)

    col = 0
    for m in range(n_monitors):
        for s in range(substeps_per_monitor):
            S = _milstein_step(S, r, sigma, dt, Z[:, col])
            col += 1
        knocked = S <= B
        alive &= (~knocked)
        if not np.any(alive):
            break

    payoffs = np.zeros(n_paths, dtype=float)
    if np.any(alive):
        payoffs[alive] = np.maximum(S[alive] - K, 0.0)
    return payoffs


# Convergence experiment (fixed monitoring; refine steps_per_year)
def barrier_convergence_experiment_fixed_monitoring(
    S0, K, B, r, sigma, T,
    monitors_per_year=12,
    steps_per_year_list=(12, 24, 48, 96, 192, 384, 768),
    n_paths=10000, seed=3888
):
    """
    For each steps_per_year in steps_per_year_list (all multiples of monitors_per_year):
      - simulate Euler & Milstein prices (same Z for fair comparison)
      - compute SE, 95% CI
    Uses the Milstein price at the *finest* steps_per_year in the list as an
    in-sweep reference to compute simple |bias| curves (no extra heavy run).
    """
    # ensure ascending order so the last entry is the finest grid
    steps_per_year_list = sorted(list(steps_per_year_list))

    rng = np.random.default_rng(seed)

    out = {
        "steps_per_year": [],
        "euler_price": [], "euler_se": [], "euler_ci": [],
        "milstein_price": [], "milstein_se": [], "milstein_ci": []
        # we'll add ref_price and biases after the loop
    }

    for spy in steps_per_year_list:
        if spy % monitors_per_year != 0:
            raise ValueError(
                f"All steps_per_year values must be multiples of monitors_per_year "
                f"(got steps_per_year={spy} and monitors_per_year={monitors_per_year})."
            )

        steps_total = int(round(spy * T))
        Z = rng.standard_normal((n_paths, steps_total))  # common random numbers per spy

        # Euler
        pay_eu = simulate_down_and_out_euler_fixed_monitoring(
            S0, K, B, r, sigma, T, monitors_per_year, spy, Z
        )
        pe, see, cie = mc_barrier_price_from_payoffs(pay_eu, r, T)

        # Milstein (reuse Z for fair variance comparison)
        pay_mi = simulate_down_and_out_milstein_fixed_monitoring(
            S0, K, B, r, sigma, T, monitors_per_year, spy, Z
        )
        pm, sem, cim = mc_barrier_price_from_payoffs(pay_mi, r, T)

        out["steps_per_year"].append(spy)
        out["euler_price"].append(pe);     out["euler_se"].append(see);     out["euler_ci"].append(cie)
        out["milstein_price"].append(pm);  out["milstein_se"].append(sem);  out["milstein_ci"].append(cim)


    return out



# Convergence experiment (fixed monitoring; increasing number of simulated paths)
def paths_convergence_experiment_fixed_steps(
    S0, K, B, r, sigma, T,
    monitors_per_year=12,
    steps_per_year=192,              # choose a reasonably fine time grid
    paths_list=(2_000, 5_000, 10_000, 20_000, 50_000, 100_000),
    base_seed=4242
):
    """
    Keep monitoring frequency and steps_per_year fixed, vary n_paths.
    Returns a dict with prices, SEs, and CIs for Euler and Milstein.
    Each n_paths run uses a fresh RNG stream (seed = base_seed + n_paths).
    """
    # sanity: steps_per_year must be a multiple of monitors_per_year
    if steps_per_year % monitors_per_year != 0:
        raise ValueError("steps_per_year must be a multiple of monitors_per_year.")
    steps_total = int(round(steps_per_year * T))

    out = {
        "paths": [],
        "euler_price": [], "euler_se": [], "euler_ci": [],
        "milstein_price": [], "milstein_se": [], "milstein_ci": []
    }

    for n_paths in paths_list:
        rng = np.random.default_rng(base_seed + int(n_paths))
        Z = rng.standard_normal((n_paths, steps_total))  # common shocks for fair comparison

        # Euler
        pay_eu = simulate_down_and_out_euler_fixed_monitoring(
            S0, K, B, r, sigma, T, monitors_per_year, steps_per_year, Z
        )
        pe, see, cie = mc_barrier_price_from_payoffs(pay_eu, r, T)

        # Milstein (reuse Z)
        pay_mi = simulate_down_and_out_milstein_fixed_monitoring(
            S0, K, B, r, sigma, T, monitors_per_year, steps_per_year, Z
        )
        pm, sem, cim = mc_barrier_price_from_payoffs(pay_mi, r, T)

        out["paths"].append(int(n_paths))
        out["euler_price"].append(pe);     out["euler_se"].append(see);     out["euler_ci"].append(cie)
        out["milstein_price"].append(pm);  out["milstein_se"].append(sem);  out["milstein_ci"].append(cim)

    return out




# Visualisation/Example
if __name__ == "__main__":
    # Market parameters: update these with yahoo finance data
    S0, K, r, sigma, T = 100.0, 100.0, 0.07, 0.20, 1.0
    B = 90.0

    # Monitoring frequency fixed (monthly)
    monitors_per_year = 12

    # Refine time grid without changing monitoring frequency
    steps_per_year_list = [12, 24, 48, 96, 192, 384, 768]  # all multiples of 12

    # Monte Carlo sizes
    n_paths = 120_000
    seed = 2025

    res = barrier_convergence_experiment_fixed_monitoring(
        S0, K, B, r, sigma, T,
        monitors_per_year,
        steps_per_year_list,
        n_paths, seed
    )

    # Price vs steps_per_year (±95% CI)
    plt.figure(figsize=(8, 5))
    e_err = np.array(res["euler_se"]) * 1.96
    m_err = np.array(res["milstein_se"]) * 1.96
    plt.errorbar(res["steps_per_year"], res["euler_price"], yerr=e_err, fmt="o-", label="Euler (±95% CI)")
    plt.errorbar(res["steps_per_year"], res["milstein_price"], yerr=m_err, fmt="s-", label="Milstein (±95% CI)")
    plt.xlabel("Time steps per year")
    plt.ylabel("Down-and-out call price")
    plt.title("Option Price Convergence over Time Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    print("steps/yr | Euler_price   SE       | Milstein_price   SE")
    for spy, pe, see, pm, sem in zip(
        res["steps_per_year"], res["euler_price"], res["euler_se"],
        res["milstein_price"], res["milstein_se"]
    ):
        print(f"{spy:8d} | {pe:12.6f} {see:9.6f} | {pm:13.6f} {sem:9.6f}")

    plt.show()


    # Fix a time grid and vary the number of paths
    fixed_steps_per_year = 192
    paths_list = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000]  # tune to your machine

    res_paths = paths_convergence_experiment_fixed_steps(
        S0, K, B, r, sigma, T,
        monitors_per_year=monitors_per_year,
        steps_per_year=fixed_steps_per_year,
        paths_list=paths_list,
        base_seed=904
    )

    # Price vs number of paths (with 95% CI error bars)
    plt.figure(figsize=(8,5))
    e_half = 1.96 * np.array(res_paths["euler_se"])
    m_half = 1.96 * np.array(res_paths["milstein_se"])
    plt.errorbar(res_paths["paths"], res_paths["euler_price"], yerr=e_half, fmt="o-", label="Euler (±95% CI)")
    plt.errorbar(res_paths["paths"], res_paths["milstein_price"], yerr=m_half, fmt="s-", label="Milstein (±95% CI)")
    plt.xscale("log")
    plt.xlabel("Number of simulated paths (log scale)")
    plt.ylabel(f"Down-and-out call price (steps/yr={fixed_steps_per_year})")
    plt.title("Variance convergence over number of paths")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    print("\npaths  | Euler_price   SE       | Milstein_price   SE")
    for n, pe, see, pm, sem in zip(
        res_paths["paths"], res_paths["euler_price"], res_paths["euler_se"],
        res_paths["milstein_price"], res_paths["milstein_se"]
    ):
        print(f"{n:6d} | {pe:12.6f} {see:9.6f} | {pm:13.6f} {sem:9.6f}")