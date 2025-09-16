import math
import numpy as np
import csv

# -----------------------------
# 1) Fan fits: x_P(E) = C_P * E^(-alpha_P) (HIGH-energy branch)
C20, a20 = 0.05225, 1.3452
C50, a50 = 0.177, 0.8817
C80, a80 = 0.3137, 0.6643

def xP_over_D(E, C, a): # percentile at energy E
    return C * (E ** (-a))

# -----------------------------
# 2) Invert r -> b (r = (1 - 0.25^(1/b)) / (4^(1/b) - 1))
def b_from_r(r, tol=1e-10, b_lo=0.1, b_hi=20.0):
    def r_of_b(b):
        t1 = 0.25 ** (1.0 / b)
        t2 = 4.0 ** (1.0 / b)
        return (1.0 - t1) / (t2 - 1.0)

    r_lo, r_hi = r_of_b(b_lo), r_of_b(b_hi)
    if not (min(r_lo, r_hi) <= r <= max(r_lo, r_hi)):
        r = max(min(r, max(r_lo, r_hi)), min(r_lo, r_hi))
    for _ in range(80):
        b_mid = 0.5 * (b_lo + b_hi)
        r_mid = r_of_b(b_mid)
        if (r_mid - r) == 0 or abs(b_hi - b_lo) < 1e-12:
            return b_mid
        if (r_mid < r) == (r_lo < r):
            b_lo, r_lo = b_mid, r_mid
        else:
            b_hi, r_hi = b_mid, r_mid
    # print(0.5 * (b_lo + b_hi))
    return 0.5 * (b_lo + b_hi)

# -----------------------------
# 3) Map energy -> x_max via Swebrec geometry with x20, x50, x80
def xmax_from_E(E):
    x20 = xP_over_D(E, C20, a20)
    x50 = xP_over_D(E, C50, a50)
    x80 = xP_over_D(E, C80, a80)
    l8050 = math.log(x80 / x50)
    l2050 = math.log(x20 / x50)
    r = -l8050 / l2050
    b = b_from_r(r)
    # print(b)
    t = (0.25) ** (1.0 / b)
    A = -l8050 / (t - 1.0)
    return x50 * math.exp(A)

# -----------------------------
# 4) Solve implicit x_max = F(E_cs * (x_max/BASE)^lambda)
def xmax_from_Ecs(Ecs, lam, BASE=25.0, x0=None, iters=50, tol=1e-10):
    if x0 is None:
        x0 = 0.5 * BASE
    y = math.log(x0)
    for _ in range(iters):
        E = Ecs * (math.exp(y) / BASE) ** lam
        x_new = xmax_from_E(E)
        y_new = math.log(x_new)
        if abs(y_new - y) < tol:
            y = y_new
            break
        y = y_new
    return math.exp(y)

# -----------------------------
# 5) Utility: energy needed to hit a target top size
def Ecs_for_target_xmax(x_target, lam, BASE=25.0):
    def H(Ecs):
        E = Ecs * (x_target / BASE) ** lam
        return xmax_from_E(E) - x_target
    lo, hi = 1e-6, 1e6
    # bracket
    hlo, hhi = H(lo), H(hi)
    expand = 0
    while hlo * hhi > 0 and hi <= 1e30:
        lo /= 10.0
        hi *= 10.0
        hlo, hhi = H(lo), H(hi)
        expand += 1
        if expand > 200:
            raise RuntimeError("Could not bracket root; check coefficients/units.")
    for _ in range(80):
        mid = math.sqrt(lo * hi) # bisection in log-space
        hmid = H(mid)
        if abs(hmid) < 1e-12:
            return mid
        if hlo * hmid < 0:
            hi, hhi = mid, hmid
        else:
            lo, hlo = mid, hmid
    return math.sqrt(lo * hi)

# -----------------------------
# 6) Generate & write x_max vs E_cs table
if __name__ == "__main__":
    # PARAMETERS (edit these):
    lam = 0.5142 # fragmentation-size coupling exponent
    BASE = 25.0 # same units as x (e.g., mm); must match fan-fit units
    npts = 101 # how many samples along E_cs
    Ecs_min, Ecs_max = 1e+0, 1e+5 # sweep range for E_cs

    Ecs = 25
    xtarget = 0.1
    xmax = xmax_from_Ecs(Ecs, lam, BASE=25.0)
    Ecs_star = Ecs_for_target_xmax(xtarget, lam, BASE=25.0)
    print(f"xmax for Ecs = {Ecs} kWh/t:", xmax)
    print(f"Ecs for xmax = {xtarget} mm:", Ecs_star)


    Ecs_grid = np.logspace(np.log10(Ecs_min), np.log10(Ecs_max), npts)
    xmax_vals = [xmax_from_Ecs(Ecs, lam=lam, BASE=BASE) for Ecs in Ecs_grid]

    # Write CSV
    out_csv = "xmax_vs_Ecs.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["E_cs", "x_max"])
        for Ecs, x in zip(Ecs_grid, xmax_vals):
            writer.writerow([f"{Ecs:.12g}", f"{x:.12g}"])

    # # Write CSV
    # out_csv = "xmax_vs_Ecs_J_kg.csv"
    # with open(out_csv, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["E_cs", "x_max"])
    #     for Ecs, x in zip(Ecs_grid, xmax_vals):
    #         writer.writerow([f"{Ecs/(0.000000277778*1000):.12g}", f"{x:.12g}"])

    print(f"Wrote {out_csv} with {len(Ecs_grid)} rows.")

    # Quick sanity print
    for i in [0, len(Ecs_grid)//2, -1]:
        print(f"E_cs={Ecs_grid[i]:.4g} -> x_max={xmax_vals[i]:.6g}")



    '''
    Andy's section

    Simple assumption:
    20% of particles between 1E5 and 2E6 are pulverized to the desired extent
    '''

    # Ecs = 316.2 # kWh/ton
    # x_max = 0.124376 # at Ecs above


    # Ecs = 1E5 # kWh/ton
    # x_max = 0.0139567 # at Ecs above



    Ecs = 20 # kWh/ton
    x_max = 0.35 # at Ecs above
    # x_20 = 0.006227811194633059 mm




    # melting point (2 MJ/kg)
    # Ecs = 555
    # x_max = 0.1
    # x_80 = 0.003109 mm






    x_80 = x_max*C80*(Ecs*(x_max/25)**lam)**(-a80)
    x_20 = x_max*C20*(Ecs*(x_max/25)**lam)**(-a20)

    Ecs_J_kg = 316.2*3.60E+03 # J/kg

    print(f'x_20={x_20:.4g}')
    print(f'x_80={x_80:.4g}')
    print(f'Ecs_J/kg={Ecs_J_kg:.4g}')

    # Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(Ecs_grid, xmax_vals)
        plt.xlabel(r"$E_\text{cs}$ (kWh/t)")
        plt.ylabel(r"$x_{\max}$ (mm)")
        plt.xlim(1, 1e5)
        plt.ylim(0.01, 1)
        plt.title(r"$x_{\max}$ vs $E_\text{cs}$")
        plt.grid(True, which="both", ls=":")
        plt.show()
    except Exception as e:
        print("Error:", e)
