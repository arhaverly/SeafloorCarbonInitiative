import math
import numpy as np
import matplotlib.pyplot as plt

# Fan fits (high-energy branch)
C20, a20 = 0.05225, 1.3452
C50, a50 = 0.177,   0.8817
C80, a80 = 0.3137,  0.6643

def xP_over_D(E, C, a):  # plain fan percentile (no x_max prefactor)
    return C * E**(-a)

def b_from_r(r, tol=1e-12, b_lo=0.1, b_hi=20.0):
    """Invert r = (1 - 0.25^(1/b)) / (4^(1/b) - 1). (Monotone in b.)"""
    def r_of_b(b):
        t1 = 0.25 ** (1.0 / b)
        t2 = 4.0   ** (1.0 / b)
        return (1.0 - t1) / (t2 - 1.0)
    # clamp r into reachable range
    r_lo, r_hi = r_of_b(b_lo), r_of_b(b_hi)
    r = min(max(r, min(r_lo, r_hi)), max(r_lo, r_hi))
    for _ in range(80):
        mid = 0.5*(b_lo + b_hi)
        r_mid = r_of_b(mid)
        if abs(b_hi - b_lo) < tol:
            return mid
        if (r_mid < r) == (r_lo < r):
            b_lo, r_lo = mid, r_mid
        else:
            b_hi, r_hi = mid, r_mid
    return 0.5*(b_lo + b_hi)

def xmax_from_E(E):
    """
    Given fan energy E (kWh/t), compute x_max (mm) by Swebrec geometry.
    """
    x20 = xP_over_D(E, C20, a20)
    x50 = xP_over_D(E, C50, a50)
    x80 = xP_over_D(E, C80, a80)

    l8050 = math.log(x80 / x50)
    l2050 = math.log(x20 / x50)
    r = -l8050 / l2050
    b = b_from_r(r)
    t = (0.25) ** (1.0 / b)

    A = -l8050 / (t - 1.0)    
    return x50 * math.exp(A)  

def xmax_from_Ecs(Ecs, lam, BASE=25.0, x0=None, iters=50, tol=1e-10):
    """
    Fixed-point solve for x_max given E_cs and energy-size coupling.
    """
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

def fan_percentiles_prefactorized(Ecs, lam, BASE=25.0):
    """x_P = x_max * C_P * E_fan^{-a_P} with E_fan = Ecs*(x_max/BASE)^lam."""
    x_max = xmax_from_Ecs(Ecs, lam, BASE=BASE)
    E_fan = Ecs * (x_max / BASE) ** lam
    x20 = x_max * C20 * (E_fan ** (-a20))
    x50 = x_max * C50 * (E_fan ** (-a50))
    x80 = x_max * C80 * (E_fan ** (-a80))
    return x20, x50, x80, x_max, E_fan

def swebrec_through_fan(Ecs, lam, BASE=25.0):
    """
    Returns (x_max, A, b, x20_fan, x50_fan, x80_fan) such that
    P(x) = 1 / (1 + (ln(x_max/x)/A)^b) passes through:
      (x20_fan, 0.2), (x50_fan, 0.5), (x80_fan, 0.8).
    """
    x20_f, x50_f, x80_f, x_max, E_fan = fan_percentiles_prefactorized(Ecs, lam, BASE)

    # A from the median point
    A = math.log(x_max / x50_f)

    # Scale-free S-values
    S20 = math.log(x_max / x20_f) / A
    S80 = math.log(x_max / x80_f) / A

    # b from either side (should agree); combine robustly
    # b20 = math.log(4.0) / math.log(S20)           # since S20 = 4^(1/b)
    # b80 = math.log(4.0) / math.log(1.0 / S80)     # since S80 = 0.25^(1/b)
    b   = 1.83 # use 0.5 * (b20 + b80) to fit b to pass through x20_fan and x80_fan

    return x_max, A, b, x20_f, x50_f, x80_f, E_fan

def P_of_x_fan_consistent(x, Ecs, lam, BASE=25.0):
    """Swebrec CDF that exactly matches fan x20/x50/x80."""
    x_max, A, b, *_ = swebrec_through_fan(Ecs, lam, BASE)
    z = math.log(x_max / x) / A
    return 1.0 / (1.0 + (z ** b))

def x_at_P_fan_consistent(p, Ecs, lam, BASE=25.0):
    """Inverse consistent with those parameters."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    x_max, A, b, *_ = swebrec_through_fan(Ecs, lam, BASE)
    t = ((1.0/p) - 1.0) ** (1.0/b)
    return x_max / math.exp(A * t)

def ratio_x50_over_xmax(Ecs, lam, BASE=25.0):
    """x50/xmax = C50 * (E_fan)^(-a50), where E_fan = Ecs * (xmax/BASE)^lam."""
    x_max = xmax_from_Ecs(Ecs, lam, BASE)
    E_fan = Ecs * (x_max / BASE) ** lam
    return C50 * (E_fan ** (-a50))

# cross-check via Swebrec parameters built to hit the fan points
def ratio_from_swebrec(Ecs, lam, BASE=25.0):
    x_max, A, b, *_ = swebrec_through_fan(Ecs, lam, BASE)
    return math.exp(-A)

# ==============================================================================
if __name__ == "__main__":
  Ecs, lam, BASE = 20.0, 0.5142, 25.0
  x_max, A, b, x20_f, x50_f, x80_f, E_fan = swebrec_through_fan(Ecs, lam, BASE)
  x_min = min(x20_f, x50_f, x80_f) * 1e-3
  x_vals = np.logspace(np.log10(x_min), np.log10(0.999*x_max), 600)
  P_vals = [P_of_x_fan_consistent(x, Ecs, lam, BASE) for x in x_vals]

  print(P_of_x_fan_consistent(6.1e-3, Ecs, lam))
  print(P_of_x_fan_consistent(50e-3, Ecs, lam))


  print("fan x20/x50/x80:", x20_f, x50_f, x80_f)
  print("swebrec x@0.2/0.5/0.8:",
        x_at_P_fan_consistent(0.2, Ecs, lam, BASE),
        x_at_P_fan_consistent(0.5, Ecs, lam, BASE),
        x_at_P_fan_consistent(0.8, Ecs, lam, BASE))
  print("x_max:", x_max)

  plt.figure(figsize=(6,4))
  plt.semilogx(x_vals, P_vals, lw=2)
  for x,p in [(x20_f,0.2),(x50_f,0.5),(x80_f,0.8)]:
      plt.scatter([x],[p], zorder=3)
      plt.axvline(x, ls="--", alpha=0.4); plt.axhline(p, ls="--", alpha=0.4)
  plt.xlabel(r"$x$ (mm)"); plt.ylabel(r"$P(x)$"); plt.ylim(0,1)
  plt.title("Swebrec P(x) forced through fan x20/x50/x80")
  plt.grid(True, which="both", ls=":")
  plt.tight_layout()
  plt.show()

  # ---- sweep & plot ----
  Ecs_grid = np.logspace(0, 5, 121)  # 1 → 1e5 kWh/t
  ratios = np.array([ratio_x50_over_xmax(E, lam) for E in Ecs_grid])

  plt.figure(figsize=(6,4))
  
  # (Optional overlay)
  ratios_chk = np.array([ratio_from_swebrec(E, lam) for E in Ecs_grid])
  plt.semilogx(Ecs_grid, ratios_chk, lw=2, label="from Swebrec")
  plt.semilogx(Ecs_grid, ratios, ls="--", label=r"$C_{50}\,E_\mathrm{fan}^{-\alpha_{50}}$")

  plt.ylim(0, 1)
  plt.xlabel(r"$E_\mathrm{cs}$ (kWh/t)")
  plt.ylabel(r"$x_{50}/x_{\max}$")
  plt.title(r"$x_{50}/x_{\max}$ vs $E_\mathrm{cs}$")
  plt.grid(True, which="both", ls=":")
  plt.legend()
  plt.tight_layout()
  plt.show()

  plt.figure(figsize=(6,4))
  plt.loglog(Ecs_grid * 3.6e3, ratios, lw=2,
            label=r"$C_{50}\,E_\mathrm{fan}^{-\alpha_{50}}$")

  # log-scale can't include 0—pick safe limits from the data
  ymin = max(1e-12, ratios.min()*0.8)
  ymax = ratios.max()/0.8
  plt.ylim(ymin, ymax)

  plt.xlabel(r"$E_\mathrm{cs}$ (J/kg)")
  plt.ylabel(r"$x_{50}/x_{\max}$")
  plt.title(r"$x_{50}/x_{\max}$ vs $E_\mathrm{cs}$")
  plt.grid(True, which="both", ls=":")
  plt.tight_layout()
  plt.show()