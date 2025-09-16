import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
b = 1.83
C50 = 0.197
alpha50 = 0.0817
lam = 0.5142

E_cs = 20 # <-- set your value here


# -----------------------------
# Corrected x_max
def x_max(E):
    return 1.1094 * E**(-0.380)

def x_50(E):
    return C50 * E**(-alpha50)
    # or, if you want the second form:
    # return C50 * (E/25)**lam * x_max(E)**(-alpha50)

def P(x, E):
    xmax = x_max(E)
    x50 = x_50(E)
    return 1.0 / (1.0 + (np.log(xmax/x) / np.log(xmax/x50))**b)

# -----------------------------
x_vals = np.linspace(1e-3, 0.999 * x_max(E_cs), 500)
P_vals = P(x_vals, E_cs)

print(P(0.006, E_cs))

plt.figure(figsize=(6,4))
plt.plot(x_vals, P_vals, lw=2)
plt.xlabel(r"$x$ (mm)")
plt.ylabel(r"$P(x)$")
plt.title(rf"$P(x)$ vs $x$ for $E_\text{{cs}}$={E_cs} kWh/t")
plt.grid(True)
plt.tight_layout()
plt.show()
