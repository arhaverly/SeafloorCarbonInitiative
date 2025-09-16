# https://chatgpt.com/share/68c2a588-79d8-8006-a1ad-42bc93071e5c




# Fit a smooth trendline through (r, E) with a physically-plausible model:
#   E(r) = A * r^{-n} * exp(-alpha * r)
# which captures geometric spreading (n) and intrinsic attenuation (alpha).
#
# Then plot the curve + anchor points.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


from swebrec_combined import *
lam = 0.5142



W = 100000 # [kt]
r_melt = 17*W**(1/3)
r_crush = 3*r_melt
r_rupture = 9*r_melt


# Given data
r = np.array([r_melt, r_crush, r_rupture])        # distance
# E = np.array([2e6, 6.8e4, 3.6e1])               # implied energy (units arbitrary)

# https://www.mdpi.com/2075-163X/13/8/1059
# new value says 4E3 for crushing
E = np.array([2e6, 4e3, 3.6e1])               # implied energy (units arbitrary)







# Solve exactly for (A, n, alpha) using log-linear system:
# log E = log A - n * log r - alpha * r
X = np.column_stack([np.ones_like(r), -np.log(r), -r])
y = np.log(E)
logA, n, alpha = np.linalg.solve(X, y)
A = float(np.exp(logA))

# Model function
def E_model(r_):
    r_ = np.asarray(r_, dtype=float)
    return A * r_**(-n) * np.exp(-alpha * r_)

# Optional: evaluate some interpolate points
r_plot = np.linspace(100, r_rupture*1.1, 400)
E_plot = E_model(r_plot)

# Print the fitted parameters for reference
print(f"Fitted model: E(r) = A * r^(-n) * exp(-alpha * r)")
print(f"A = {A:.6g}")
print(f"n = {n:.6g}")
print(f"alpha = {alpha:.6g}  (1/m)")

# Plot
plt.figure(figsize=(7,5))
plt.plot(r_plot, E_plot, label="Fit: A r^{-n} e^{-α r}")
plt.scatter(r, E, marker='o', label="Data points")


labels = ['Melt Zone Radius', 'Crush Zone Radius', 'Rupture Zone Radius']
for i, (x, y) in enumerate(zip(r, E)):
    plt.text(x+250, y*1.7, labels[i], ha='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2')
    )  # 1.1 shifts the text upward



plt.axvline(x=r_melt, color='r', linestyle='--', label='')
plt.axvline(x=1300, color='r', linestyle='--', label='')





# plt.axhline(y=2e6, color='r', linestyle='--', label='')
# plt.axhline(y=8e4, color='r', linestyle='--', label='')

plt.yscale('log')
plt.xlabel("Distance (m)")
plt.ylabel("Energy (J/kg)")
plt.title("Energy vs. Distance")
# plt.legend()
plt.tight_layout()

plt.savefig('energy_vs_distance.png', dpi=300)









print('==================================')

# for W in [1000, 10000, 100000]:
for W in [100000]: # 100 Mt
    r_melt = 17*W**(1/3)
    r_crush = 3*r_melt
    r_rupture = 9*r_melt
    crush_zone_range = r_crush-r_melt
    crush_zone_volume = 4/3*3.14159*(r_crush**3) - 4/3*3.14159*(r_melt**3)
    crush_zone_mass = crush_zone_volume*3

    r_desired_particles = 1300
    percent_taken = 0.2
    desired_particles_volume = 4/3*3.14159*(r_desired_particles**3) - 4/3*3.14159*(r_melt**3)
    desired_particles_mass = desired_particles_volume*3*percent_taken # [t]

    print(r_melt)
    print(f'{crush_zone_range:e}')
    print(f'{crush_zone_volume:e}')
    print(f'{crush_zone_mass:e}')
    # print(f'{4/3*3.14159*(r_melt**3):e}')
    print(f'{E_model(r_desired_particles):e}')

    print(f'{desired_particles_volume:e}')
    print(f'{desired_particles_mass:e}')
    print()

    # Integration limits
    r_min, r_max = r_melt, r_desired_particles

    # Numerical integration
    area, error = quad(E_model, r_min, r_max)
    print(f'{area:e}')





print('==================================')


desired_particle_size = 6.1e-3 # [mm]
desired_particle_size = 50e-3
# desired_particle_size = 50e-3




step_size = 1
rock_density = 3 # [t]
total_mass_of_desired = 0
for r in range(int(r_melt), int(r_crush), step_size):
    # volume of this step
    volume_shell = 4/3*3.14159*(r**3) - 4/3*3.14159*((r-1)**3)
    # print(volume_shell)

    # energy at this step
    Ecs = E_model(r)

    # percent of mass is the desired particle size
    # taken from the swebrec
    percent_of_mass_desired = P_of_x_fan_consistent(desired_particle_size, Ecs/3.6e3, lam)
    print(r)
    print(percent_of_mass_desired)

    # mass of desired
    mass_of_desired = volume_shell*rock_density*percent_of_mass_desired

    total_mass_of_desired += mass_of_desired


print(f'Total Mass of Particles at the Desired Size = {total_mass_of_desired:.4g} tons')
print(f'Total Mass of Particles compared to mass required = {total_mass_of_desired/6E10*100:.4g}%')


















plt.show()




