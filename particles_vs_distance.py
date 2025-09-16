from swebrec_combined import *

















if __name__ == "__main__":
    Ecs, lam, BASE = 20.0, 0.5142, 25.0
    x_max, A, b, x20_f, x50_f, x80_f, E_fan = swebrec_through_fan(Ecs, lam, BASE)
    x_min = min(x20_f, x50_f, x80_f) * 1e-3
    x_vals = np.logspace(np.log10(x_min), np.log10(0.999*x_max), 600)
    P_vals = [P_of_x_fan_consistent(x, Ecs, lam, BASE) for x in x_vals]

    print(P_of_x_fan_consistent(6.1e-3, Ecs, lam))