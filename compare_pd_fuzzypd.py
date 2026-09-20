"""
Compares the Fuzzy-PD controller against the underlying fixed-gain PD
controller (obtained by disabling the fuzzy correction, i.e. forcing
Delta K_p = Delta K_d = 0 so only the baseline gains K_p0, K_d0 act) on the
same reference trajectory and disturbance used in quadrotor_fuzzy_pd.py.

This produces the PD vs. Fuzzy-PD comparison (trajectory, tracking error,
RMSE/MAE) referenced in the project report but not covered by
quadrotor_fuzzy_pd.py alone (which only simulates the Fuzzy-PD case).

Run:
    python compare_pd_fuzzypd.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from fis_reader import evalfis, read_fis

HERE = os.path.dirname(os.path.abspath(__file__))


def sat(x: float, a: float, b: float) -> float:
    return min(max(x, a), b)


def run_simulation(use_fuzzy: bool):
    """Simulate the closed-loop quadrotor.

    use_fuzzy=True  -> full Fuzzy-PD controller (Delta K_p, Delta K_d from FIS)
    use_fuzzy=False -> fixed-gain PD controller only (Delta K_p = Delta K_d = 0)
    """
    m = 1.7
    g = 9.81
    Ix = 4e-3
    Iy = 4e-3
    Iz = 8.4e-3
    Jx = 5.7e-4
    Jy = 5.7e-4
    Jz = 5.7e-4
    tau_diff = 0.01

    x0 = np.zeros(14)
    x0[2] = 1.0
    t0, Tf = 0.0, 60.0

    fis1 = read_fis(os.path.join(HERE, "delKp.fis"))
    fis2 = read_fis(os.path.join(HERE, "delKd.fis"))

    Kp_p = np.array([0.1, 0.1, 0.1])
    Kd_p = np.array([1 / 3, 1 / 3, 1 / 3])
    Kp_a = np.array([0.1, 0.1, 0.1])
    Kd_a = np.array([1 / 3, 1 / 3, 1 / 3])

    def f(t, X):
        xd = np.cos(np.pi * t / 20)
        yd = np.sin(np.pi * t / 20)
        zd = 2.0
        psi_d = 0.0
        psi_d_dot = 0.0

        xd_dot = -(np.pi / 20) * np.sin(np.pi * t / 20)
        yd_dot = (np.pi / 20) * np.cos(np.pi * t / 20)
        zd_dot = 0.0

        x, y, z = X[0], X[1], X[2]
        x_dot, y_dot, z_dot = X[3], X[4], X[5]
        phi, theta, psi = X[6], X[7], X[8]
        phi_dot, theta_dot, psi_dot = X[9], X[10], X[11]
        phi_d_prev = X[12]
        theta_d_prev = X[13]

        ex = sat(xd - x, -5, 5)
        ey = sat(yd - y, -5, 5)
        ez = sat(zd - z, -5, 5)
        ex_dot = sat(xd_dot - x_dot, -5, 5)
        ey_dot = sat(yd_dot - y_dot, -5, 5)
        ez_dot = sat(zd_dot - z_dot, -5, 5)

        if use_fuzzy:
            delKpx = evalfis(fis1, [ex, ex_dot])[0]
            delKdx = evalfis(fis2, [ex, ex_dot])[0]
            delKpy = evalfis(fis1, [ey, ey_dot])[0]
            delKdy = evalfis(fis2, [ey, ey_dot])[0]
            delKpz = evalfis(fis1, [ez, ez_dot])[0]
            delKdz = evalfis(fis2, [ez, ez_dot])[0]
        else:
            delKpx = delKdx = delKpy = delKdy = delKpz = delKdz = 0.0

        ux = (Kp_p[0] + delKpx) * ex + (Kd_p[0] + delKdx) * ex_dot
        uy = (Kp_p[1] + delKpy) * ey + (Kd_p[1] + delKdy) * ey_dot
        uz = (Kp_p[2] + delKpz) * ez + (Kd_p[2] + delKdz) * ez_dot

        T = m * np.sqrt(ux**2 + uy**2 + (uz + g) ** 2)
        phi_d = np.arcsin((ux * np.sin(psi_d) - uy * np.cos(psi_d)) / np.sqrt(ux**2 + uy**2 + (uz + g) ** 2))
        theta_d = np.arctan((ux * np.cos(psi_d) + uy * np.sin(psi_d)) / (uz + g))

        phi_d_dot = (phi_d - phi_d_prev) / tau_diff
        theta_d_dot = (theta_d - theta_d_prev) / tau_diff

        e_phi = phi_d - phi
        e_theta = theta_d - theta
        e_psi = psi_d - psi
        e_phi_dot = phi_d_dot - phi_dot
        e_theta_dot = theta_d_dot - theta_dot
        e_psi_dot = psi_d_dot - psi_dot

        tau_phi = Kp_a[0] * e_phi + Kd_a[0] * e_phi_dot
        tau_theta = Kp_a[1] * e_theta + Kd_a[1] * e_theta_dot
        tau_psi = Kp_a[2] * e_psi + Kd_a[2] * e_psi_dot

        dis_x = 0.05 * np.sin(t) * np.cos(0.5 * t) ** 2
        dis_y = 0.05 * np.sin(t) * np.cos(0.5 * t) ** 2
        dis_z = 0.05 * np.sin(t) * np.cos(0.5 * t) ** 2
        dis_phi = 0.02 * np.sin(0.7 * t)
        dis_theta = 0.02 * np.sin(0.7 * t)
        dis_psi = 0.02 * np.sin(0.7 * t)

        phi_dot2 = theta_dot * psi_dot * (Iy - Iz) / Ix + tau_phi / Ix + dis_phi
        theta_dot2 = phi_dot * psi_dot * (Iz - Ix) / Iy + tau_theta / Iy + dis_theta
        psi_dot2 = phi_dot * theta_dot * (Ix - Iy) / Iz + tau_psi / Iz + dis_psi

        x_dot2 = (T / m) * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) - (Jx / m) * x_dot + dis_x
        y_dot2 = (T / m) * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) - (Jy / m) * y_dot + dis_y
        z_dot2 = (T / m) * np.cos(phi) * np.cos(theta) - g - (Jz / m) * z_dot + dis_z

        deq = np.zeros(14)
        deq[0], deq[1], deq[2] = x_dot, y_dot, z_dot
        deq[3], deq[4], deq[5] = x_dot2, y_dot2, z_dot2
        deq[6], deq[7], deq[8] = phi_dot, theta_dot, psi_dot
        deq[9], deq[10], deq[11] = phi_dot2, theta_dot2, psi_dot2
        deq[12], deq[13] = phi_d_dot, theta_d_dot
        return deq

    sol = solve_ivp(f, (t0, Tf), x0, method="RK45", rtol=1e-3, atol=1e-5, max_step=0.02)
    return sol.t, sol.y.T


def rmse(e):
    return float(np.sqrt(np.mean(e**2)))


def mae(e):
    return float(np.mean(np.abs(e)))


def main():
    t_f, X_f = run_simulation(use_fuzzy=True)
    t_p, X_p = run_simulation(use_fuzzy=False)

    def desired(t):
        xd = np.cos(np.pi * t / 20)
        yd = np.sin(np.pi * t / 20)
        zd = 2 + 0 * t
        return xd, yd, zd

    xd_f, yd_f, zd_f = desired(t_f)
    xd_p, yd_p, zd_p = desired(t_p)

    ex_f, ey_f, ez_f = xd_f - X_f[:, 0], yd_f - X_f[:, 1], zd_f - X_f[:, 2]
    ex_p, ey_p, ez_p = xd_p - X_p[:, 0], yd_p - X_p[:, 1], zd_p - X_p[:, 2]

    metrics = {
        "Fuzzy PD": {
            "RMSE(ex)": rmse(ex_f), "MAE(ex)": mae(ex_f),
            "RMSE(ey)": rmse(ey_f), "MAE(ey)": mae(ey_f),
            "RMSE(ez)": rmse(ez_f), "MAE(ez)": mae(ez_f),
        },
        "PD": {
            "RMSE(ex)": rmse(ex_p), "MAE(ex)": mae(ex_p),
            "RMSE(ey)": rmse(ey_p), "MAE(ey)": mae(ey_p),
            "RMSE(ez)": rmse(ez_p), "MAE(ez)": mae(ez_p),
        },
    }

    out_dir = os.path.join(HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Table (CSV + printed) ----
    csv_path = os.path.join(out_dir, "pd_vs_fuzzypd_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("Controller,RMSE(ex),MAE(ex),RMSE(ey),MAE(ey),RMSE(ez),MAE(ez)\n")
        for name, m_ in metrics.items():
            fp.write(
                f"{name},{m_['RMSE(ex)']:.4f},{m_['MAE(ex)']:.4f},"
                f"{m_['RMSE(ey)']:.4f},{m_['MAE(ey)']:.4f},"
                f"{m_['RMSE(ez)']:.4f},{m_['MAE(ez)']:.4f}\n"
            )
    print(f"Saved metrics table to {csv_path}")
    for name, m_ in metrics.items():
        print(name, m_)

    # ---- Figure: 3D trajectory comparison ----
    fig1 = plt.figure("3D Trajectory Comparison")
    ax = fig1.add_subplot(projection="3d")
    ax.plot(X_f[:, 0], X_f[:, 1], X_f[:, 2], "r", linewidth=1.5, label="Fuzzy PD")
    ax.plot(X_p[:, 0], X_p[:, 1], X_p[:, 2], "b--", linewidth=1.5, label="PD")
    ax.plot(xd_f, yd_f, zd_f, "k:", linewidth=2, label="Desired")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Trajectory comparison: Fuzzy PD vs. PD")
    ax.legend()
    fig1.savefig(os.path.join(out_dir, "compare_3d_trajectory.png"), dpi=150)

    # ---- Figure: tracking error comparison ----
    fig2, axs = plt.subplots(3, 1, num="Tracking error comparison", sharex=True, figsize=(7, 8))
    axs[0].plot(t_f, ex_f, "r", label="Fuzzy PD")
    axs[0].plot(t_p, ex_p, "b--", label="PD")
    axs[0].set_ylabel("$e_x$ (m)")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("Tracking error: Fuzzy PD vs. PD")
    axs[1].plot(t_f, ey_f, "r", label="Fuzzy PD")
    axs[1].plot(t_p, ey_p, "b--", label="PD")
    axs[1].set_ylabel("$e_y$ (m)")
    axs[1].legend()
    axs[1].grid(True)
    axs[2].plot(t_f, ez_f, "r", label="Fuzzy PD")
    axs[2].plot(t_p, ez_p, "b--", label="PD")
    axs[2].set_ylabel("$e_z$ (m)")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend()
    axs[2].grid(True)
    fig2.savefig(os.path.join(out_dir, "compare_tracking_error.png"), dpi=150)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
