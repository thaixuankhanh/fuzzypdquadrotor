"""
Generates the full set of simulation results used in the project report
(Chuong 4): position tracking, attitude tracking, tracking-error comparison,
online gain adaptation (Delta K_p, Delta K_d), 3D trajectories, and a
robustness scenario with amplified disturbance -- for both the Fuzzy-PD
controller and the underlying fixed-gain PD controller.

This supersedes compare_pd_fuzzypd.py (kept for backward compatibility) by
also logging the intermediate signals (errors, online gains, attitude
references) needed to plot the additional figures, and by adding a
second, higher-disturbance scenario for a robustness comparison.

Run:
    python full_report_simulations.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from fis_reader import evalfis, read_fis

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

M = 1.7
G = 9.81
IX = 4e-3
IY = 4e-3
IZ = 8.4e-3
JX = 5.7e-4
JY = 5.7e-4
JZ = 5.7e-4
TAU_DIFF = 0.01

FIS1 = read_fis(os.path.join(HERE, "delKp.fis"))
FIS2 = read_fis(os.path.join(HERE, "delKd.fis"))

KP_P = np.array([0.1, 0.1, 0.1])
KD_P = np.array([1 / 3, 1 / 3, 1 / 3])
KP_A = np.array([0.1, 0.1, 0.1])
KD_A = np.array([1 / 3, 1 / 3, 1 / 3])


def sat(x: float, a: float, b: float) -> float:
    return min(max(x, a), b)


def desired_trajectory(t):
    xd = np.cos(np.pi * t / 20)
    yd = np.sin(np.pi * t / 20)
    zd = 2.0 + 0.0 * t
    return xd, yd, zd


def make_rhs(use_fuzzy: bool, dist_scale: float):
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
            delKpx = evalfis(FIS1, [ex, ex_dot])[0]
            delKdx = evalfis(FIS2, [ex, ex_dot])[0]
            delKpy = evalfis(FIS1, [ey, ey_dot])[0]
            delKdy = evalfis(FIS2, [ey, ey_dot])[0]
            delKpz = evalfis(FIS1, [ez, ez_dot])[0]
            delKdz = evalfis(FIS2, [ez, ez_dot])[0]
        else:
            delKpx = delKdx = delKpy = delKdy = delKpz = delKdz = 0.0

        ux = (KP_P[0] + delKpx) * ex + (KD_P[0] + delKdx) * ex_dot
        uy = (KP_P[1] + delKpy) * ey + (KD_P[1] + delKdy) * ey_dot
        uz = (KP_P[2] + delKpz) * ez + (KD_P[2] + delKdz) * ez_dot

        T = M * np.sqrt(ux**2 + uy**2 + (uz + G) ** 2)
        phi_d = np.arcsin((ux * np.sin(psi_d) - uy * np.cos(psi_d)) / np.sqrt(ux**2 + uy**2 + (uz + G) ** 2))
        theta_d = np.arctan((ux * np.cos(psi_d) + uy * np.sin(psi_d)) / (uz + G))

        phi_d_dot = (phi_d - phi_d_prev) / TAU_DIFF
        theta_d_dot = (theta_d - theta_d_prev) / TAU_DIFF

        e_phi = phi_d - phi
        e_theta = theta_d - theta
        e_psi = psi_d - psi
        e_phi_dot = phi_d_dot - phi_dot
        e_theta_dot = theta_d_dot - theta_dot
        e_psi_dot = psi_d_dot - psi_dot

        tau_phi = KP_A[0] * e_phi + KD_A[0] * e_phi_dot
        tau_theta = KP_A[1] * e_theta + KD_A[1] * e_theta_dot
        tau_psi = KP_A[2] * e_psi + KD_A[2] * e_psi_dot

        dis_x = dist_scale * 0.05 * np.sin(t) * np.cos(0.5 * t) ** 2
        dis_y = dist_scale * 0.05 * np.sin(t) * np.cos(0.5 * t) ** 2
        dis_z = dist_scale * 0.05 * np.sin(t) * np.cos(0.5 * t) ** 2
        dis_phi = dist_scale * 0.02 * np.sin(0.7 * t)
        dis_theta = dist_scale * 0.02 * np.sin(0.7 * t)
        dis_psi = dist_scale * 0.02 * np.sin(0.7 * t)

        phi_dot2 = theta_dot * psi_dot * (IY - IZ) / IX + tau_phi / IX + dis_phi
        theta_dot2 = phi_dot * psi_dot * (IZ - IX) / IY + tau_theta / IY + dis_theta
        psi_dot2 = phi_dot * theta_dot * (IX - IY) / IZ + tau_psi / IZ + dis_psi

        x_dot2 = (T / M) * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) - (JX / M) * x_dot + dis_x
        y_dot2 = (T / M) * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) - (JY / M) * y_dot + dis_y
        z_dot2 = (T / M) * np.cos(phi) * np.cos(theta) - G - (JZ / M) * z_dot + dis_z

        deq = np.zeros(14)
        deq[0], deq[1], deq[2] = x_dot, y_dot, z_dot
        deq[3], deq[4], deq[5] = x_dot2, y_dot2, z_dot2
        deq[6], deq[7], deq[8] = phi_dot, theta_dot, psi_dot
        deq[9], deq[10], deq[11] = phi_dot2, theta_dot2, psi_dot2
        deq[12], deq[13] = phi_d_dot, theta_d_dot
        return deq

    return f


def run_simulation(use_fuzzy: bool, dist_scale: float = 1.0):
    x0 = np.zeros(14)
    x0[2] = 1.0
    sol = solve_ivp(
        make_rhs(use_fuzzy, dist_scale),
        (0.0, 60.0),
        x0,
        method="RK45",
        rtol=1e-3,
        atol=1e-5,
        max_step=0.02,
    )
    return sol.t, sol.y.T


def postprocess(t, X, use_fuzzy: bool):
    """Recompute the intermediate control signals (errors, online gains,
    attitude references) at each returned time sample, for plotting."""
    n = len(t)
    ex = np.zeros(n)
    ey = np.zeros(n)
    ez = np.zeros(n)
    dKpx = np.zeros(n)
    dKdx = np.zeros(n)
    dKpy = np.zeros(n)
    dKdy = np.zeros(n)
    dKpz = np.zeros(n)
    dKdz = np.zeros(n)
    phi_d = np.zeros(n)
    theta_d = np.zeros(n)

    xd, yd, zd = desired_trajectory(t)
    xd_dot = -(np.pi / 20) * np.sin(np.pi * t / 20)
    yd_dot = (np.pi / 20) * np.cos(np.pi * t / 20)
    zd_dot = np.zeros(n)

    for i in range(n):
        x, y, z = X[i, 0], X[i, 1], X[i, 2]
        x_dot, y_dot, z_dot = X[i, 3], X[i, 4], X[i, 5]
        ex[i] = sat(xd[i] - x, -5, 5)
        ey[i] = sat(yd[i] - y, -5, 5)
        ez[i] = sat(zd[i] - z, -5, 5)
        ex_dot = sat(xd_dot[i] - x_dot, -5, 5)
        ey_dot = sat(yd_dot[i] - y_dot, -5, 5)
        ez_dot = sat(zd_dot[i] - z_dot, -5, 5)

        if use_fuzzy:
            dKpx[i] = evalfis(FIS1, [ex[i], ex_dot])[0]
            dKdx[i] = evalfis(FIS2, [ex[i], ex_dot])[0]
            dKpy[i] = evalfis(FIS1, [ey[i], ey_dot])[0]
            dKdy[i] = evalfis(FIS2, [ey[i], ey_dot])[0]
            dKpz[i] = evalfis(FIS1, [ez[i], ez_dot])[0]
            dKdz[i] = evalfis(FIS2, [ez[i], ez_dot])[0]

        ux = (KP_P[0] + dKpx[i]) * ex[i] + (KD_P[0] + dKdx[i]) * ex_dot
        uy = (KP_P[1] + dKpy[i]) * ey[i] + (KD_P[1] + dKdy[i]) * ey_dot
        uz = (KP_P[2] + dKpz[i]) * ez[i] + (KD_P[2] + dKdz[i]) * ez_dot
        phi_d[i] = np.arcsin((ux * 0 - uy * 1) / np.sqrt(ux**2 + uy**2 + (uz + G) ** 2))
        theta_d[i] = np.arctan((ux * 1 + uy * 0) / (uz + G))

    return {
        "xd": xd, "yd": yd, "zd": zd,
        "ex": ex, "ey": ey, "ez": ez,
        "dKpx": dKpx, "dKdx": dKdx,
        "dKpy": dKpy, "dKdy": dKdy,
        "dKpz": dKpz, "dKdz": dKdz,
        "phi_d": phi_d, "theta_d": theta_d,
    }


def rmse(e):
    return float(np.sqrt(np.mean(e**2)))


def mae(e):
    return float(np.mean(np.abs(e)))


def save_metrics_csv(path, rows):
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("Controller,RMSE(ex),MAE(ex),RMSE(ey),MAE(ey),RMSE(ez),MAE(ez)\n")
        for name, m in rows.items():
            fp.write(
                f"{name},{m['RMSE(ex)']:.4f},{m['MAE(ex)']:.4f},"
                f"{m['RMSE(ey)']:.4f},{m['MAE(ey)']:.4f},"
                f"{m['RMSE(ez)']:.4f},{m['MAE(ez)']:.4f}\n"
            )


def metrics_of(errs):
    ex, ey, ez = errs
    return {
        "RMSE(ex)": rmse(ex), "MAE(ex)": mae(ex),
        "RMSE(ey)": rmse(ey), "MAE(ey)": mae(ey),
        "RMSE(ez)": rmse(ez), "MAE(ez)": mae(ez),
    }


def main():
    # ------------------------------------------------------------------
    # Scenario 1: nominal disturbance (as in the report Section 4.1)
    # ------------------------------------------------------------------
    t_f, X_f = run_simulation(use_fuzzy=True, dist_scale=1.0)
    t_p, X_p = run_simulation(use_fuzzy=False, dist_scale=1.0)
    pp_f = postprocess(t_f, X_f, use_fuzzy=True)
    pp_p = postprocess(t_p, X_p, use_fuzzy=False)

    metrics_nom = {
        "Fuzzy PD": metrics_of((pp_f["ex"], pp_f["ey"], pp_f["ez"])),
        "PD": metrics_of((pp_p["ex"], pp_p["ey"], pp_p["ez"])),
    }
    save_metrics_csv(os.path.join(OUT_DIR, "pd_vs_fuzzypd_metrics.csv"), metrics_nom)
    print("Nominal disturbance:", metrics_nom)

    # ---- Fig: 3D trajectory ----
    fig1 = plt.figure("3D Trajectory Comparison", figsize=(6, 5))
    ax = fig1.add_subplot(projection="3d")
    ax.plot(X_f[:, 0], X_f[:, 1], X_f[:, 2], "r", linewidth=1.5, label="Fuzzy PD")
    ax.plot(X_p[:, 0], X_p[:, 1], X_p[:, 2], "b--", linewidth=1.5, label="PD")
    ax.plot(pp_f["xd"], pp_f["yd"], pp_f["zd"], "k:", linewidth=2, label="Desired")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("Trajectory comparison: Fuzzy PD vs. PD")
    ax.legend()
    fig1.savefig(os.path.join(OUT_DIR, "compare_3d_trajectory.png"), dpi=150)

    # ---- Fig: tracking error comparison ----
    fig2, axs = plt.subplots(3, 1, num="Tracking error comparison", sharex=True, figsize=(7, 8))
    for ax_, e_f, e_p, lab in zip(axs, (pp_f["ex"], pp_f["ey"], pp_f["ez"]), (pp_p["ex"], pp_p["ey"], pp_p["ez"]), ("x", "y", "z")):
        ax_.plot(t_f, e_f, "r", label="Fuzzy PD")
        ax_.plot(t_p, e_p, "b--", label="PD")
        ax_.set_ylabel(f"$e_{lab}$ (m)")
        ax_.legend(); ax_.grid(True)
    axs[0].set_title("Tracking error: Fuzzy PD vs. PD")
    axs[2].set_xlabel("Time (s)")
    fig2.savefig(os.path.join(OUT_DIR, "compare_tracking_error.png"), dpi=150)

    # ---- Fig: position tracking (actual vs desired, both controllers) ----
    fig3, axs3 = plt.subplots(3, 1, num="Position tracking comparison", sharex=True, figsize=(7, 8))
    labels_pos = [("x", 0, "xd"), ("y", 1, "yd"), ("z", 2, "zd")]
    for ax_, (lab, idx, dkey) in zip(axs3, labels_pos):
        ax_.plot(t_f, X_f[:, idx], "r", label="Fuzzy PD")
        ax_.plot(t_p, X_p[:, idx], "b--", label="PD")
        ax_.plot(t_f, pp_f[dkey], "k:", linewidth=2, label="Desired")
        ax_.set_ylabel(f"${lab}$ (m)")
        ax_.legend(); ax_.grid(True)
    axs3[0].set_title("Position tracking: Fuzzy PD vs. PD vs. Reference")
    axs3[2].set_xlabel("Time (s)")
    fig3.savefig(os.path.join(OUT_DIR, "compare_position_tracking.png"), dpi=150)

    # ---- Fig: attitude tracking comparison (actual vs reference, both controllers) ----
    fig4, axs4 = plt.subplots(3, 1, num="Attitude tracking comparison", sharex=True, figsize=(7, 8))
    axs4[0].plot(t_f, X_f[:, 6], "r", label="Fuzzy PD $\\phi$")
    axs4[0].plot(t_p, X_p[:, 6], "b--", label="PD $\\phi$")
    axs4[0].set_ylabel("$\\phi$ (rad)")
    axs4[0].legend(fontsize=8); axs4[0].grid(True)
    axs4[0].set_title("Attitude response: Fuzzy PD vs. PD")

    axs4[1].plot(t_f, X_f[:, 7], "r", label="Fuzzy PD $\\theta$")
    axs4[1].plot(t_p, X_p[:, 7], "b--", label="PD $\\theta$")
    axs4[1].set_ylabel("$\\theta$ (rad)")
    axs4[1].legend(fontsize=8); axs4[1].grid(True)

    axs4[2].plot(t_f, X_f[:, 8], "r", label="Fuzzy PD $\\psi$")
    axs4[2].plot(t_p, X_p[:, 8], "b--", label="PD $\\psi$")
    axs4[2].set_ylabel("$\\psi$ (rad)")
    axs4[2].set_xlabel("Time (s)")
    axs4[2].legend(fontsize=8); axs4[2].grid(True)
    fig4.savefig(os.path.join(OUT_DIR, "compare_attitude_tracking.png"), dpi=150)

    # ---- Fig: online gain adaptation (Fuzzy PD only) ----
    fig5, axs5 = plt.subplots(3, 2, num="Gain adaptation", sharex=True, figsize=(9, 8))
    axis_data = [
        ("x", pp_f["dKpx"], pp_f["dKdx"]),
        ("y", pp_f["dKpy"], pp_f["dKdy"]),
        ("z", pp_f["dKpz"], pp_f["dKdz"]),
    ]
    for row, (lab, dkp, dkd) in enumerate(axis_data):
        axs5[row, 0].plot(t_f, dkp, "g")
        axs5[row, 0].set_ylabel(f"$\\Delta K_p^{{{lab}}}$")
        axs5[row, 0].grid(True)
        axs5[row, 1].plot(t_f, dkd, "m")
        axs5[row, 1].set_ylabel(f"$\\Delta K_d^{{{lab}}}$")
        axs5[row, 1].grid(True)
    axs5[0, 0].set_title("$\\Delta K_p^i(t)$")
    axs5[0, 1].set_title("$\\Delta K_d^i(t)$")
    axs5[2, 0].set_xlabel("Time (s)")
    axs5[2, 1].set_xlabel("Time (s)")
    fig5.suptitle("Online Kp, Kd gain adaptation (Fuzzy PD controller)")
    fig5.tight_layout()
    fig5.savefig(os.path.join(OUT_DIR, "gain_adaptation.png"), dpi=150)

    # ------------------------------------------------------------------
    # Scenario 2: robustness under amplified disturbance (x3)
    # ------------------------------------------------------------------
    SCALE = 3.0
    t_f2, X_f2 = run_simulation(use_fuzzy=True, dist_scale=SCALE)
    t_p2, X_p2 = run_simulation(use_fuzzy=False, dist_scale=SCALE)
    pp_f2 = postprocess(t_f2, X_f2, use_fuzzy=True)
    pp_p2 = postprocess(t_p2, X_p2, use_fuzzy=False)

    metrics_robust = {
        "Fuzzy PD": metrics_of((pp_f2["ex"], pp_f2["ey"], pp_f2["ez"])),
        "PD": metrics_of((pp_p2["ex"], pp_p2["ey"], pp_p2["ez"])),
    }
    save_metrics_csv(os.path.join(OUT_DIR, "pd_vs_fuzzypd_metrics_robust.csv"), metrics_robust)
    print(f"Amplified disturbance (x{SCALE}):", metrics_robust)

    fig6, axs6 = plt.subplots(3, 1, num="Robustness comparison", sharex=True, figsize=(7, 8))
    for ax_, e_f, e_p, lab in zip(
        axs6, (pp_f2["ex"], pp_f2["ey"], pp_f2["ez"]), (pp_p2["ex"], pp_p2["ey"], pp_p2["ez"]), ("x", "y", "z")
    ):
        ax_.plot(t_f2, e_f, "r", label="Fuzzy PD")
        ax_.plot(t_p2, e_p, "b--", label="PD")
        ax_.set_ylabel(f"$e_{lab}$ (m)")
        ax_.legend(); ax_.grid(True)
    axs6[0].set_title(f"Tracking error with disturbance amplified {SCALE:g}x")
    axs6[2].set_xlabel("Time (s)")
    fig6.savefig(os.path.join(OUT_DIR, "compare_tracking_error_robust.png"), dpi=150)

    print("All figures and metrics saved to", OUT_DIR)


if __name__ == "__main__":
    main()
