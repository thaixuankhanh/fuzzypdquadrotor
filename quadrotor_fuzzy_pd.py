"""
Python port of FuzzyPDController.m

Two Mamdani FIS blocks (delKp.fis, delKd.fis) compute online
corrections to the P and D gains of the position controller based on
the position error and its derivative, exactly as in the original
MATLAB/Simulink simulation. The attitude loop uses a fixed-gain PD
controller, as in the paper's simulation model.

Run:
    python quadrotor_fuzzy_pd.py
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


def main() -> None:
    # ------------------------------------------------
    # Physical parameters (same values as FuzzyPDController.m)
    # ------------------------------------------------
    m = 1.7  # Quadrotor mass (kg)
    g = 9.81  # Gravity acceleration (m/s^2)

    Ix = 4e-3  # Moments of inertia (kg*m^2)
    Iy = 4e-3
    Iz = 8.4e-3

    k = 2.98e-5  # Lift constant   (unused directly below, kept for parity)
    b_drag = 3.23e-7  # Drag constant
    l = 0.25  # Arm length (m)

    Jx = 5.7e-4  # Drag coefficients
    Jy = 5.7e-4
    Jz = 5.7e-4

    # Time constant of the "dirty derivative" filter used to obtain
    # phi_d_dot / theta_d_dot from phi_d / theta_d (see f() below). This is
    # a fixed filter constant, deliberately decoupled from the ODE solver's
    # (adaptive) step size -- it emulates the fixed-rate differentiator a
    # real digital attitude-reference generator would use.
    tau_diff = 0.01

    # Initial state vector x0 (14 x 1):
    # [x, y, z, vx, vy, vz, phi, theta, psi, phi_dot, theta_dot, psi_dot,
    #  phi_d_prev, theta_d_prev]
    x0 = np.zeros(14)
    x0[2] = 1.0  # z(0) = 1

    t0, Tf = 0.0, 60.0

    fis1 = read_fis(os.path.join(HERE, "delKp.fis"))
    fis2 = read_fis(os.path.join(HERE, "delKd.fis"))

    Kp_p = np.array([0.1, 0.1, 0.1])
    Kd_p = np.array([1 / 3, 1 / 3, 1 / 3])

    Kp_a = np.array([0.1, 0.1, 0.1])
    Kd_a = np.array([1 / 3, 1 / 3, 1 / 3])

    # ------------------------------------------------
    # Closed-loop dynamics (equivalent of the nested function f(t, X))
    # ------------------------------------------------
    def f(t: float, X: np.ndarray) -> np.ndarray:
        # Reference trajectory
        xd = np.cos(np.pi * t / 20)
        yd = np.sin(np.pi * t / 20)
        zd = 2.0
        psi_d = 0.0
        psi_d_dot = 0.0

        xd_dot = -(np.pi / 20) * np.sin(np.pi * t / 20)
        yd_dot = (np.pi / 20) * np.cos(np.pi * t / 20)
        zd_dot = 0.0

        # State unpacking
        x, y, z = X[0], X[1], X[2]
        x_dot, y_dot, z_dot = X[3], X[4], X[5]

        phi, theta, psi = X[6], X[7], X[8]
        phi_dot, theta_dot, psi_dot = X[9], X[10], X[11]

        phi_d_prev = X[12]
        theta_d_prev = X[13]

        # --- Fuzzy PD position controller -----------------------------
        ex = xd - x
        ex_dot = xd_dot - x_dot
        ey = yd - y
        ey_dot = yd_dot - y_dot
        ez = zd - z
        ez_dot = zd_dot - z_dot

        ex = sat(ex, -5, 5)
        ey = sat(ey, -5, 5)
        ez = sat(ez, -5, 5)

        ex_dot = sat(ex_dot, -5, 5)
        ey_dot = sat(ey_dot, -5, 5)
        ez_dot = sat(ez_dot, -5, 5)

        delKpx = evalfis(fis1, [ex, ex_dot])[0]
        delKdx = evalfis(fis2, [ex, ex_dot])[0]

        delKpy = evalfis(fis1, [ey, ey_dot])[0]
        delKdy = evalfis(fis2, [ey, ey_dot])[0]

        delKpz = evalfis(fis1, [ez, ez_dot])[0]
        delKdz = evalfis(fis2, [ez, ez_dot])[0]

        ux = (Kp_p[0] + delKpx) * ex + (Kd_p[0] + delKdx) * ex_dot
        uy = (Kp_p[1] + delKpy) * ey + (Kd_p[1] + delKdy) * ey_dot
        uz = (Kp_p[2] + delKpz) * ez + (Kd_p[2] + delKdz) * ez_dot

        T = m * np.sqrt(ux**2 + uy**2 + (uz + g) ** 2)

        phi_d = np.arcsin((ux * np.sin(psi_d) - uy * np.cos(psi_d)) / np.sqrt(ux**2 + uy**2 + (uz + g) ** 2))
        theta_d = np.arctan((ux * np.cos(psi_d) + uy * np.sin(psi_d)) / (uz + g))

        # "Dirty derivative" (band-limited differentiator): phi_d_prev /
        # theta_d_prev are themselves integrated states that chase phi_d /
        # theta_d with time constant tau_diff, giving a smooth estimate of
        # their derivative without differentiating the fuzzy PD output
        # (ux, uy, uz) symbolically.
        phi_d_dot = (phi_d - phi_d_prev) / tau_diff
        theta_d_dot = (theta_d - theta_d_prev) / tau_diff

        # --- PD attitude controller ------------------------------------
        e_phi = phi_d - phi
        e_theta = theta_d - theta
        e_psi = psi_d - psi

        e_phi_dot = phi_d_dot - phi_dot
        e_theta_dot = theta_d_dot - theta_dot
        e_psi_dot = psi_d_dot - psi_dot

        tau_phi = Kp_a[0] * e_phi + Kd_a[0] * e_phi_dot
        tau_theta = Kp_a[1] * e_theta + Kd_a[1] * e_theta_dot
        tau_psi = Kp_a[2] * e_psi + Kd_a[2] * e_psi_dot

        # --- Disturbances (time-varying, per axis) -----------------------
        dis_x = 0.05 * np.sin(t) * np.cos(0.5 * t) ** 2
        dis_y = 0.05 * np.sin(t) * np.cos(0.5 * t) ** 2
        dis_z = 0.05 * np.sin(t) * np.cos(0.5 * t) ** 2

        dis_phi = 0.02 * np.sin(0.7 * t)
        dis_theta = 0.02 * np.sin(0.7 * t)
        dis_psi = 0.02 * np.sin(0.7 * t)

        # --- Derivative equations of attitudes ---------------------------
        phi_dot2 = theta_dot * psi_dot * (Iy - Iz) / Ix + tau_phi / Ix + dis_phi
        theta_dot2 = phi_dot * psi_dot * (Iz - Ix) / Iy + tau_theta / Iy + dis_theta
        psi_dot2 = phi_dot * theta_dot * (Ix - Iy) / Iz + tau_psi / Iz + dis_psi

        # --- Derivative equations of positions ----------------------------
        x_dot2 = (T / m) * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) - (Jx / m) * x_dot + dis_x
        y_dot2 = (T / m) * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) - (Jy / m) * y_dot + dis_y
        z_dot2 = (T / m) * np.cos(phi) * np.cos(theta) - g - (Jz / m) * z_dot + dis_z

        deq = np.zeros(14)
        deq[0] = x_dot
        deq[1] = y_dot
        deq[2] = z_dot

        deq[3] = x_dot2
        deq[4] = y_dot2
        deq[5] = z_dot2

        deq[6] = phi_dot
        deq[7] = theta_dot
        deq[8] = psi_dot

        deq[9] = phi_dot2
        deq[10] = theta_dot2
        deq[11] = psi_dot2

        deq[12] = phi_d_dot
        deq[13] = theta_d_dot
        return deq

    # ------------------------------------------------
    # Integrate (ode45 -> RK45, same tolerances / max step)
    # ------------------------------------------------
    sol = solve_ivp(
        f,
        (t0, Tf),
        x0,
        method="RK45",
        rtol=1e-3,
        atol=1e-5,
        max_step=0.02,
        dense_output=False,
    )

    t = sol.t
    X = sol.y.T

    # Extract states
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    phi = X[:, 6]
    theta = X[:, 7]
    psi = X[:, 8]

    phi_d_prev = X[:, 12]
    theta_d_prev = X[:, 13]

    xd = np.cos(np.pi * t / 20)
    yd = np.sin(np.pi * t / 20)
    zd = 2 + 0 * t
    psi_d = np.zeros_like(t)

    # ------------------------------------------------
    # Plots
    # ------------------------------------------------
    out_dir = os.path.join(HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    fig1 = plt.figure("3D Trajectory")
    ax = fig1.add_subplot(projection="3d")
    ax.plot(x, y, z, "b", linewidth=2, label="Actual")
    ax.plot(xd, yd, zd, "r--", linewidth=2, label="Desired")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("3D Trajectory Tracking")
    ax.legend()
    ax.grid(True)
    fig1.savefig(os.path.join(out_dir, "3d_trajectory.png"), dpi=150)

    fig2, axs2 = plt.subplots(3, 1, num="Position tracking", sharex=True)
    axs2[0].plot(t, x, "b", t, xd, "r--", linewidth=1.5)
    axs2[0].set_ylabel("x (m)")
    axs2[0].legend(["Actual", "Desired"])
    axs2[0].set_title("Position tracking")
    axs2[0].grid(True)
    axs2[1].plot(t, y, "b", t, yd, "r--", linewidth=1.5)
    axs2[1].set_ylabel("y (m)")
    axs2[1].legend(["Actual", "Desired"])
    axs2[1].grid(True)
    axs2[2].plot(t, z, "b", t, zd, "r--", linewidth=1.5)
    axs2[2].set_ylabel("z (m)")
    axs2[2].set_xlabel("Time (s)")
    axs2[2].legend(["Actual", "Desired"])
    axs2[2].grid(True)
    fig2.savefig(os.path.join(out_dir, "position_tracking.png"), dpi=150)

    fig3, axs3 = plt.subplots(3, 1, num="Attitude tracking", sharex=True)
    fig3.patch.set_facecolor("w")
    axs3[0].plot(t, phi, "b", t, phi_d_prev, "r--", linewidth=1.5)
    axs3[0].set_ylabel(r"$\phi$ (rad)")
    axs3[0].legend([r"$\phi$", r"$\phi_d$"])
    axs3[0].set_title("Attitude tracking")
    axs3[0].grid(True)
    axs3[1].plot(t, theta, "b", t, theta_d_prev, "r--", linewidth=1.5)
    axs3[1].set_ylabel(r"$\theta$ (rad)")
    axs3[1].legend([r"$\theta$", r"$\theta_d$"])
    axs3[1].grid(True)
    axs3[2].plot(t, psi, "b", t, psi_d, "r--", linewidth=1.5)
    axs3[2].set_ylabel(r"$\psi$ (rad)")
    axs3[2].set_xlabel("Time (s)")
    axs3[2].legend([r"$\psi$", r"$\psi_d$"])
    axs3[2].grid(True)
    fig3.savefig(os.path.join(out_dir, "attitude_tracking.png"), dpi=150)

    print(f"Simulation done: {len(t)} time steps, saved figures to '{out_dir}'.")
    plt.show()


if __name__ == "__main__":
    main()
