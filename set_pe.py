#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import expm, solve_discrete_are, inv
from numpy.linalg import pinv # Use NumPy's pinv
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
import seaborn as sns
import random # For potential future use, though APF is deterministic here
import math

# --- Clohessy-Wiltshire (CW) Equation Matrices ---

def calculate_cw_matrices(omega):
    """ Calculates continuous-time state-space matrices (A, B) for 3D CW equations. """
    A_cont = np.array([
        [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
        [3 * omega**2, 0, 0, 0, 2 * omega, 0],
        [0, 0, 0, -2 * omega, 0, 0], [0, 0, -omega**2, 0, 0, 0]
    ])
    B_cont = np.array([
        [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ])
    return A_cont, B_cont

# --- Spectral Analysis Function (Using SETS Normalization) ---

def cw_spectral_analysis_sets(omega, u_bar, u_underline, dt, H):
    """ Performs spectral analysis using SETS normalization. """
    if H <= 0: raise ValueError("Horizon H must be positive.")
    if u_bar.shape != (3,) or u_underline.shape != (3,): raise ValueError("Control bounds shape")
    if np.any(u_bar <= u_underline): raise ValueError("Upper < Lower bounds")
    A_cont, B_cont = calculate_cw_matrices(omega)
    n_states, n_inputs = A_cont.shape[0], B_cont.shape[1]
    system_cont = (A_cont, B_cont, np.eye(n_states), np.zeros((n_states, n_inputs)))
    A_d, B_d, _, _, _ = cont2discrete(system_cont, dt, method='zoh')
    s_diag = [2.0 / r if r > 1e-9 else 1e9 for r in (u_bar - u_underline)]
    if any(r < 1e-9 for r in (u_bar - u_underline)): print("Warning: Zero control range detected.")
    S = np.diag(s_diag)
    B_norm = B_d @ S
    C_list = []
    Ad_powers = {0: np.eye(n_states)}
    if H > 1:
      current_Ad_power = np.eye(n_states)
      for k in range(1, H): Ad_powers[k] = Ad_powers[k-1] @ A_d
    for k in range(H): C_list.insert(0, Ad_powers[H - 1 - k] @ B_norm)
    C_H = np.hstack(C_list)
    G = C_H @ C_H.T; G = (G + G.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    tol = 1e-10
    valid_indices = eigenvalues_sorted > tol
    eigenvalues_filtered = eigenvalues_sorted[valid_indices]
    eigenvectors_filtered = eigenvectors_sorted[:, valid_indices]
    if len(eigenvalues_filtered) < len(eigenvalues_sorted): print(f"Warning: Filtered modes.")
    return eigenvalues_filtered, eigenvectors_filtered, A_d, B_d, S, C_H

# --- LQR Feedback Controller Gain Calculation ---
def calculate_lqr_gain(A_d, B_d, Q, R):
    """ Calculates the discrete-time LQR feedback gain K using DARE. """
    try:
        P = solve_discrete_are(A_d, B_d, Q, R)
        term_inv = R + B_d.T @ P @ B_d
        if abs(np.linalg.det(term_inv)) < 1e-12:
             print("Warning: R + B'PB is nearly singular in LQR gain calculation. Using pseudoinverse.")
             term_inv_actual = np.linalg.pinv(term_inv, rcond=1e-10) # Added rcond here too
        else:
             term_inv_actual = inv(term_inv)
        K = term_inv_actual @ (B_d.T @ P @ A_d)
        return K
    except Exception as e:
        print(f"Error calculating LQR gain: {e}")
        return np.zeros((B_d.shape[1], A_d.shape[0]))

# --- MODIFIED APF Evader Simulation Step ---
def simulate_evader_step_apf(evader_state, pursuer_state, A_d_evader, B_d_evader,
                             k_rep, d_safe, # APF repulsive parameters
                             evader_accel_limit, dt):
    """ Simulates one step of an evader using APF logic. """
    n_states = evader_state.shape[0]
    n_inputs = B_d_evader.shape[1]
    pos_evader = evader_state[:3] # Use first 3 elements for position
    pos_pursuer = pursuer_state[:3] # Use first 3 elements for position

    pos_rel_ep = pos_pursuer - pos_evader # Vector from evader to pursuer
    dist_rel = np.linalg.norm(pos_rel_ep)

    # Calculate repulsive force/acceleration (pointing away from pursuer)
    force_repulsive = np.zeros(n_inputs) # Initialize as zero vector (3,)
    if 1e-6 < dist_rel < d_safe: # Apply repulsion only within safety distance
        # --- ADJUSTED REPULSION FORMULA (Example: 1/r^2 type) ---
        # magnitude = k_rep / dist_rel**2 # Simple inverse square
        # OR use the gradient formula (as before) - let's stick with gradient for now
        magnitude = k_rep * (1.0 / dist_rel - 1.0 / d_safe) * (1.0 / dist_rel**2)
        # ---

        direction = -pos_rel_ep / dist_rel # Force points from pursuer to evader (away from pursuer)
        force_repulsive = magnitude * direction

    # No attractive force in this version
    a_desired = force_repulsive

    # Limit the acceleration
    accel_magnitude = np.linalg.norm(a_desired)
    if accel_magnitude > evader_accel_limit:
        a_evader_actual = a_desired / accel_magnitude * evader_accel_limit
    else:
        a_evader_actual = a_desired

    # Simulate next state
    unforced_next_state = A_d_evader @ evader_state
    control_effect = B_d_evader @ a_evader_actual
    next_state = unforced_next_state + control_effect

    return next_state


# --- Simulation and Evaluation Function (Using Fixed Target LQR & APF Evader) ---
def simulate_sets_expansion(
    current_pursuer_state, current_evader_state,
    mode_index, eigenvalues, eigenvectors, H, dt,
    A_d_pursuer, B_d_pursuer, K_lqr,
    A_d_evader, B_d_evader, # Evader dynamics matrices
    k_rep, d_safe, evader_accel_limit, # APF Parameters
    Q_eval, R_eval,                 # Cost matrices for evaluation
    u_bar, u_underline              # Control limits for clipping
    # Removed C_H, S as not needed for this fixed target logic
    ):
    """
    Simulates one expansion branch using fixed target LQR and APF evader prediction.
    """
    # (Implementation remains the same as the previous 'fixed target LQR' version)
    pursuer_traj = [current_pursuer_state]
    evader_traj = [current_evader_state]
    control_sequence = []
    total_cost = 0.0
    n_states, n_inputs = A_d_pursuer.shape[0], B_d_pursuer.shape[1]
    mode_num = mode_index // 2
    mode_sign = 1.0 if mode_index % 2 == 0 else -1.0
    if mode_num >= len(eigenvalues): return np.inf, np.zeros(n_inputs), np.array([current_pursuer_state]* (H+1)), np.array([current_evader_state]*(H+1))
    lambda_i = max(0, eigenvalues[mode_num])
    v_i = eigenvectors[:, mode_num]
    target_deviation = mode_sign * np.sqrt(lambda_i) * v_i
    target_state_absolute_ref = current_pursuer_state + target_deviation
    pursuer_state_k = current_pursuer_state
    evader_state_k = current_evader_state
    for k in range(H):
        state_error = pursuer_state_k - target_state_absolute_ref
        actual_control_k = -K_lqr @ state_error
        actual_control_k_clipped = np.clip(actual_control_k, u_underline, u_bar)
        control_sequence.append(actual_control_k_clipped)
        pursuer_state_k_plus_1 = A_d_pursuer @ pursuer_state_k + B_d_pursuer @ actual_control_k_clipped
        evader_state_k_plus_1 = simulate_evader_step_apf( # Use APF prediction
            evader_state_k, pursuer_state_k,
            A_d_evader, B_d_evader,
            k_rep, d_safe, evader_accel_limit, dt
        )
        pursuer_traj.append(pursuer_state_k_plus_1)
        evader_traj.append(evader_state_k_plus_1)
        relative_state_k_plus_1 = pursuer_state_k_plus_1 - evader_state_k_plus_1
        step_cost = relative_state_k_plus_1.T @ Q_eval @ relative_state_k_plus_1 + actual_control_k_clipped.T @ R_eval @ actual_control_k_clipped
        if not np.isfinite(step_cost) or step_cost > 1e15: return np.inf, np.zeros(n_inputs), np.array(pursuer_traj), np.array(evader_traj)
        total_cost += step_cost
        pursuer_state_k = pursuer_state_k_plus_1
        evader_state_k = evader_state_k_plus_1
    first_control = control_sequence[0] if control_sequence else np.zeros(n_inputs)
    final_p_traj = np.array(pursuer_traj); final_e_traj = np.array(evader_traj)
    if final_p_traj.shape[0] < H + 1: final_p_traj = np.vstack([final_p_traj] + [final_p_traj[-1]] * (H + 1 - final_p_traj.shape[0]))
    if final_e_traj.shape[0] < H + 1: final_e_traj = np.vstack([final_e_traj] + [final_e_traj[-1]] * (H + 1 - final_e_traj.shape[0]))
    return total_cost, first_control, final_p_traj, final_e_traj

# --- Core Decision Logic (Uses APF Evader Sim Params) ---

def choose_best_control_sets_like(
    current_pursuer_state, current_evader_state,
    pursuer_eigenvalues, pursuer_eigenvectors, H, dt,
    A_d_pursuer, B_d_pursuer, K_lqr,
    A_d_evader, B_d_evader, # Evader dynamics matrices
    k_rep, d_safe, evader_accel_limit, # APF Parameters
    Q_eval, R_eval, # Removed C_H, S if not needed by sim logic
    u_bar, u_underline,
    previous_mode_index=-1, mode_switch_penalty=0.0
    ):
    """ Selects best control based on evaluating SETS-like expansions with APF evader. """
    # (Implementation remains the same as the previous version)
    best_cost_eval = np.inf
    best_control_u0 = np.zeros(B_d_pursuer.shape[1])
    best_mode_index_current = -1
    best_pursuer_traj = None
    best_evader_traj = None
    num_eigenmodes = len(pursuer_eigenvalues)
    n_inputs = B_d_pursuer.shape[1]
    for i in range(2 * num_eigenmodes):
        cost, u0, p_traj, e_traj = simulate_sets_expansion(
            current_pursuer_state, current_evader_state,
            i, pursuer_eigenvalues, pursuer_eigenvectors, H, dt,
            A_d_pursuer, B_d_pursuer, K_lqr,
            A_d_evader, B_d_evader,
            k_rep, d_safe, evader_accel_limit, # Pass APF params
            Q_eval, R_eval,
            u_bar, u_underline
        )
        switch_penalty = 0.0
        if previous_mode_index != -1 and i != previous_mode_index: switch_penalty = mode_switch_penalty
        current_eval_cost = cost + switch_penalty
        if np.isinf(cost): current_eval_cost = np.inf
        if current_eval_cost < best_cost_eval:
            best_cost_eval = current_eval_cost; best_control_u0 = u0
            best_mode_index_current = i; best_pursuer_traj = p_traj; best_evader_traj = e_traj
    if best_pursuer_traj is None or np.isinf(best_cost_eval):
         print("Warning: All mode simulations failed/Inf cost.")
         best_pursuer_traj = np.array([current_pursuer_state]* (H+1)); best_evader_traj = np.array([current_evader_state]* (H+1))
         best_control_u0 = np.zeros(n_inputs); best_mode_index_current = -1
    return best_control_u0, best_mode_index_current, best_pursuer_traj, best_evader_traj


# --- Main Simulation Loop ---
if __name__ == "__main__":
    # --- Simulation Parameters ---
    print("Setting up simulation parameters with APF Evader...")
    G = 6.67430e-11; M_earth = 5.972e24; r_orbit = (6371 + 400) * 1000
    omega = np.sqrt(G * M_earth / r_orbit**3)

    # Control Authority
    accel_max_pursuer_val = 0.05
    pursuer_u_bar = np.array([accel_max_pursuer_val] * 3)
    pursuer_u_underline = np.array([-accel_max_pursuer_val] * 3)
    evader_accel_limit_val = 0.05 # Evader is less agile

    # Time / Horizon
    dt = 5.0
    H = 30 # Slightly longer horizon for reactive opponent
    T_sim = 500
    n_steps = int(T_sim / dt)
    CAPTURE_RADIUS = 15.0

    # --- APF Parameters (TUNABLE) ---
    APF_K_REP = 5000.0   # Repulsion strength - INCREASED SIGNIFICANTLY
    APF_D_SAFE = 1050.0 # Safety distance [m] - INCREASED SIGNIFICANTLY

    # --- Initial States ---
    pursuer_state_initial = np.array([0.0, -150.0, 10.0, 0.0, 0.0, 0.0])
    evader_state_initial = np.array([500.0, 0.0, 20.0, 0.0, 0.0, 0.0])

    # --- Evader Dynamics ---
    A_cont_evader, B_cont_evader = calculate_cw_matrices(omega)
    system_cont_evader = (A_cont_evader, B_cont_evader, np.eye(6), np.zeros((6,3)))
    A_d_evader, B_d_evader, _, _, _ = cont2discrete(system_cont_evader, dt, method='zoh')

    # --- Pursuer Model & Spectrum ---
    print("Calculating Pursuer's Constant CW Spectrum (SETS-like)...")
    pursuer_eigvals, pursuer_eigvecs, A_d_p, B_d_p, S_p, C_H_p = \
        cw_spectral_analysis_sets(omega, pursuer_u_bar, pursuer_u_underline, dt, H)
    if len(pursuer_eigvals) == 0: exit()
    n_eigenvectors_found = pursuer_eigvecs.shape[1]
    print(f"  Done. Found {n_eigenvectors_found} valid modes.")

    # --- LQR Controller Gain (Moderate tuning from last attempt) ---
    print("Calculating Pursuer's LQR Gain...")
    Q_lqr = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    R_lqr = np.diag([100.0, 100.0, 100.0]) # Keep R high initially for stability
    K_lqr = calculate_lqr_gain(A_d_p, B_d_p, Q_lqr, R_lqr)
    if np.any(np.isnan(K_lqr)): exit()
    print("  Done.")

    # --- Evaluation Cost Matrices ---
    Q_eval = np.diag([10.0, 10.0, 10.0, 0.1, 0.1, 0.1])
    R_eval = np.diag([1.0, 1.0, 1.0]) # Keep eval R moderate

    # --- Optional: Mode Switch Penalty ---
    MODE_SWITCH_PENALTY_WEIGHT = 0.0

    # --- Simulation Loop ---
    pursuer_history = [pursuer_state_initial]
    evader_history = [evader_state_initial]
    control_history = []
    chosen_mode_history = []
    simulated_pursuer_lookahead = None
    simulated_evader_lookahead = None
    previous_chosen_mode = -1

    current_pursuer_state = pursuer_state_initial.copy()
    current_evader_state = evader_state_initial.copy()

    print(f"\nStarting {n_steps}-step simulation (Target time: {T_sim}s)...")
    capture_step = -1

    for step in range(n_steps):
        # --- Core Decision ---
        u0, chosen_mode_idx, sim_p_traj, sim_e_traj = choose_best_control_sets_like(
            current_pursuer_state, current_evader_state,
            pursuer_eigvals, pursuer_eigvecs, H, dt,
            A_d_p, B_d_p, K_lqr,
            A_d_evader, B_d_evader,
            APF_K_REP, APF_D_SAFE, evader_accel_limit_val, # Pass APF params
            Q_eval, R_eval,
            # C_H_p, S_p, # Not needed for fixed target LQR
            pursuer_u_bar, pursuer_u_underline,
            previous_mode_index=previous_chosen_mode,
            mode_switch_penalty=MODE_SWITCH_PENALTY_WEIGHT
        )
        control_history.append(u0)
        chosen_mode_history.append(chosen_mode_idx)
        previous_chosen_mode = chosen_mode_idx

        if step == 0:
             simulated_pursuer_lookahead = sim_p_traj
             simulated_evader_lookahead = sim_e_traj

        # --- Apply Control and Update "Real" States ---
        next_pursuer_state = A_d_p @ current_pursuer_state + B_d_p @ u0
        # Evader moves using APF in the actual simulation
        next_evader_state = simulate_evader_step_apf(
            current_evader_state, current_pursuer_state,
            A_d_evader, B_d_evader,
            APF_K_REP, APF_D_SAFE, evader_accel_limit_val, dt
        )

        pursuer_history.append(next_pursuer_state)
        evader_history.append(next_evader_state)
        current_pursuer_state = next_pursuer_state
        current_evader_state = next_evader_state

        # Check capture & print status
        relative_dist = np.linalg.norm(current_pursuer_state[0:3] - current_evader_state[0:3])
        # ... (print status logic remains the same) ...
        if chosen_mode_idx != -1: mode_print_str = f"Mode {(chosen_mode_idx // 2) + 1}{'+' if chosen_mode_idx % 2 == 0 else '-'}"
        else: mode_print_str = "None"
        print(f"  Step {step+1:3d}/{n_steps} - Rel Dist: {relative_dist:7.2f} m - Chosen: {mode_print_str} - Control: [{u0[0]:+.4f} {u0[1]:+.4f} {u0[2]:+.4f}]")

        if relative_dist < CAPTURE_RADIUS and capture_step == -1:
            print(f"\nCAPTURE! Relative distance {relative_dist:.2f} m < {CAPTURE_RADIUS} m at step {step+1}.")
            capture_step = step + 1
            break # Stop on capture

        elif step == n_steps - 1:
             print("\nSimulation finished.")
             if capture_step == -1: print(f"  Capture condition not met (Final Dist: {relative_dist:.2f}m).")

    # --- Convert history ---
    pursuer_history = np.array(pursuer_history)
    evader_history = np.array(evader_history)
    control_history = np.array(control_history)
    chosen_mode_history = np.array(chosen_mode_history)
    time_axis = np.arange(len(pursuer_history)) * dt
    control_time_axis = time_axis[:-1] if len(time_axis) > 1 else np.array([])

    # --- Plotting Results ---
    # (Plotting code remains the same)
    print("\nGenerating plots...")
    # ... (Include all 4 plots: XY Trajectory, Relative Distance, Control Input, Chosen Mode) ...
    # XY Plane Trajectory Plot
    fig_xy, ax_xy = plt.subplots(figsize=(10, 8))
    ax_xy.plot(evader_history[:, 1], evader_history[:, 0], 'r--', label='Evader Trajectory')
    ax_xy.plot(evader_history[0, 1], evader_history[0, 0], 'ro', markersize=8, label='Evader Start')
    ax_xy.plot(evader_history[-1, 1], evader_history[-1, 0], 'rs', markersize=8, label='Evader End')
    if capture_step > 0: ax_xy.plot(evader_history[capture_step, 1], evader_history[capture_step, 0], 'rx', markersize=12, mew=2, label='Evader Capture Pt')

    ax_xy.plot(pursuer_history[:, 1], pursuer_history[:, 0], 'b-', label='Pursuer Trajectory')
    ax_xy.plot(pursuer_history[0, 1], pursuer_history[0, 0], 'bo', markersize=8, label='Pursuer Start')
    ax_xy.plot(pursuer_history[-1, 1], pursuer_history[-1, 0], 'bs', markersize=8, label='Pursuer End')
    if capture_step > 0: ax_xy.plot(pursuer_history[capture_step, 1], pursuer_history[capture_step, 0], 'bx', markersize=12, mew=2, label='Pursuer Capture Pt')

    if simulated_pursuer_lookahead is not None and simulated_evader_lookahead is not None:
        ax_xy.plot(simulated_evader_lookahead[:, 1], simulated_evader_lookahead[:, 0], ':', color='orange', alpha=0.5, linewidth=1, label=f'Sim. Evader Lookahead (H={H})')
        ax_xy.plot(simulated_pursuer_lookahead[:, 1], simulated_pursuer_lookahead[:, 0], ':', color='cyan', alpha=0.5, linewidth=1, label=f'Sim. Pursuer Lookahead (H={H})')

    ax_xy.set_xlabel('Y (Along-track) [m]')
    ax_xy.set_ylabel('X (Radial) [m]')
    ax_xy.set_title(f'Pursuer-Evader Trajectory (XY Plane - LVLH) - Capture {"Successful" if capture_step > 0 else "Failed"}')
    ax_xy.legend(loc='best')
    ax_xy.grid(True)
    ax_xy.axis('equal')
    fig_xy.tight_layout()

    # Relative Distance Plot
    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
    relative_distance = np.linalg.norm(pursuer_history[:, 0:3] - evader_history[:, 0:3], axis=1)
    ax_dist.plot(time_axis, relative_distance, 'k-')
    ax_dist.axhline(CAPTURE_RADIUS, color='r', linestyle='--', label=f'Capture Radius ({CAPTURE_RADIUS}m)')
    if capture_step > 0: ax_dist.plot(time_axis[capture_step], relative_distance[capture_step], 'rx', markersize=10, mew=2, label='Capture')
    ax_dist.set_xlabel('Time [s]')
    ax_dist.set_ylabel('Relative Distance [m]')
    ax_dist.set_title('Relative Distance Over Time')
    ax_dist.legend()
    ax_dist.grid(True)
    fig_dist.tight_layout()

    # Control Input Plot
    fig_ctrl, ax_ctrl = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['ax (Radial)', 'ay (Along-track)', 'az (Cross-track)']
    colors = ['r', 'g', 'b']
    if len(control_history) > 0:
        for i in range(3):
            ax_ctrl[i].plot(control_time_axis, control_history[:, i], f'{colors[i]}-', linewidth=1, label=labels[i])
            ax_ctrl[i].axhline(pursuer_u_bar[i], color=colors[i], linestyle=':', alpha=0.5)
            ax_ctrl[i].axhline(pursuer_u_underline[i], color=colors[i], linestyle=':', alpha=0.5)
            ax_ctrl[i].set_ylabel(f'{labels[i]} [m/s^2]')
            ax_ctrl[i].grid(True)
            ax_ctrl[i].legend(loc='upper right')
        ax_ctrl[-1].set_xlabel('Time [s]')
        fig_ctrl.suptitle('Pursuer Control Input Over Time')
        fig_ctrl.tight_layout(rect=[0, 0.03, 1, 0.97])
    else: print("No control history generated.")

    # Chosen Mode Plot
    fig_mode, ax_mode = plt.subplots(figsize=(10, 6))
    if len(chosen_mode_history) > 0:
        valid_mode_indices = chosen_mode_history != -1
        if np.any(valid_mode_indices):
            ax_mode.plot(control_time_axis[valid_mode_indices], chosen_mode_history[valid_mode_indices],
                         'o', markersize=3, linestyle='-', label='Chosen Mode Index')
            ax_mode.set_xlabel('Time [s]')
            ax_mode.set_ylabel('Chosen Mode Index (0 to 2n-1)')
            ax_mode.set_title('Mode Selection Over Time')
            num_total_modes = 2 * n_eigenvectors_found
            ax_mode.set_ylim(-1, num_total_modes)
            ax_mode.set_yticks(np.arange(0, num_total_modes, 2))
            ax_mode.grid(True, axis='y', linestyle=':')
            ax_mode.legend()
            fig_mode.tight_layout()
        else: print("No valid modes were chosen during the simulation to plot.")
    else: print("No mode history generated to plot.")

    plt.show()
    print("Plots generated.")