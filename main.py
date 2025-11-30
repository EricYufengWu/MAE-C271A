import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18 

#---------- Parameter from project prompt ----------
# True acceleration: a(t) = a * sin (w*t)
a = 10          # Ampliture
w = 0.2         # rad/sec
sim_time = 30   # s

# Accelarometer noise parameters
a0_ac = 0       # m/s^2
W_k = 0.0004     # (m/s^2)^2
# Accelerometer bias parameters
a0_ba = 0       # m/s^2
M0_ba = 0.01    # (m/s^2)^2
b_bias = np.random.normal(0, np.sqrt(M0_ba))
# Accelerometer sample rate
Ts_ac = 0.005   # s^(-1), 200 Hz
delta_t = Ts_ac

# Initial conditions parameters
v0_bar = 100    # m/s
M0_v = 1        # (m/s)^2
p0_bar = 0      # m
M0_p = 100      # m^2

# GPS measurement parameters
V_p = 1         # m^2
V_v = 0.0016    # (m/s)^2
# GPS sample rate
Ts_gps = 0.2    # s^(-1), 5 Hz
gps_ratio = int(Ts_gps / Ts_ac)


# ---------- Construct the numerical terms ----------
Phi_k = np.array([[1, delta_t, -0.5 * delta_t**2],
                  [0, 1, -delta_t],
                  [0, 0, 1]])
Gam_k = np.array([[0.5 * delta_t**2],
                  [delta_t],
                  [0]])
H_k = np.array([[1, 0, 0],
                [0, 1, 0]])
V_k = np.array([[V_p, 0],
                [0, V_v]])
x_hat_0 = np.array([[0], [0], [0]])
M_0 = np.array([[M0_p, 0, 0],
                [0, M0_v, 0],
                [0, 0, M0_ba]])
P_0 = M_0


# ---------- Kalman filter iteration functions ----------
def generate_true_state(t):
    v_true = v0_bar + (a / w) - (a / w) * np.cos(w * t)
    p_true = p0_bar + (v0_bar + a / w) * t - (a / w**2) * np.sin(w * t)
    a_true = a * np.sin(w * t)
    return p_true, v_true, a_true

def generate_accel_state(t, a_true, v_c_prev, p_c_prev):
    acc_noise = np.random.normal(0, np.sqrt(W_k))
    a_c = a_true + b_bias + acc_noise
    v_c = v_c_prev + a_c * delta_t
    p_c = p_c_prev + v_c_prev * delta_t + 0.5 * a_c * delta_t**2
    return p_c, v_c, a_c

def generate_z_k(t, p_c, v_c):
    p_t, v_t, _ = generate_true_state(t)
    p_gps = p_t + np.random.normal(0, np.sqrt(V_p))
    v_gps = v_t + np.random.normal(0, np.sqrt(V_v))
    z_k = np.array([[p_gps - p_c],
                    [v_gps - v_c]])
    return z_k

def advance_without_measurement(x_hat_k, P_k):
    # Predict state
    x_bar_kp1 = Phi_k @ x_hat_k
    M_kp1 = Phi_k @ P_k @ Phi_k.T + Gam_k * W_k * Gam_k.T
    # Advance k
    x_bar_k, M_k = x_bar_kp1, M_kp1
    # Update without new measurement
    P_k_new = M_k
    x_hat_k_new = x_bar_k
    return x_hat_k_new, P_k_new

def advance_with_measurement(x_hat_k, P_k, z_k):
    # Predict state
    x_bar_kp1 = Phi_k @ x_hat_k
    M_kp1 = Phi_k @ P_k @ Phi_k.T + Gam_k * W_k * Gam_k.T
    # Advance k
    x_bar_k, M_k = x_bar_kp1, M_kp1
    # Update without new measurement
    P_k_new = np.linalg.inv(np.linalg.inv(M_k) + H_k.T @ np.linalg.inv(V_k) @ H_k)
    x_hat_k_new = x_bar_k + P_k_new @ (H_k.T @ (np.linalg.inv(V_k) @ (z_k - H_k @ x_bar_k)))
    return x_hat_k_new, P_k_new


# ---------- Simulate the exact stochastic system over 30 sec. ----------
ticks = np.linspace(0, sim_time, int(sim_time / Ts_ac) + 1)


# ---------- Add our Kalman filter implementation ----------
x_hat_k, P_k = advance_without_measurement(x_hat_0, P_0)
v_c_prev, p_c_prev = v0_bar, p0_bar
p_true, v_true, a_true = [], [], []
p_acc, v_acc, a_acc = [], [], []
p_hist, v_hist = [], []
x_hat_hist, P_hist = [], []
for n, j in enumerate(ticks):
    # print(f"Time step {n}, Time {j:.3f} s")
    p_t, v_t, a_t = generate_true_state(j)
    p_true.append(p_t)
    v_true.append(v_t)
    a_true.append(a_t)
    p_c, v_c, a_c = generate_accel_state(j, a_t, v_c_prev, p_c_prev)
    p_acc.append(p_c)
    v_acc.append(v_c)
    a_acc.append(a_c)
    if n % gps_ratio == 0 and n != 0:
        z_k = generate_z_k(j, p_c, v_c)
        # print(f"GPS update at time {j:.3f} s: p = {z_k[0][0]:.3f} m, v = {z_k[1][0]:.3f} m/s")
        x_hat_k, P_k = advance_with_measurement(x_hat_k, P_k, z_k)
    else:
        x_hat_k, P_k = advance_without_measurement(x_hat_k, P_k)
    # Record KF-corrected estimates
    p_hat = p_c + x_hat_k[0, 0]
    v_hat = v_c + x_hat_k[1, 0]
    p_hist.append(p_hat)
    v_hist.append(v_hat)
    x_hat_hist.append(x_hat_k)
    P_hist.append(P_k)
    # Keep integrating from the accel-only states
    p_c_prev, v_c_prev = p_c, v_c


# # ---------- Plot true results ----------
# fig, axs = plt.subplots(3, 1, figsize=(10, 15))
# axs[0].plot(ticks, p_true, label='True Position', linewidth=0.7)
# axs[0].plot(ticks, p_acc, label='Accel-only Position', linewidth=0.7)
# axs[0].plot(ticks, p_hist, label='KF-corrected Position', linewidth=0.7)
# axs[1].plot(ticks, v_true, label='True Velocity', linewidth=0.7)
# axs[1].plot(ticks, v_acc, label='Accel-only Velocity', linewidth=0.7)
# axs[1].plot(ticks, v_hist, label='KF-corrected Velocity', linewidth=0.7)
# axs[2].plot(ticks, a_true, label='True Acceleration', linewidth=0.7)
# axs[2].plot(ticks, a_acc, label='Accel-only Acceleration', linewidth=0.7)
# axs[0].set_ylabel('Position (m)')
# axs[1].set_ylabel('Velocity (m/s)')
# axs[2].set_ylabel('Acceleration (m/s²)')
# axs[2].set_xlabel('Time (s)')
# axs[0].legend()
# axs[1].legend()
# axs[2].legend()
# fig.tight_layout()
# plt.show()

# Convert lists to arrays for slicing
p_true = np.array(p_true)
v_true = np.array(v_true)
p_hist = np.array(p_hist)
v_hist = np.array(v_hist)
x_hat_history = np.array(x_hat_hist)       # shape (N, 3)
P_history = np.array(P_hist)               # shape (N, 3, 3)

# Extract bias estimate separately
b_hat = x_hat_history[:, 2, 0] 

# Extract covariance diagonals
P_p = P_history[:, 0, 0]
P_v = P_history[:, 1, 1]
P_b = P_history[:, 2, 2]

# ----- Figure 1: Position estimate & error with 1-sigma bounds -----
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Top plot: truth and estimate
ax[0].plot(ticks, p_true, 'k', label='True Position')
ax[0].plot(ticks, p_acc, 'c', linewidth=0.7, label='Extimated Position (uncorrected)')
ax[0].plot(ticks, p_hist, 'm', linewidth=0.7, label='Estimated Position (KF)')
# ax[0].plot(ticks, p_hist + np.sqrt(P_p), 'r--', alpha=0.6, label='+1σ Bound')
# ax[0].plot(ticks, p_hist - np.sqrt(P_p), 'r--', alpha=0.6)
ax[0].set_ylabel('Position (m)')
ax[0].legend()
ax[0].grid(True)

# Bottom plot: position estimation error
ax[1].plot(ticks, p_hist - p_true, 'g', label='Position Error')
ax[1].plot(ticks, np.sqrt(P_p), 'r--', alpha=0.6, label='±1σ Bound')
ax[1].plot(ticks, -np.sqrt(P_p), 'r--', alpha=0.6)
ax[1].set_ylabel('Error (m)')
ax[1].set_xlabel('Time (s)')
ax[1].legend()
ax[1].grid(True)

fig.tight_layout()
plt.show()


# ----- Figure 2: Velocity estimate & error with 1-sigma bounds -----
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Top plot
ax[0].plot(ticks, v_true, 'k', label='True Velocity')
ax[0].plot(ticks, v_acc, 'c', linewidth=0.7, label='Extimated Velocity (uncorrected)')
ax[0].plot(ticks, v_hist, 'm', linewidth=0.7, label='Estimated Velocity (KF)')
# ax[0].plot(ticks, v_hist + np.sqrt(P_v), 'r--', alpha=0.6, label='+1σ Bound')
# ax[0].plot(ticks, v_hist - np.sqrt(P_v), 'r--', alpha=0.6)
ax[0].set_ylabel('Velocity (m/s)')
ax[0].legend()
ax[0].grid(True)

# Bottom plot
ax[1].plot(ticks, v_hist - v_true, 'g', label='Velocity Error')
ax[1].plot(ticks, np.sqrt(P_v), 'r--', alpha=0.6, label='±1σ Bound')
ax[1].plot(ticks, -np.sqrt(P_v), 'r--', alpha=0.6)
ax[1].set_ylabel('Error (m/s)')
ax[1].set_xlabel('Time (s)')
ax[1].legend()
ax[1].grid(True)

fig.tight_layout()
plt.show()


# ----- Figure 3: Bias estimate & error with 1-sigma bounds -----
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

true_bias = b_bias

# TOP: Bias estimate
ax[0].axhline(true_bias, color='k', label='True Bias')
ax[0].plot(ticks, b_hat, 'b', label='Estimated Bias (KF)')
ax[0].plot(ticks, b_hat + np.sqrt(P_b), 'r--', label='+1σ Bound')
ax[0].plot(ticks, b_hat - np.sqrt(P_b), 'r--')
ax[0].set_ylabel('Bias (m/s²)')
ax[0].legend()
ax[0].grid(True)

# BOTTOM: Bias error
bias_error = b_hat - true_bias
ax[1].plot(ticks, bias_error, 'g', label='Bias Error')
ax[1].plot(ticks, +np.sqrt(P_b), 'r--')
ax[1].plot(ticks, -np.sqrt(P_b), 'r--')
ax[1].set_ylabel('Error (m/s²)')
ax[1].set_xlabel('Time (s)')
ax[1].legend()
ax[1].grid(True)

fig.tight_layout()
plt.show()


# ----- Figure 4: Covariance evolution -----
fig, ax = plt.subplots(3, 1, figsize=(10, 12))

ax[0].plot(ticks, P_p, 'b')
ax[0].set_ylabel('P₁₁ (position var)')
ax[0].grid(True)

ax[1].plot(ticks, P_v, 'g')
ax[1].set_ylabel('P₂₂ (velocity var)')
ax[1].grid(True)

ax[2].plot(ticks, P_b, 'r')
ax[2].set_ylabel('P₃₃ (bias var)')
ax[2].set_xlabel('Time (s)')
ax[2].grid(True)

fig.tight_layout()
plt.show()
