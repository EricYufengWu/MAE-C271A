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

def generate_accel_state_w_bias(t, a_true, v_c_prev, p_c_prev, b_bias_run):
    acc_noise = np.random.normal(0, np.sqrt(W_k))
    a_c = a_true + b_bias_run + acc_noise
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
    return x_hat_k_new, P_k_new, x_bar_k

def advance_with_measurement(x_hat_k, P_k, z_k):
    # Predict state
    x_bar_kp1 = Phi_k @ x_hat_k
    M_kp1 = Phi_k @ P_k @ Phi_k.T + Gam_k * W_k * Gam_k.T
    # Advance k
    x_bar_k, M_k = x_bar_kp1, M_kp1
    # Update without new measurement
    P_k_new = np.linalg.inv(np.linalg.inv(M_k) + H_k.T @ np.linalg.inv(V_k) @ H_k)
    x_hat_k_new = x_bar_k + P_k_new @ (H_k.T @ (np.linalg.inv(V_k) @ (z_k - H_k @ x_bar_k)))
    return x_hat_k_new, P_k_new, x_bar_k


# ---------- Simulate the exact stochastic system over 30 sec. ----------
ticks = np.linspace(0, sim_time, int(sim_time / Ts_ac) + 1)


# ---------- Add our Kalman filter implementation ----------
x_hat_k, P_k, _ = advance_without_measurement(x_hat_0, P_0)
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
        x_hat_k, P_k, _ = advance_with_measurement(x_hat_k, P_k, z_k)
    else:
        x_hat_k, P_k, _ = advance_without_measurement(x_hat_k, P_k)
    # Record KF-corrected estimates
    p_hat = p_c + x_hat_k[0, 0]
    v_hat = v_c + x_hat_k[1, 0]
    p_hist.append(p_hat)
    v_hist.append(v_hat)
    x_hat_hist.append(x_hat_k)
    P_hist.append(P_k)
    # Keep integrating from the accel-only states
    p_c_prev, v_c_prev = p_c, v_c


# ---------- Run multiple times to get ensemble statistics ----------
N_runs = 50
N_steps = len(ticks)

# Storage for errors from all runs
e_runs = np.zeros((N_runs, 3, N_steps))   
x_true_runs = np.zeros((N_runs, 3, N_steps))
x_hat_runs = np.zeros((N_runs, 3, N_steps))
innov_runs = np.zeros((N_runs, 2, N_steps))

for l in range(N_runs):
    # Vary truth initial connditions
    p_true0 = np.random.normal(p0_bar, np.sqrt(M0_p))
    v_true0 = np.random.normal(v0_bar, np.sqrt(M0_v))
    b_bias  = np.random.normal(0,      np.sqrt(M0_ba))   # new bias each run
    p_true_l = p_true0
    v_true_l = v_true0
    b_true_l = b_bias
    p_c = p_true0
    v_c = v_true0

    x_hat_k = x_hat_0.copy()
    P_k = P_0.copy()

    gps_flag = False
    for k, t in enumerate(ticks):
        p_true_l, v_true_l, a_true = generate_true_state(t)
        if k == 0:
            p_true_l, v_true_l = p_true0, v_true0
        p_c, v_c, a_c = generate_accel_state_w_bias(t, a_true, v_c, p_c, b_bias)
        if k % gps_ratio == 0 and k != 0:
            gps_flag = True
            z_k = generate_z_k(t, p_c, v_c)
            x_hat_k, P_k, x_bar_k = advance_with_measurement(x_hat_k, P_k, z_k)
        else:
            gps_flag = False
            x_hat_k, P_k, x_bar_k = advance_without_measurement(x_hat_k, P_k)
        # True error state
        delta_x_true = np.array([
            p_true_l - p_c,
            v_true_l - v_c,
            b_true_l
        ])
        # Posteriori estimation error
        e_k = delta_x_true - x_hat_k.flatten()
        e_runs[l, :, k] = e_k
        # Orthogonality checks
        x_true_runs[l, :, k] = delta_x_true
        x_hat_runs[l, :, k]  = x_hat_k.flatten()
        if gps_flag:
            r_k = z_k - H_k @ x_bar_k   # innovation using a priori state
            innov_runs[l,:,k] = r_k.flatten()
        else:
            r_k = np.zeros((2,1))
            innov_runs[l,:,k] = np.nan          
    
# Ensemble average error as a function of time
e_ave = np.mean(e_runs, axis=0)

# Ensemble covariance as a function of time
P_ave = np.zeros((3, 3, N_steps))
for k in range(N_steps):
    diffs = e_runs[:, :, k] - e_ave[:, k]
    P_ave[:, :, k] = (diffs.T @ diffs) / (N_runs - 1)


# ---------- Orthogonality Checks ----------
# pick two measurement times j < k (indices in ticks)
j_idx = gps_ratio      # first GPS measurement
k_idx = 2 * gps_ratio  # second GPS measurement

valid = ~np.isnan(innov_runs[:,0,j_idx]) & ~np.isnan(innov_runs[:,0,k_idx])

C_r = np.zeros((2, 2))
for l in np.where(valid)[0]:
    r_j = innov_runs[l,:,j_idx].reshape(2,1)
    r_k = innov_runs[l,:,k_idx].reshape(2,1)
    C_r += r_k @ r_j.T

C_r /= np.sum(valid)
print("Innovation cross-covariance C_r(k,j) ≈")
print(C_r)
print("Norm =", np.linalg.norm(C_r))


# ---------- Prepare for Plotting ----------
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
# Extract ensemble covariance diagonals
P_p_ave = P_ave[0, 0, :]
P_v_ave = P_ave[1, 1, :]
P_b_ave = P_ave[2, 2, :]

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

ax[0].plot(ticks, P_p, 'b', linewidth=2, label='Filter Variance, Position')
ax[0].plot(ticks, P_p_ave, 'c--', linewidth=1.5, label='Ensemble Variance, Position')
ax[0].set_ylabel('P₁₁ (position var)')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(ticks, P_v, 'g', linewidth=2, label='Filter Variance, Velocity')
ax[1].plot(ticks, P_v_ave, 'm--', linewidth=1.5, label='Ensemble Variance, Velocity')
ax[1].set_ylabel('P₂₂ (velocity var)')
ax[1].legend()
ax[1].grid(True)

ax[2].plot(ticks, P_b, 'r', linewidth=2, label='Filter Variance, Accelerometer Bias')
ax[2].plot(ticks, P_b_ave, 'y--', linewidth=1.5, label='Ensemble Variance, Accelerometer Bias')
ax[2].set_ylabel('P₃₃ (bias var)')
ax[2].set_xlabel('Time (s)')
ax[2].legend()
ax[2].grid(True)

fig.tight_layout()
plt.show()


