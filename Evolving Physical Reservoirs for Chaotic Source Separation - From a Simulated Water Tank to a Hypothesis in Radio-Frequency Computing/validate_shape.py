import numpy as np
from scipy.integrate import solve_ivp
import numba
import time
import matplotlib.pyplot as plt

# --- 1. GLOBAL PARAMETERS - These must match the production run ---
GRID_SIZE = 64
N_TIME_STEPS = 5000
DT = np.float32(0.002)
N_SENSORS = 100
DRIVE_STRENGTH = np.float32(0.01)
DIFFUSION_COEFFICIENT = np.float32(0.01)
N_SHAPE_PARAMS = 8
BASE_RADIUS = np.float32(0.6)

# --- PASTE THE WINNING PARAMETERS FROM THE LAST RUN HERE ---
OPTIMIZED_SHAPE_PARAMS = np.array([
    -0.08783984, -0.35238224, -0.03142271, -0.3044501,  -0.03095008, -0.33157533,
     0.1114569,   0.15349805
], dtype=np.float32)
# ---

BASELINE_SHAPE_PARAMS = np.zeros(N_SHAPE_PARAMS, dtype=np.float32)

# --- 2. SIGNAL GENERATORS (Expanded Suite) ---
def normalize_signal(data):
    """Pre-processes any signal to have zero mean and unit variance."""
    # Ensure data is 2D
    if data.ndim == 1:
        data = data[:, np.newaxis]
    # Pad to 3 dimensions if needed, for consistency
    if data.shape[1] < 3:
        padding = np.zeros((data.shape[0], 3 - data.shape[1]))
        data = np.hstack([data, padding])
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def generate_lorenz(n_steps, dt, initial_state=[0., 1., 1.05]):
    def lorenz_system(t, state, sigma=10, rho=28, beta=8/3):
        x, y, z = state; return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    t_span = [0, n_steps * dt]; t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    sol = solve_ivp(lorenz_system, t_span, initial_state, t_eval=t_eval)
    return normalize_signal(sol.y.T)

def generate_rossler(n_steps, dt, initial_state=[0.1, 0.1, 0.1]):
    def rossler_system(t, state, a=0.2, b=0.2, c=5.7):
        x, y, z = state; return [-y - z, x + a * y, b + z * (x - c)]
    t_span = [0, n_steps * dt]; t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    sol = solve_ivp(rossler_system, t_span, initial_state, t_eval=t_eval)
    return normalize_signal(sol.y.T)
    
def generate_sine_wave(n_steps, dt):
    t = np.linspace(0, n_steps * dt, n_steps)
    # Different frequencies for the 3 dimensions
    s = np.c_[np.sin(t), np.sin(2.1*t + 1), np.sin(3.3*t + 2)]
    return normalize_signal(s)

def generate_white_noise(n_steps):
    return normalize_signal(np.random.randn(n_steps, 3))


# --- 3. CORE SIMULATION AND EVALUATION FUNCTIONS (Unchanged) ---
@numba.jit(nopython=True)
def create_tank_mask(shape_params, grid_size):
    mask = np.zeros((grid_size, grid_size), dtype=numba.boolean)
    center = grid_size / 2.0
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i - center) / center; y = (j - center) / center
            r = np.sqrt(x**2 + y**2); theta = np.arctan2(y, x)
            boundary_r = BASE_RADIUS
            for n in range(N_SHAPE_PARAMS // 2):
                boundary_r += shape_params[2*n] * np.cos((n + 1) * theta)
                boundary_r += shape_params[2*n+1] * np.sin((n + 1) * theta)
            if r < boundary_r: mask[i, j] = True
    return mask

@numba.jit(nopython=True)
def manual_roll(arr, shift, axis):
    new_arr = np.empty_like(arr)
    if axis == 0:
        for i in range(arr.shape[0]): new_arr[i] = arr[(i - shift) % arr.shape[0]]
    elif axis == 1:
        for j in range(arr.shape[1]): new_arr[:, j] = arr[:, (j - shift) % arr.shape[1]]
    return new_arr

@numba.jit(nopython=True)
def run_simulation(tank_mask, mixed_signal, input_filters, sensor_coords, drive_strength, diffusion_coeff):
    grid_size = tank_mask.shape[0]; n_steps = mixed_signal.shape[0]
    h = np.ones((grid_size, grid_size), dtype=numba.float32)
    u = np.zeros((grid_size, grid_size), dtype=numba.float32)
    v = np.zeros((grid_size, grid_size), dtype=numba.float32)
    g = np.float32(9.8); b = np.float32(0.3); TWO = np.float32(2.0)
    h_history = np.zeros((n_steps, N_SENSORS), dtype=numba.float32)
    for t in range(n_steps):
        u_ip1=manual_roll(u,-1,0); u_im1=manual_roll(u,1,0); v_jp1=manual_roll(v,-1,1); v_jm1=manual_roll(v,1,1)
        h_ip1=manual_roll(h,-1,0); h_im1=manual_roll(h,1,0); h_jp1=manual_roll(h,-1,1); h_jm1=manual_roll(h,1,1)
        dh_dx=(h_ip1-h_im1)/TWO; dh_dy=(h_jp1-h_jm1)/TWO
        du_dx=(u_ip1-u_im1)/TWO; dv_dy=(v_jp1-v_jm1)/TWO
        laplacian_u = (u_ip1 + u_im1 + manual_roll(u,-1,1) + manual_roll(u,1,1) - 4 * u)
        laplacian_v = (manual_roll(v,-1,0) + manual_roll(v,1,0) + v_jp1 + v_jm1 - 4 * v)
        perturbation = np.zeros((grid_size, grid_size), dtype=numba.float32)
        for i in range(mixed_signal.shape[1]):
            perturbation += input_filters[i] * mixed_signal[t, i]
        h_new = h - DT * (h * (du_dx + dv_dy) + u * dh_dx + v * dh_dy - drive_strength * perturbation)
        u_new = u - DT * (u * du_dx + g * dh_dx + b * u) + diffusion_coeff * laplacian_u
        v_new = v - DT * (v * dv_dy + g * dh_dy + b * v) + diffusion_coeff * laplacian_v
        h=h_new*tank_mask; u=u_new*tank_mask; v=v_new*tank_mask
        for i in range(N_SENSORS):
            h_history[t, i] = h[sensor_coords[i, 0], sensor_coords[i, 1]]
    return h_history

def train_and_evaluate(h_history, true_s_a, true_s_b):
    if not np.all(np.isfinite(h_history)): return 1e6
    h_dev = h_history - 1.0
    X = np.hstack([h_dev, np.tanh(h_dev**2), np.tanh(h_dev**3)])
    Y = np.hstack([true_s_a, true_s_b])
    try: W, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError: return 1e6
    Y_pred = X @ W
    if Y_pred.shape != Y.shape: return 1e6
    mse = np.mean((Y - Y_pred)**2)
    return mse if np.isfinite(mse) else 1e6

# --- MAIN VALIDATION SCRIPT ---
if __name__ == "__main__":
    print("--- Shape Generalizability Validation Tool ---")

    # Define the test suite of signal pairs
    test_suite = {
        "Lorenz + Lorenz (Control)": (
            generate_lorenz(N_TIME_STEPS, dt=DT*5, initial_state=[0., 1., 1.05]),
            generate_lorenz(N_TIME_STEPS, dt=DT*5, initial_state=[0., 1., 1.06])
        ),
        "Lorenz + Rössler": (
            generate_lorenz(N_TIME_STEPS, dt=DT*5, initial_state=[0., 1., 1.05]),
            generate_rossler(N_TIME_STEPS, dt=DT*5)
        ),
        "Sine Wave + White Noise": (
            generate_sine_wave(N_TIME_STEPS, dt=DT*5),
            generate_white_noise(N_TIME_STEPS)
        ),
    }

    results = {}
    
    for test_name, (s_a, s_b) in test_suite.items():
        print(f"\n--- Testing Pair: {test_name} ---")
        
        mixed_signal_f32 = (s_a + s_b).astype(np.float32)
        n_dims = s_a.shape[1]
        input_filters = np.array([(np.random.randn(GRID_SIZE,GRID_SIZE) - f.mean()).astype(np.float32) for f in np.random.randn(n_dims, GRID_SIZE, GRID_SIZE)])
        sensor_coords = np.random.randint(0, GRID_SIZE, size=(N_SENSORS, 2))
        
        test_results = {}
        
        # Test 1: The Optimized Shape
        print("Testing OPTIMIZED shape...", end='', flush=True)
        opt_mask = create_tank_mask(OPTIMIZED_SHAPE_PARAMS, GRID_SIZE)
        opt_history = run_simulation(opt_mask, mixed_signal_f32, input_filters, sensor_coords, DRIVE_STRENGTH, DIFFUSION_COEFFICIENT)
        opt_mse = train_and_evaluate(opt_history, s_a, s_b)
        test_results["Optimized"] = opt_mse
        print(f" done. | MSE = {opt_mse:.6f}")

        # Test 2: The Baseline Circular Shape
        print("Testing BASELINE (circle) shape...", end='', flush=True)
        base_mask = create_tank_mask(BASELINE_SHAPE_PARAMS, GRID_SIZE)
        base_history = run_simulation(base_mask, mixed_signal_f32, input_filters, sensor_coords, DRIVE_STRENGTH, DIFFUSION_COEFFICIENT)
        base_mse = train_and_evaluate(base_history, s_a, s_b)
        test_results["Baseline"] = base_mse
        print(f" done. | MSE = {base_mse:.6f}")

        results[test_name] = test_results

    print("\n\n--- Final Validation Report ---")
    print("Lower MSE is better.")
    print("-" * 50)
    for test_name, res in results.items():
        opt_mse = res["Optimized"]
        base_mse = res["Baseline"]
        print(f"TASK: {test_name}")
        print(f"  > Optimized Shape MSE: {opt_mse:.6f}")
        print(f"  > Baseline Shape MSE:  {base_mse:.6f}")
        
        if opt_mse < base_mse * 0.95: # 5% better
            improvement = 100 * (base_mse - opt_mse) / base_mse
            print(f"  > VERDICT: SUCCESS! Optimized shape is {improvement:.1f}% better.")
        elif base_mse < opt_mse * 0.95:
            print(f"  > VERDICT: SPECIALIZED. Baseline shape performed better.")
        else:
            print(f"  > VERDICT: NO DIFFERENCE. Performance is comparable.")
        print("-" * 50)