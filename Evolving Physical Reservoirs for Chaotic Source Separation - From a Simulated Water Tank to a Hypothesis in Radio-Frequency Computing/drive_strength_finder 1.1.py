import numpy as np
from scipy.integrate import solve_ivp
import numba
import time

# --- All core functions are the same except run_simulation ---
# --- I've included them here for a complete, runnable file ---

GRID_SIZE = 64
N_TIME_STEPS = 2000
DT = np.float32(0.002)
N_SENSORS = 100
BASE_RADIUS = np.float32(0.6)
N_SHAPE_PARAMS = 8

# --- KEY NEW PARAMETER ---
# This is the diffusion coefficient. A small value to calm the simulation.
DIFFUSION_COEFFICIENT = np.float32(0.0001)

def generate_lorenz_attractor(n_steps, dt, initial_state):
    def lorenz_system(t, state, sigma=10, rho=28, beta=8/3):
        x, y, z = state; return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    t_span = [0, n_steps * dt]; t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    sol = solve_ivp(lorenz_system, t_span, initial_state, t_eval=t_eval)
    data = sol.y.T
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

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
def run_simulation(tank_mask, mixed_signal, input_filters, sensor_coords, drive_strength):
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
        
        # --- STABILITY FIX: Calculate the Laplacian for diffusion ---
        laplacian_u = (u_ip1 + u_im1 + v_jp1 + v_jm1 - 4 * u)
        laplacian_v = (manual_roll(v,-1,0) + manual_roll(v,1,0) + v_jp1 + v_jm1 - 4 * v)
        
        perturbation = np.zeros((grid_size, grid_size), dtype=numba.float32)
        for i in range(mixed_signal.shape[1]):
            perturbation += input_filters[i] * mixed_signal[t, i]
            
        h_new = h - DT * (h * (du_dx + dv_dy) + u * dh_dx + v * dh_dy - drive_strength * perturbation)
        
        # --- STABILITY FIX: Add the diffusion term to the velocity updates ---
        u_new = u - DT * (u * du_dx + g * dh_dx + b * u) + DIFFUSION_COEFFICIENT * laplacian_u
        v_new = v - DT * (v * dv_dy + g * dh_dy + b * v) + DIFFUSION_COEFFICIENT * laplacian_v

        h=h_new*tank_mask; u=u_new*tank_mask; v=v_new*tank_mask
        for i in range(N_SENSORS):
            h_history[t, i] = h[sensor_coords[i, 0], sensor_coords[i, 1]]
    return h_history

def train_and_evaluate(h_history, true_s_a, true_s_b):
    if not np.all(np.isfinite(h_history)): return 1e6
    h_dev = h_history - 1.0 
    X = np.hstack([h_dev, np.tanh(h_dev**2), np.tanh(h_dev**3)])
    Y = np.hstack([true_s_a, true_s_b])
    try:
        W, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError: return 1e6
    Y_pred = X @ W
    if Y_pred.shape != Y.shape: return 1e6
    mse = np.mean((Y - Y_pred)**2)
    return mse if np.isfinite(mse) else 1e6

# --- MAIN DIAGNOSTIC SCRIPT ---
if __name__ == "__main__":
    print("--- Drive Strength Diagnostic Tool (with Stabilized Physics) ---")
    
    # Setup fixed signals and filters
    s_a = generate_lorenz_attractor(N_TIME_STEPS, dt=DT*5, initial_state=[0., 1., 1.05])
    s_b = generate_lorenz_attractor(N_TIME_STEPS, dt=DT*5, initial_state=[0., 1., 1.06])
    mixed_signal_f32 = (s_a + s_b).astype(np.float32)
    n_dims = s_a.shape[1]
    input_filters = np.array([(np.random.randn(GRID_SIZE,GRID_SIZE) - f.mean()).astype(np.float32) for f in np.random.randn(n_dims, GRID_SIZE, GRID_SIZE)])
    sensor_coords = np.random.randint(0, GRID_SIZE, size=(N_SENSORS, 2))

    circular_shape_params = np.zeros(N_SHAPE_PARAMS, dtype=np.float32)
    tank_mask = create_tank_mask(circular_shape_params, GRID_SIZE)
    
    # We will test the same range again. Now we expect to see different results.
    drive_strengths_to_test = [1.0, 0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    print("\nTesting a simple circular tank with different drive strengths...")
    print("----------------------------------------------------------")
    
    results = {}
    for strength in drive_strengths_to_test:
        print(f"Testing DRIVE_STRENGTH = {strength:.4f}...", end='', flush=True)
        start_time = time.time()
        
        h_history = run_simulation(tank_mask, mixed_signal_f32, input_filters, sensor_coords, np.float32(strength))
        mse = train_and_evaluate(h_history, s_a, s_b)
        results[strength] = mse
        
        print(f" done. | MSE = {mse:.6f} | Time = {time.time() - start_time:.2f}s")

    print("----------------------------------------------------------")
    print("\nSweep Complete. Results:")
    for strength, mse in results.items():
        print(f"DRIVE_STRENGTH: {strength:<10.4f} -> MSE: {mse:.6f}")
        
    best_strength = min(results, key=results.get)
    best_mse = results[best_strength]
    
    print("\n--- Recommendation ---")
    if best_mse >= 0.99:
         print("All tested strengths still resulted in a poor MSE (>= 0.99).")
         print("This could mean the signal is too weak OR the diffusion is too strong.")
         print("Try increasing drive strength values or decreasing DIFFUSION_COEFFICIENT.")
    else:
        print(f"SUCCESS! The best performing value is DRIVE_STRENGTH = {best_strength}.")
        print("This is the 'sweet spot'. Update this value in the main GA script to proceed.")