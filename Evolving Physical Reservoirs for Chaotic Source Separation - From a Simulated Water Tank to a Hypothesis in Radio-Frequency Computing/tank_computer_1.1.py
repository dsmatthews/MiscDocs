import numpy as np
from scipy.integrate import solve_ivp
import numba
import time
import matplotlib.pyplot as plt

# --- 1. GLOBAL PARAMETERS (Enhanced for a more serious run) ---
# Simulation Parameters
GRID_SIZE = 64
N_TIME_STEPS = 2000
DT = np.float32(0.002)
N_SENSORS = 100
DRIVE_STRENGTH = np.float32(0.01)

# Genetic Algorithm Parameters
POPULATION_SIZE = 20
N_GENERATIONS = 10
N_ELITES = 4
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.1

# Shape Parameters
N_SHAPE_PARAMS = 8
BASE_RADIUS = np.float32(0.6)

# --- 2. CHAOTIC SIGNAL GENERATION ---
def generate_lorenz_attractor(n_steps, dt, initial_state):
    def lorenz_system(t, state, sigma=10, rho=28, beta=8/3):
        x, y, z = state; return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    t_span = [0, n_steps * dt]; t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    sol = solve_ivp(lorenz_system, t_span, initial_state, t_eval=t_eval)
    data = sol.y.T
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# --- 3. TANK SHAPE AND PHYSICS ---
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
def run_simulation(tank_mask, mixed_signal, input_filters, sensor_coords):
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
        
        perturbation = np.zeros((grid_size, grid_size), dtype=numba.float32)
        for i in range(mixed_signal.shape[1]):
            perturbation += input_filters[i] * mixed_signal[t, i]

        h_new = h - DT * (h * (du_dx + dv_dy) + u * dh_dx + v * dh_dy - DRIVE_STRENGTH * perturbation)
        u_new = u - DT * (u * du_dx + g * dh_dx + b * u)
        v_new = v - DT * (v * dv_dy + g * dh_dy + b * v)
        h=h_new*tank_mask; u=u_new*tank_mask; v=v_new*tank_mask

        for i in range(N_SENSORS):
            h_history[t, i] = h[sensor_coords[i, 0], sensor_coords[i, 1]]
    return h_history

# --- 4. LEARNING AND EVALUATION ---
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

def evaluate_fitness(shape_params, mixed_signal_f32, s_a, s_b, input_filters, sensor_coords):
    tank_mask = create_tank_mask(shape_params, GRID_SIZE)
    if tank_mask.sum() < (GRID_SIZE * GRID_SIZE / 10): return 1e6
    h_history = run_simulation(tank_mask, mixed_signal_f32, input_filters, sensor_coords)
    return train_and_evaluate(h_history, s_a, s_b)

# --- 5. THE GENETIC ALGORITHM ---
def initialize_population():
    return [(np.random.randn(N_SHAPE_PARAMS) * 0.1).astype(np.float32) for _ in range(POPULATION_SIZE)]
def crossover(p1, p2): return ((p1 + p2) / np.float32(2.0))
def mutate(ind):
    mutated = ind.copy()
    for i in range(len(mutated)):
        if np.random.rand() < MUTATION_RATE: mutated[i] += np.random.randn() * MUTATION_STRENGTH
    return mutated.astype(np.float32)

# --- 6. VISUALIZATION ---
def visualize_and_save_shape(shape_params, filename="optimized_tank.png"):
    """Creates a PNG image of the final tank shape."""
    mask = create_tank_mask(shape_params, GRID_SIZE)
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray', origin='lower')
    plt.title("Optimized Tank Shape")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)
    print(f"\nFinal tank shape saved to {filename}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- Chaotic Source Separation Tank Optimizer ---")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, Time Steps: {N_TIME_STEPS}")
    print(f"GA: {POPULATION_SIZE} individuals, {N_GENERATIONS} generations")
    print("\nSetting up the experiment...")

    s_a = generate_lorenz_attractor(N_TIME_STEPS, dt=DT*5, initial_state=[0., 1., 1.05])
    s_b = generate_lorenz_attractor(N_TIME_STEPS, dt=DT*5, initial_state=[0., 1., 1.06])
    mixed_signal_f32 = (s_a + s_b).astype(np.float32)
    
    n_dims = s_a.shape[1]
    input_filters = [(np.random.randn(GRID_SIZE,GRID_SIZE)).astype(np.float32) for _ in range(n_dims)]
    input_filters = np.array([f - f.mean() for f in input_filters])
    sensor_coords = np.random.randint(0, GRID_SIZE, size=(N_SENSORS, 2))

    population = initialize_population()
    best_fitness_overall = float('inf')
    best_shape_overall = None
    
    print("Starting evolution...")
    start_time = time.time()
    
    for gen in range(N_GENERATIONS):
        gen_start_time = time.time()
        fitness_scores = [evaluate_fitness(ind, mixed_signal_f32, s_a, s_b, input_filters, sensor_coords) for ind in population]
        
        sorted_indices = np.argsort(fitness_scores)
        population = [population[i] for i in sorted_indices]; fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        if fitness_scores[0] < best_fitness_overall:
            best_fitness_overall = fitness_scores[0]
            best_shape_overall = population[0]

        print(f"Gen {gen+1:2d}/{N_GENERATIONS} | Best MSE: {best_fitness_overall:.6f} | Time: {time.time()-gen_start_time:.2f}s")

        elites = population[:N_ELITES]
        next_population = elites[:]
        
        # --- BUG FIX START ---
        # Select parents by index, not by passing the whole list of arrays
        while len(next_population) < POPULATION_SIZE:
            # Choose two different indices from the elite pool
            idx1, idx2 = np.random.choice(N_ELITES, 2, replace=False)
            parent1 = elites[idx1]
            parent2 = elites[idx2]
            
            child = crossover(parent1, parent2)
            next_population.append(mutate(child))
        # --- BUG FIX END ---
        
        population = next_population

    print("\n--- Evolution Complete ---")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Best overall MSE found: {best_fitness_overall:.6f}")
    print("Best shape parameters found:\n", best_shape_overall)
    
    # Visualize the final best shape
    if best_shape_overall is not None:
        visualize_and_save_shape(best_shape_overall)
    else:
        print("\nNo stable shape found to visualize.")