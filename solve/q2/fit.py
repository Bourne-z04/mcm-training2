import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# 1. Load Data and Time Estimation
def load_trajectory_data(filepath):
    """Load trajectory data and estimate time points"""
    df = pd.read_csv(filepath)
    x_data = df['x'].values
    y_data = df['y'].values
    z_data = df['z'].values

    # Estimate time based on trajectory characteristics
    # Method 1: Based on average velocity (more robust)
    total_distance = np.linalg.norm([x_data[-1] - x_data[0],
                                   y_data[-1] - y_data[0],
                                   z_data[-1] - z_data[0]])
    n_points = len(x_data)

    # Assume typical football flight time for this distance (20-25m in ~1-1.2s)
    # Based on Ronaldo's free kick characteristics
    estimated_flight_time = 1.0  # seconds

    # Alternative: estimate from vertical motion
    # z_max_idx = np.argmax(z_data)
    # if z_max_idx > 0 and z_max_idx < len(z_data)-1:
    #     z_max = z_data[z_max_idx]
    #     # Time to apex: t = sqrt(2*z_max/g)
    #     t_to_apex = np.sqrt(2 * z_max / 9.81)
    #     estimated_flight_time = 2 * t_to_apex

    dt = estimated_flight_time / (n_points - 1)
    t_data = np.arange(n_points) * dt

    return t_data, x_data, y_data, z_data

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'optimized_trajectory.csv')
t_data, x_data, y_data, z_data = load_trajectory_data(data_path)

# 2. Physics Constants (Standard Football Values)
m = 0.436  # kg (FIFA standard)
r = 0.11   # m (ball radius)
rho = 1.225  # kg/m³ (air density at 15°C)
g = 9.81   # m/s² (gravity)
mu = 1.8e-5  # Pa·s (dynamic viscosity)

# Aerodynamic parameters
S = np.pi * r**2  # cross-sectional area
V_vol = 4/3 * np.pi * r**3  # volume

# Magnus force coefficient
C_M = 0.2

# Pre-computed coefficients
alpha_base = rho * S / (2 * m)
beta = rho * V_vol * C_M / (2 * m)

def get_drag_coefficient(v_magnitude, model_type):
    """
    Get drag coefficient based on velocity and model type
    """
    Re = rho * v_magnitude * 2 * r / mu  # Reynolds number

    if model_type == 'Elevator':
        # Knuckleball: variable Cd due to turbulent transition
        Cd_min = 0.1   # laminar
        Cd_max = 0.5   # turbulent
        Re_crit = 1e5  # critical Reynolds number
        width = 0.5 * Re_crit
        transition = 1 / (1 + np.exp((Re - Re_crit) / width))
        return Cd_min + (Cd_max - Cd_min) * transition
    else:
        # Standard drag coefficient for smooth ball
        if Re < 1e5:
            return 0.5  # subcritical
        else:
            return 0.25  # supercritical

def ball_dynamics(t, state, model_type, omega_params):
    """
    Ball dynamics with improved Magnus force modeling

    Parameters:
    state: [x, y, z, vx, vy, vz]
    model_type: 'Banana', 'Leaf', or 'Elevator'
    omega_params: rotation parameters (different for each model)
    """
    x, y, z, vx, vy, vz = state
    v_vec = np.array([vx, vy, vz])
    v_mag = np.linalg.norm(v_vec)

    if v_mag < 1e-6:
        return [vx, vy, vz, 0, 0, -g]

    # Get drag coefficient
    Cd = get_drag_coefficient(v_mag, model_type)
    alpha = alpha_base * Cd

    # Drag force
    F_drag = -alpha * v_mag * v_vec

    # Magnus force (depends on model)
    if model_type == 'Banana':
        # Side spin: ω = [0, 0, ω_z]
        omega_val = omega_params[0]
        omega_vec = np.array([0, 0, omega_val])
        # Include spin decay (typical decay rate k=0.05-0.1 s⁻¹)
        omega_decay = omega_val * np.exp(-0.07 * t)
        omega_vec = np.array([0, 0, omega_decay])
        F_magnus = beta * np.cross(omega_vec, v_vec)

    elif model_type == 'Leaf':
        # Top spin: ω = [ω_x, 0, 0]
        omega_val = omega_params[0]
        omega_decay = omega_val * np.exp(-0.07 * t)
        omega_vec = np.array([omega_decay, 0, 0])
        F_magnus = beta * np.cross(omega_vec, v_vec)

        # Additional downward component for leaf ball effect
        # This creates the characteristic "falling leaf" motion
        downward_factor = 0.3 * beta * omega_decay * v_mag
        F_magnus[2] -= downward_factor

    elif model_type == 'Elevator':
        # Low spin: ω = [ω_x, 0, 0] (small)
        omega_val = omega_params[0]
        omega_decay = omega_val * np.exp(-0.05 * t)  # slower decay
        omega_vec = np.array([omega_decay, 0, 0])
        F_magnus = beta * np.cross(omega_vec, v_vec)

    # Gravity
    F_gravity = np.array([0, 0, -m * g])

    # Total force
    F_total = F_drag + F_magnus + F_gravity
    acceleration = F_total / m

    return [vx, vy, vz, acceleration[0], acceleration[1], acceleration[2]]

def objective(params, model_type):
    """
    Objective function for trajectory fitting

    Parameters:
    params: [v0, theta, phi, omega] where:
        v0: initial speed (m/s)
        theta: horizontal angle (rad)
        phi: vertical angle (rad)
        omega: spin rate (rad/s)
    """
    v0, theta, phi, omega = params

    # Convert spherical to Cartesian coordinates
    vx = v0 * np.cos(phi) * np.cos(theta)
    vy = v0 * np.cos(phi) * np.sin(theta)
    vz = v0 * np.sin(phi)

    # Initial state
    state0 = [x_data[0], y_data[0], z_data[0], vx, vy, vz]

    try:
        # Solve ODE
        sol = solve_ivp(ball_dynamics, [0, t_data[-1]], state0,
                       args=(model_type, [omega]), t_eval=t_data,
                       rtol=1e-4, atol=1e-6, method='RK45')

        if not sol.success or len(sol.t) != len(t_data):
            return 1e9

        # Calculate weighted MSE loss
        # Weight z-coordinate more heavily as it's more important for goal analysis
        x_error = sol.y[0] - x_data
        y_error = sol.y[1] - y_data
        z_error = sol.y[2] - z_data

        # Weights: emphasize accuracy in goal direction (x) and height (z)
        weights = np.array([2.0, 1.0, 3.0])  # x, y, z weights
        weighted_errors = np.array([x_error, y_error, z_error]) * weights[:, np.newaxis]

        mse_loss = np.mean(np.sum(weighted_errors**2, axis=0))

        return mse_loss

    except Exception as e:
        return 1e9

# Models Configuration with Realistic Bounds
models_config = {
    'Banana': {
        'bounds': [
            (25, 35),      # v0: high speed for banana kick
            (-0.3, 0.3),   # theta: small horizontal angle
            (0.1, 0.5),    # phi: moderate launch angle
            (100, 200)     # omega: high side spin
        ],
        'initial_guess': [30, 0.0, 0.3, 150],
        'label': 'Banana (Side Spin)',
        'description': 'High side spin causing horizontal curve'
    },
    'Leaf': {
        'bounds': [
            (20, 32),      # v0: moderate speed
            (-0.2, 0.2),   # theta: small horizontal angle
            (0.2, 0.6),    # phi: higher launch angle
            (50, 120)      # omega: moderate top spin
        ],
        'initial_guess': [26, 0.0, 0.4, 80],
        'label': 'Leaf (Top Spin)',
        'description': 'Top spin causing late drop'
    },
    'Elevator': {
        'bounds': [
            (28, 38),      # v0: very high speed
            (-0.2, 0.2),   # theta: minimal horizontal deviation
            (0.05, 0.3),   # phi: low launch angle
            (5, 25)        # omega: low spin (knuckleball)
        ],
        'initial_guess': [33, 0.0, 0.15, 15],
        'label': 'Elevator (Knuckle)',
        'description': 'Low spin, variable drag causing sudden drop'
    }
}

# Main Optimization Loop
def run_trajectory_fitting():
    """Run trajectory fitting for all three ball models"""

    print("=" * 80)
    print("FOOTBALL TRAJECTORY FITTING ANALYSIS")
    print("=" * 80)
    print(f"Trajectory points: {len(x_data)}")
    print(f"X range: {x_data[0]:.3f} to {x_data[-1]:.3f} m")
    print(f"Y range: {y_data[0]:.3f} to {y_data[-1]:.3f} m")
    print(f"Z range: {z_data[0]:.3f} to {z_data[-1]:.3f} m")
    print()

    print(f"{'Model':<12} | {'Loss (MSE)':<12} | {'v0 (m/s)':<10} | {'ω (rad/s)':<12} | {'Launch °':<10} | {'Status'}")
    print("-" * 90)

    results = {}
    trajectories = {}

    for name, cfg in models_config.items():
        print(f"Fitting {name} model...")

        # Use model-specific initial guess
        p0 = cfg['initial_guess']

        # Run optimization
        res = minimize(objective, p0, args=(name,),
                      bounds=cfg['bounds'],
                      method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-6})

        results[name] = res

        # Extract optimized parameters
        v0, theta, phi, omega = res.x
        launch_angle_deg = np.degrees(phi)

        status = "✓" if res.success else "✗"

        print(f"{name:<12}"
              f"{res.fun:<12.4f}"
              f"{v0:<10.2f}"
              f"{omega:<12.1f}"
              f"{launch_angle_deg:<10.1f}"
              f"{status:<8}")

        # Generate trajectory with optimized parameters
        vx = v0 * np.cos(phi) * np.cos(theta)
        vy = v0 * np.cos(phi) * np.sin(theta)
        vz = v0 * np.sin(phi)
        state0 = [x_data[0], y_data[0], z_data[0], vx, vy, vz]

        sol = solve_ivp(ball_dynamics, [0, t_data[-1]], state0,
                       args=(name, [omega]), t_eval=t_data,
                       rtol=1e-4, atol=1e-6)
        trajectories[name] = sol

    return results, trajectories

results, trajectories = run_trajectory_fitting()

def analyze_results(results, trajectories):
    """Analyze fitting results and determine best model"""

    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)

    # Find best model
    losses = {name: res.fun for name, res in results.items()}
    best_model = min(losses, key=losses.get)

    print(f"Best fitting model: {best_model} (MSE = {losses[best_model]:.4f})")
    print()

    # Detailed analysis for each model
    for name, res in results.items():
        v0, theta, phi, omega = res.x
        launch_angle = np.degrees(phi)

        print(f"{name} Model Analysis:")
        print(f"  Initial Speed: {v0:.2f} m/s")
        print(f"  Launch Angle: {launch_angle:.1f}°")
        print(f"  Horizontal Angle: {np.degrees(theta):.1f}°")
        print(f"  Spin Rate: {omega:.1f} rad/s ({omega*180/np.pi:.1f}°/s)")
        print(f"  MSE Loss: {res.fun:.4f}")
        print(f"  Converged: {'Yes' if res.success else 'No'}")
        print()

    # Trajectory characteristics
    print("Trajectory Characteristics:")
    sol_best = trajectories[best_model]
    x_final = sol_best.y[0, -1]
    y_final = sol_best.y[1, -1]
    z_final = sol_best.y[2, -1]
    z_max = np.max(sol_best.y[2])
    flight_time = t_data[-1]

    print(f"  Final X position: {x_final:.1f} m")
    print(f"  Final Y position: {y_final:.1f} m")
    print(f"  Final Z position: {z_final:.1f} m")
    print(f"  Maximum height: {z_max:.1f} m")
    print(f"  Flight time: {flight_time:.2f} s")

def create_visualization(results, trajectories):
    """Create comprehensive trajectory visualization"""

    # Color scheme
    colors = {'Banana': '#FF6B6B', 'Leaf': '#4ECDC4', 'Elevator': '#45B7D1'}
    markers = {'Banana': 'o', 'Leaf': 's', 'Elevator': '^'}

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 3D Trajectory Plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(x_data, y_data, z_data, 'ko-', linewidth=2, markersize=4, label='Observed')
    for name, sol in trajectories.items():
        ax1.plot(sol.y[0], sol.y[1], sol.y[2],
                color=colors[name], linewidth=2, marker=markers[name],
                markersize=3, markevery=10, label=f'{name} (fit)')
    ax1.set_xlabel('X (m)'), ax1.set_ylabel('Y (m)'), ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Side View (X-Z plane)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(x_data, z_data, 'ko-', linewidth=2, markersize=4, label='Observed')
    for name, sol in trajectories.items():
        loss = results[name].fun
        ax2.plot(sol.y[0], sol.y[2], color=colors[name], linewidth=2,
                label=f'{name} (Err={loss:.4f})')
    ax2.set_xlabel('X (m)'), ax2.set_ylabel('Z (m)')
    ax2.set_title('Side View (X-Z Plane)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Top View (X-Y plane)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(x_data, y_data, 'ko-', linewidth=2, markersize=4, label='Observed')
    for name, sol in trajectories.items():
        ax3.plot(sol.y[0], sol.y[1], color=colors[name], linewidth=2, label=name)
    ax3.set_xlabel('X (m)'), ax3.set_ylabel('Y (m)')
    ax3.set_title('Top View (X-Y Plane)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Error Analysis
    ax4 = fig.add_subplot(2, 3, 4)
    models = list(results.keys())
    losses = [results[name].fun for name in models]
    bars = ax4.bar(models, losses, color=[colors[m] for m in models], alpha=0.7)
    ax4.set_ylabel('MSE Loss')
    ax4.set_title('Fitting Error Comparison')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(losses)*0.02,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=10)

    # Parameter Comparison
    ax5 = fig.add_subplot(2, 3, 5)
    params = ['v0', 'angle', 'omega']
    param_labels = ['Speed (m/s)', 'Launch (°)', 'Spin (rad/s)']

    for i, (param, label) in enumerate(zip(['v0', 'phi', 'omega'], param_labels)):
        values = []
        for name in models:
            if param == 'phi':
                values.append(np.degrees(results[name].x[2]))  # Convert phi to degrees
            elif param == 'v0':
                values.append(results[name].x[0])
            elif param == 'omega':
                values.append(results[name].x[3])

        x_pos = np.arange(len(models)) + i*0.25
        ax5.bar(x_pos, values, width=0.2, label=label,
               color=[colors[m] for m in models], alpha=0.7)

    ax5.set_xticks(np.arange(len(models)) + 0.25)
    ax5.set_xticklabels(models)
    ax5.set_title('Optimized Parameters')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # Height Profile with Goal Analysis
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(x_data, z_data, 'ko-', linewidth=2, markersize=4, label='Observed')

    # Add goal posts
    goal_x = 52.5
    goal_height = 2.44  # FIFA standard
    ax6.axvline(x=goal_x, color='red', linestyle='--', alpha=0.7, label=f'Goal line: {goal_x:.1f}m')
    ax6.axhline(y=goal_height, color='red', linestyle=':', alpha=0.7, label=f'Crossbar: {goal_height:.2f}m')
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Ground')

    for name, sol in trajectories.items():
        ax6.plot(sol.y[0], sol.y[2], color=colors[name], linewidth=2, label=name)

        # Find intersection with goal line
        goal_idx = np.argmin(np.abs(sol.y[0] - goal_x))
        if goal_idx < len(sol.y[2]):
            goal_height_fit = sol.y[2, goal_idx]
            ax6.plot(goal_x, goal_height_fit, marker=markers[name],
                    color=colors[name], markersize=8, markeredgecolor='black')

    ax6.set_xlabel('X (m)'), ax6.set_ylabel('Z (m)')
    ax6.set_title('Goal Analysis')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(min(x_data)-1, max(x_data)+1)
    ax6.set_ylim(-0.5, max(z_data)+0.5)

    plt.tight_layout()
    return fig

# Run analysis and visualization
analyze_results(results, trajectories)
fig = create_visualization(results, trajectories)

# Save plots
output_dir = os.path.dirname(script_dir)
plot_path = os.path.join(output_dir, 'q2', 'trajectory_fitting_analysis.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {plot_path}")

# Additional statistics
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)

losses = [results[name].fun for name in results.keys()]
best_loss = min(losses)
worst_loss = max(losses)
improvement_ratio = worst_loss / best_loss

print(f"Best model loss: {best_loss:.4f}")
print(f"Worst model loss: {worst_loss:.4f}")
print(f"Improvement ratio: {improvement_ratio:.1f}x")
print(f"Average loss: {np.mean(losses):.2f}")

# Model ranking
ranking = sorted(results.items(), key=lambda x: x[1].fun)
print("\nModel Ranking (by fitting quality):")
for i, (name, res) in enumerate(ranking, 1):
    print(f"{i}. {name}: MSE = {res.fun:.4f}")

print("\nAnalysis complete!")