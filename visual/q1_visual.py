import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches
import os
import plotly.graph_objects as go

# Soccer field dimensions (meters)
FIELD_LENGTH = 105  # Field length
FIELD_WIDTH = 68    # Field width
GOAL_WIDTH = 7.32   # Goal width
GOAL_HEIGHT = 2.44  # Goal height
PENALTY_SPOT_DISTANCE = 11  # Distance from goal line to penalty spot
CENTER_CIRCLE_RADIUS = 9.15  # Center circle radius
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_HALF_WIDTH = 20.16
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_HALF_WIDTH = 9.16

def create_2d_soccer_field_background(ax):
    """
    Add soccer field background to the given axes
    """
    # 1. Draw field boundary
    field_boundary = patches.Rectangle((0, -FIELD_WIDTH/2),
                                     FIELD_LENGTH/2, FIELD_WIDTH,
                                     linewidth=2, edgecolor='white', facecolor='darkgreen', alpha=0.7)
    ax.add_patch(field_boundary)

    # 2. Draw center line
    ax.axvline(x=0, ymin=0.1, ymax=0.9, color='white', linewidth=3)

    # 3. Draw center circle
    center_circle = patches.Circle((0, 0), CENTER_CIRCLE_RADIUS,
                                 fill=False, color='white', linewidth=3)
    ax.add_patch(center_circle)

    # 4. Draw center spot
    ax.scatter([0], [0], color='white', s=30)

    # 5. Draw penalty areas - light brown rectangles
    penalty_area_right = patches.Rectangle((FIELD_LENGTH/2 - PENALTY_AREA_DEPTH, -PENALTY_AREA_HALF_WIDTH),
                                         PENALTY_AREA_DEPTH, PENALTY_AREA_HALF_WIDTH * 2,
                                         linewidth=2, edgecolor='white',
                                         facecolor=(222/255, 184/255, 135/255), alpha=1.0, label='Penalty Area')
    ax.add_patch(penalty_area_right)

    # 6. Draw goal areas - gold/yellow rectangles (overlay on penalty areas)
    goal_area_right = patches.Rectangle((FIELD_LENGTH/2 - GOAL_AREA_DEPTH, -GOAL_AREA_HALF_WIDTH),
                                      GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH * 2,
                                      linewidth=2, edgecolor='white',
                                      facecolor=(255/255, 215/255, 0/255), alpha=1.0, label='Goal Area')
    ax.add_patch(goal_area_right)

    # 7. Draw goal
    goal_x = FIELD_LENGTH/2
    ax.plot([goal_x, goal_x], [-GOAL_WIDTH/2, GOAL_WIDTH/2], color='white', linewidth=4)
    ax.plot([goal_x, goal_x + 1], [-GOAL_WIDTH/2, -GOAL_WIDTH/2], color='white', linewidth=4)
    ax.plot([goal_x, goal_x + 1], [GOAL_WIDTH/2, GOAL_WIDTH/2], color='white', linewidth=4)
    ax.plot([goal_x + 1, goal_x + 1], [-GOAL_WIDTH/2, GOAL_WIDTH/2], color='white', linewidth=4)

    # 8. Draw penalty spot
    penalty_x = FIELD_LENGTH/2 - PENALTY_SPOT_DISTANCE
    ax.scatter([penalty_x], [0], color='white', s=25)

    # Set limits
    ax.set_xlim(-5, FIELD_LENGTH/2 + 10)
    ax.set_ylim(-FIELD_WIDTH/2 - 10, FIELD_WIDTH/2 + 10)
    ax.set_aspect('equal')

# Load the generated trajectory
# 使用绝对路径以确保从任何工作目录运行都能找到轨迹文件
script_dir = os.path.dirname(os.path.abspath(__file__))
traj_path = os.path.join(script_dir, '..', 'solve', 'optimized_trajectory.csv')
traj = pd.read_csv(traj_path)

# Create individual plots

# 1. Top view (x-y) with soccer field background
fig1, ax1 = plt.subplots(figsize=(12, 8))
create_2d_soccer_field_background(ax1)
ax1.plot(traj['x'], traj['y'], color='blue', linewidth=2, label='Ball Trajectory')
ax1.scatter(traj['x'][::4], traj['y'][::4], color='red', s=30, alpha=0.8, label='Ball positions (0.04s intervals)')
ax1.scatter([52.5], [0], color='orange', marker='x', s=100, linewidth=3, label='Goal Center')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Top View - Ball Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)
plot_path1 = os.path.join(script_dir, 'trajectory_top_view.png')
fig1.savefig(plot_path1, dpi=300, bbox_inches='tight')
plt.close(fig1)

# 2. Side view (x-z)
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(traj['x'], traj['z'], color='blue', linewidth=2, label='Ball Trajectory')
ax2.scatter(traj['x'][::4], traj['z'][::4], color='red', s=30, alpha=0.8, label='Ball positions (0.04s intervals)')
ax2.axvline(x=52.5, color='orange', linestyle='--', linewidth=2, label='Goal Line')
ax2.axhline(y=2.44, color='purple', linestyle=':', linewidth=2, label='Crossbar')
ax2.axhline(y=0, color='green', linestyle='-', linewidth=1, alpha=0.5, label='Ground')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Z (m)')
ax2.set_title('Side View - Ball Trajectory')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(30, 55)
ax2.set_ylim(0, 4)
plot_path2 = os.path.join(script_dir, 'trajectory_side_view.png')
fig2.savefig(plot_path2, dpi=300, bbox_inches='tight')
plt.close(fig2)

# 3. Interactive 3D Plot with soccer field (using Plotly)
def create_interactive_trajectory_plotly(traj_data):
    """
    Create interactive 3D visualization of ball trajectory with soccer field
    """
    fig = go.Figure()

    # 1. Draw field surface (ground) - using a semi-transparent green plane
    x_ground = np.linspace(0, FIELD_LENGTH/2, 50)
    y_ground = np.linspace(-FIELD_WIDTH/2, FIELD_WIDTH/2, 50)
    X_ground, Y_ground = np.meshgrid(x_ground, y_ground)
    Z_ground = np.zeros_like(X_ground)

    fig.add_trace(go.Surface(
        x=X_ground, y=Y_ground, z=Z_ground,
        colorscale=[[0, 'rgb(34,139,34)'], [1, 'rgb(34,139,34)']],  # Dark green
        opacity=0.7,
        name='Field Surface',
        showscale=False
    ))

    # 2. Draw center line (x=0) - white solid line
    y_center = np.linspace(-FIELD_WIDTH/2, FIELD_WIDTH/2, 100)
    z_center = np.zeros_like(y_center)

    fig.add_trace(go.Scatter3d(
        x=np.zeros_like(y_center),
        y=y_center,
        z=z_center,
        mode='lines',
        line=dict(color='white', width=4),
        name='Center Line',
        showlegend=False
    ))

    # 3. Draw center spot and circle
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        marker=dict(size=3, color='white'),
        text=['Center'],
        textposition="top center",
        textfont=dict(size=8, color='black'),
        name='Center Spot',
        showlegend=False
    ))

    # Center circle
    theta = np.linspace(0, 2*np.pi, 100)
    x_center_circle = CENTER_CIRCLE_RADIUS * np.cos(theta)
    y_center_circle = CENTER_CIRCLE_RADIUS * np.sin(theta)
    z_center_circle = np.zeros_like(theta)

    fig.add_trace(go.Scatter3d(
        x=x_center_circle,
        y=y_center_circle,
        z=z_center_circle,
        mode='lines',
        line=dict(color='white', width=4),
        name='Center Circle',
        showlegend=False
    ))

    # 4. Draw goal posts and crossbars
    goal_x_right = FIELD_LENGTH/2

    # Right goal posts
    fig.add_trace(go.Scatter3d(
        x=[goal_x_right, goal_x_right],
        y=[-GOAL_WIDTH/2, -GOAL_WIDTH/2],
        z=[0, GOAL_HEIGHT],
        mode='lines',
        line=dict(color='white', width=6),
        name='Right Goal Posts',
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[goal_x_right, goal_x_right],
        y=[GOAL_WIDTH/2, GOAL_WIDTH/2],
        z=[0, GOAL_HEIGHT],
        mode='lines',
        line=dict(color='white', width=6),
        showlegend=False
    ))

    # Right crossbar
    fig.add_trace(go.Scatter3d(
        x=[goal_x_right, goal_x_right],
        y=[-GOAL_WIDTH/2, GOAL_WIDTH/2],
        z=[GOAL_HEIGHT, GOAL_HEIGHT],
        mode='lines',
        line=dict(color='white', width=6),
        name='Right Crossbar',
        showlegend=False
    ))

    # Goal corner coordinates
    goal_corners_x = [goal_x_right, goal_x_right, goal_x_right, goal_x_right]
    goal_corners_y = [-GOAL_WIDTH/2, GOAL_WIDTH/2, -GOAL_WIDTH/2, GOAL_WIDTH/2]
    goal_corners_z = [0, 0, GOAL_HEIGHT, GOAL_HEIGHT]

    fig.add_trace(go.Scatter3d(
        x=goal_corners_x,
        y=goal_corners_y,
        z=goal_corners_z,
        mode='markers+text',
        marker=dict(size=4, color='red'),
        text=[f'({goal_x_right:.1f}, {-GOAL_WIDTH/2:.1f}, 0)',
              f'({goal_x_right:.1f}, {GOAL_WIDTH/2:.1f}, 0)',
              f'({goal_x_right:.1f}, {-GOAL_WIDTH/2:.1f}, {GOAL_HEIGHT:.1f})',
              f'({goal_x_right:.1f}, {GOAL_WIDTH/2:.1f}, {GOAL_HEIGHT:.1f})'],
        textposition="top center",
        textfont=dict(size=8, color='black'),
        showlegend=False
    ))

    # 5. Draw penalty areas
    penalty_x_right = FIELD_LENGTH/2
    penalty_x_right_inner = penalty_x_right - PENALTY_AREA_DEPTH

    x_penalty_right = [penalty_x_right, penalty_x_right_inner, penalty_x_right_inner, penalty_x_right]
    y_penalty_right = [-PENALTY_AREA_HALF_WIDTH, -PENALTY_AREA_HALF_WIDTH, PENALTY_AREA_HALF_WIDTH, PENALTY_AREA_HALF_WIDTH]
    z_penalty_right = [0.01, 0.01, 0.01, 0.01]

    fig.add_trace(go.Mesh3d(
        x=x_penalty_right,
        y=y_penalty_right,
        z=z_penalty_right,
        color='rgba(222, 184, 135, 1.0)',  # Light brown
        name='Penalty Area',
        showscale=False,
        showlegend=True
    ))

    # 6. Draw goal areas (overlay)
    goal_x_right_inner = FIELD_LENGTH/2 - GOAL_AREA_DEPTH

    x_goal_right = [FIELD_LENGTH/2, goal_x_right_inner, goal_x_right_inner, FIELD_LENGTH/2]
    y_goal_right = [-GOAL_AREA_HALF_WIDTH, -GOAL_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH]
    z_goal_right = [0.02, 0.02, 0.02, 0.02]

    fig.add_trace(go.Mesh3d(
        x=x_goal_right,
        y=y_goal_right,
        z=z_goal_right,
        color='rgba(255, 215, 0, 1.0)',  # Gold/yellow
        name='Goal Area',
        showscale=False,
        showlegend=True
    ))

    # 7. Plot ball trajectory
    fig.add_trace(go.Scatter3d(
        x=traj_data['x'],
        y=traj_data['y'],
        z=traj_data['z'],
        mode='lines',
        line=dict(color='red', width=6),
        name='Ball Trajectory'
    ))

    # 8. Plot ball position markers (every 0.04s)
    fig.add_trace(go.Scatter3d(
        x=traj_data['x'][::4],
        y=traj_data['y'][::4],
        z=traj_data['z'][::4],
        mode='markers',
        marker=dict(size=4, color='black', symbol='circle'),
        name='Ball Positions (0.04s intervals)'
    ))

    # Set layout
    vis_height = 8  # meters
    x_range_total = FIELD_LENGTH/2 + 20
    y_range_total = FIELD_WIDTH + 20
    z_range_total = vis_height

    fig.update_layout(
        title='Interactive 3D Ball Trajectory with Soccer Field',
        scene=dict(
            xaxis_title='X Axis (meters) - Field Length',
            yaxis_title='Y Axis (meters) - Field Width',
            zaxis_title='Z Axis (meters) - Height',
            xaxis=dict(range=[0, FIELD_LENGTH/2]),
            yaxis=dict(range=[-FIELD_WIDTH/2, FIELD_WIDTH/2]),
            zaxis=dict(range=[0, vis_height]),
            aspectratio=dict(x=x_range_total, y=y_range_total, z=z_range_total),
            aspectmode='manual',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

# Create interactive 3D plot
fig3_interactive = create_interactive_trajectory_plotly(traj)
html_path3 = os.path.join(script_dir, 'trajectory_3d_interactive.html')
fig3_interactive.write_html(html_path3)


# 4. Time vs Height plot
fig4, ax4 = plt.subplots(figsize=(12, 6))
time_points = np.linspace(0, 1.2, len(traj))  # 1.2 seconds total
ax4.plot(time_points, traj['z'], color='blue', linewidth=2, label='Ball Height')
ax4.scatter(time_points[::4], traj['z'][::4], color='red', s=30, alpha=0.8, label='Position markers (0.04s intervals)')
ax4.axhline(y=2.44, color='purple', linestyle='--', linewidth=2, label='Crossbar Height')
ax4.axhline(y=0, color='green', linestyle='-', linewidth=1, alpha=0.5, label='Ground')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Height (m)')
ax4.set_title('Ball Height vs Time')
ax4.legend()
ax4.grid(True, alpha=0.3)
plot_path4 = os.path.join(script_dir, 'trajectory_height_time.png')
fig4.savefig(plot_path4, dpi=300, bbox_inches='tight')
plt.close(fig4)

print("Individual plots saved:")
print(f"1. Top View: {plot_path1}")
print(f"2. Side View: {plot_path2}")
print(f"3. Height vs Time: {plot_path4}")
print(f"4. Interactive 3D View (HTML): {html_path3}")

# Check height at goal
# Find point closest to x=52.5
idx = (traj['x'] - 52.5).abs().idxmin()
print(f"Ball position at Goal Line (x~52.5):")
print(traj.iloc[idx])