import numpy as np
import plotly.graph_objects as go

# Soccer field dimensions (meters)
FIELD_LENGTH = 105  # Field length
FIELD_WIDTH = 68    # Field width
GOAL_WIDTH = 7.32   # Goal width
GOAL_HEIGHT = 2.44  # Goal height
PENALTY_SPOT_DISTANCE = 11  # Distance from goal line to penalty spot
CENTER_CIRCLE_RADIUS = 9.15  # Center circle radius

def create_interactive_soccer_field():
    """
    Create interactive 3D visualization of soccer field with movable camera
    """
    fig = go.Figure()

    # 1. Draw field surface (ground) - using a semi-transparent green plane
    x_ground = np.linspace(-FIELD_LENGTH/2, FIELD_LENGTH/2, 50)
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
    # Center spot
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        marker=dict(size=3, color='white'),
        text=['(0.0, 0.0, 0.0)'],
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
    goal_x_left = -FIELD_LENGTH/2

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

    # Goal corner coordinates (4 points) - labeled but not in legend
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

    # Left goal posts
    fig.add_trace(go.Scatter3d(
        x=[goal_x_left, goal_x_left],
        y=[-GOAL_WIDTH/2, -GOAL_WIDTH/2],
        z=[0, GOAL_HEIGHT],
        mode='lines',
        line=dict(color='white', width=6),
        name='Left Goal Posts',
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[goal_x_left, goal_x_left],
        y=[GOAL_WIDTH/2, GOAL_WIDTH/2],
        z=[0, GOAL_HEIGHT],
        mode='lines',
        line=dict(color='white', width=6),
        showlegend=False
    ))

    # Left crossbar
    fig.add_trace(go.Scatter3d(
        x=[goal_x_left, goal_x_left],
        y=[-GOAL_WIDTH/2, GOAL_WIDTH/2],
        z=[GOAL_HEIGHT, GOAL_HEIGHT],
        mode='lines',
        line=dict(color='white', width=6),
        name='Left Crossbar',
        showlegend=False
    ))

    # 5. Draw penalty spots
    penalty_x_right = FIELD_LENGTH/2 - PENALTY_SPOT_DISTANCE
    penalty_x_left = -FIELD_LENGTH/2 + PENALTY_SPOT_DISTANCE

    fig.add_trace(go.Scatter3d(
        x=[penalty_x_right, penalty_x_left],
        y=[0, 0],
        z=[0, 0],
        mode='markers+text',
        marker=dict(size=2, color='white'),
        text=[f'({penalty_x_right:.1f}, 0.0, 0.0)', f'({penalty_x_left:.1f}, 0.0, 0.0)'],
        textposition="top center",
        textfont=dict(size=8, color='black'),
        name='Penalty Spots',
        showlegend=False
    ))

    # 6. Draw penalty arcs
    # Right penalty arc
    theta_arc_right = np.linspace(-np.pi/2, np.pi/2, 50)
    x_arc_right = penalty_x_right + CENTER_CIRCLE_RADIUS * np.cos(theta_arc_right)
    y_arc_right = CENTER_CIRCLE_RADIUS * np.sin(theta_arc_right)
    z_arc_right = np.zeros_like(theta_arc_right)

    fig.add_trace(go.Scatter3d(
        x=x_arc_right,
        y=y_arc_right,
        z=z_arc_right,
        mode='lines',
        line=dict(color='white', width=3, dash='dash'),
        name='Penalty Arcs',
        showlegend=False
    ))

    # Left penalty arc
    theta_arc_left = np.linspace(np.pi/2, 3*np.pi/2, 50)
    x_arc_left = penalty_x_left + CENTER_CIRCLE_RADIUS * np.cos(theta_arc_left)
    y_arc_left = CENTER_CIRCLE_RADIUS * np.sin(theta_arc_left)
    z_arc_left = np.zeros_like(theta_arc_left)

    fig.add_trace(go.Scatter3d(
        x=x_arc_left,
        y=y_arc_left,
        z=z_arc_left,
        mode='lines',
        line=dict(color='white', width=3, dash='dash'),
        name='Penalty Arcs',
        showlegend=False
    ))

    # 7. Draw penalty areas first (18-yard boxes) as colored regions
    PENALTY_AREA_DEPTH = 16.5
    PENALTY_AREA_HALF_WIDTH = 20.16

    # Right penalty area (x = 52.5) - light green color for harmony
    penalty_x_right = FIELD_LENGTH/2  # 52.5
    penalty_x_right_inner = penalty_x_right - PENALTY_AREA_DEPTH

    x_penalty_right = [penalty_x_right, penalty_x_right_inner, penalty_x_right_inner, penalty_x_right]
    y_penalty_right = [-PENALTY_AREA_HALF_WIDTH, -PENALTY_AREA_HALF_WIDTH, PENALTY_AREA_HALF_WIDTH, PENALTY_AREA_HALF_WIDTH]
    z_penalty_right = [0.01, 0.01, 0.01, 0.01]  # Slightly above ground

    fig.add_trace(go.Mesh3d(
        x=x_penalty_right,
        y=y_penalty_right,
        z=z_penalty_right,
        color='rgba(222, 184, 135, 1.0)',  # Light brown (burlywood), fully opaque
        name='Penalty Area',
        showscale=False,
        showlegend=True
    ))

    # Left penalty area (x = -52.5) - light green color for harmony
    penalty_x_left = -FIELD_LENGTH/2  # -52.5
    penalty_x_left_inner = penalty_x_left + PENALTY_AREA_DEPTH

    x_penalty_left = [penalty_x_left, penalty_x_left_inner, penalty_x_left_inner, penalty_x_left]
    y_penalty_left = [-PENALTY_AREA_HALF_WIDTH, -PENALTY_AREA_HALF_WIDTH, PENALTY_AREA_HALF_WIDTH, PENALTY_AREA_HALF_WIDTH]
    z_penalty_left = [0.01, 0.01, 0.01, 0.01]

    fig.add_trace(go.Mesh3d(
        x=x_penalty_left,
        y=y_penalty_left,
        z=z_penalty_left,
        color='rgba(222, 184, 135, 1.0)',  # Light brown (burlywood), fully opaque
        showlegend=False
    ))

    # Penalty area corner coordinates (key points)
    penalty_corners_x = [FIELD_LENGTH/2, FIELD_LENGTH/2, penalty_x_right_inner, penalty_x_right_inner,
                        penalty_x_left, penalty_x_left, penalty_x_left_inner, penalty_x_left_inner]
    penalty_corners_y = [-PENALTY_AREA_HALF_WIDTH, PENALTY_AREA_HALF_WIDTH, PENALTY_AREA_HALF_WIDTH, -PENALTY_AREA_HALF_WIDTH,
                        -PENALTY_AREA_HALF_WIDTH, PENALTY_AREA_HALF_WIDTH, PENALTY_AREA_HALF_WIDTH, -PENALTY_AREA_HALF_WIDTH]
    penalty_corners_z = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    penalty_corner_labels = [
        f'({FIELD_LENGTH/2:.1f}, {-PENALTY_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({FIELD_LENGTH/2:.1f}, {PENALTY_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({penalty_x_right_inner:.1f}, {PENALTY_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({penalty_x_right_inner:.1f}, {-PENALTY_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({penalty_x_left:.1f}, {-PENALTY_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({penalty_x_left:.1f}, {PENALTY_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({penalty_x_left_inner:.1f}, {PENALTY_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({penalty_x_left_inner:.1f}, {-PENALTY_AREA_HALF_WIDTH:.1f}, 0.0)'
    ]

    fig.add_trace(go.Scatter3d(
        x=penalty_corners_x,
        y=penalty_corners_y,
        z=penalty_corners_z,
        mode='markers+text',
        marker=dict(size=3, color='blue'),
        text=penalty_corner_labels,
        textposition="top center",
        textfont=dict(size=6, color='black'),
        showlegend=False
    ))

    # 8. Draw goal areas (6-yard boxes) as colored regions - drawn last to overlay penalty areas
    GOAL_AREA_DEPTH = 5.5
    GOAL_AREA_HALF_WIDTH = 9.16

    # Right goal area (x = 52.5) - bright yellow color
    goal_x_right = FIELD_LENGTH/2  # 52.5
    goal_x_right_inner = goal_x_right - GOAL_AREA_DEPTH

    # Create goal area surface
    x_goal_right = [goal_x_right, goal_x_right_inner, goal_x_right_inner, goal_x_right]
    y_goal_right = [-GOAL_AREA_HALF_WIDTH, -GOAL_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH]
    z_goal_right = [0.02, 0.02, 0.02, 0.02]  # Higher than penalty areas to ensure overlay

    fig.add_trace(go.Mesh3d(
        x=x_goal_right,
        y=y_goal_right,
        z=z_goal_right,
        color='rgba(255, 215, 0, 1.0)',  # Gold/yellow, fully opaque
        name='Goal Area',
        showscale=False,
        showlegend=True
    ))

    # Left goal area (x = -52.5) - bright yellow color
    goal_x_left = -FIELD_LENGTH/2  # -52.5
    goal_x_left_inner = goal_x_left + GOAL_AREA_DEPTH

    x_goal_left = [goal_x_left, goal_x_left_inner, goal_x_left_inner, goal_x_left]
    y_goal_left = [-GOAL_AREA_HALF_WIDTH, -GOAL_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH]
    z_goal_left = [0.02, 0.02, 0.02, 0.02]  # Higher than penalty areas to ensure overlay

    fig.add_trace(go.Mesh3d(
        x=x_goal_left,
        y=y_goal_left,
        z=z_goal_left,
        color='rgba(255, 215, 0, 1.0)',  # Gold/yellow, fully opaque
        showlegend=False
    ))

    # Goal area corner coordinates (key points)
    goal_area_corners_x = [FIELD_LENGTH/2, FIELD_LENGTH/2, goal_x_right_inner, goal_x_right_inner,
                          goal_x_left, goal_x_left, goal_x_left_inner, goal_x_left_inner]
    goal_area_corners_y = [-GOAL_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH, -GOAL_AREA_HALF_WIDTH,
                          -GOAL_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH, -GOAL_AREA_HALF_WIDTH]
    goal_area_corners_z = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

    goal_area_corner_labels = [
        f'({FIELD_LENGTH/2:.1f}, {-GOAL_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({FIELD_LENGTH/2:.1f}, {GOAL_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({goal_x_right_inner:.1f}, {GOAL_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({goal_x_right_inner:.1f}, {-GOAL_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({goal_x_left:.1f}, {-GOAL_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({goal_x_left:.1f}, {GOAL_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({goal_x_left_inner:.1f}, {GOAL_AREA_HALF_WIDTH:.1f}, 0.0)',
        f'({goal_x_left_inner:.1f}, {-GOAL_AREA_HALF_WIDTH:.1f}, 0.0)'
    ]

    fig.add_trace(go.Scatter3d(
        x=goal_area_corners_x,
        y=goal_area_corners_y,
        z=goal_area_corners_z,
        mode='markers+text',
        marker=dict(size=3, color='orange'),
        text=goal_area_corner_labels,
        textposition="top center",
        textfont=dict(size=6, color='black'),
        showlegend=False
    ))


    # Calculate proper aspect ratios for realistic proportions
    # Field dimensions: 105m x 68m, with visualization height of 5m for better viewing
    vis_height = 5  # meters - reasonable height for 3D visualization
    x_range_total = FIELD_LENGTH/2 + 20  # Add padding (half field)
    y_range_total = FIELD_WIDTH + 20   # Add padding
    z_range_total = vis_height

    # Update layout for realistic proportions
    fig.update_layout(
        title='Soccer Field 3D Model',
        scene=dict(
            xaxis_title='X Axis (meters) - Field Length',
            yaxis_title='Y Axis (meters) - Field Width',
            zaxis_title='Z Axis (meters) - Height',
            xaxis=dict(range=[0, FIELD_LENGTH/2]),
            yaxis=dict(range=[-FIELD_WIDTH/2, FIELD_WIDTH/2]),
            zaxis=dict(range=[0, vis_height]),
            # Set aspect ratio to match real field proportions
            aspectratio=dict(x=x_range_total, y=y_range_total, z=z_range_total),
            aspectmode='manual',  # Use manual aspect ratio for realistic scaling
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),  # Initial camera position
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'   # Transparent plot area
    )

    return fig


if __name__ == "__main__":
    # Create Plotly interactive version
    fig_plotly = create_interactive_soccer_field()
    fig_plotly.write_html('/home/ubuntu/projects/mcm/training2/visual/soccer_field_interactive.html')
    print("Interactive 3D model saved as: soccer_field_interactive.html")
