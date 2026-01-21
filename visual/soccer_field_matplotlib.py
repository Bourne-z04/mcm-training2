import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# Soccer field dimensions (meters)
FIELD_LENGTH = 105  # Field length
FIELD_WIDTH = 68    # Field width
GOAL_WIDTH = 7.32   # Goal width
GOAL_HEIGHT = 2.44  # Goal height
PENALTY_SPOT_DISTANCE = 11  # Distance from goal line to penalty spot
CENTER_CIRCLE_RADIUS = 9.15  # Center circle radius

def create_3d_soccer_field_matplotlib():
    """
    Create 3D visualization of soccer field using matplotlib
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Draw field surface (ground) - green plane
    x_ground = np.linspace(0, FIELD_LENGTH/2, 50)
    y_ground = np.linspace(-FIELD_WIDTH/2, FIELD_WIDTH/2, 50)
    X_ground, Y_ground = np.meshgrid(x_ground, y_ground)
    Z_ground = np.zeros_like(X_ground)

    ax.plot_surface(X_ground, Y_ground, Z_ground, color='darkgreen', alpha=0.7)

    # 2. Draw center line (x=0) - white line
    y_center = np.linspace(-FIELD_WIDTH/2, FIELD_WIDTH/2, 100)
    z_center = np.zeros_like(y_center)
    ax.plot(np.zeros_like(y_center), y_center, z_center,
            color='white', linewidth=3)

    # 3. Draw center spot and circle
    # Center spot
    ax.scatter([0], [0], [0.01], color='white', s=20)

    # Center circle
    theta = np.linspace(0, 2*np.pi, 100)
    x_center_circle = CENTER_CIRCLE_RADIUS * np.cos(theta)
    y_center_circle = CENTER_CIRCLE_RADIUS * np.sin(theta)
    z_center_circle = np.zeros_like(theta)
    ax.plot(x_center_circle, y_center_circle, z_center_circle,
            color='white', linewidth=3)

    # 4. Draw goal posts and crossbars
    goal_x_right = FIELD_LENGTH/2

    # Right goal posts (shown as lines in 3D)
    ax.plot([goal_x_right, goal_x_right], [-GOAL_WIDTH/2, -GOAL_WIDTH/2], [0, GOAL_HEIGHT],
            color='white', linewidth=4)
    ax.plot([goal_x_right, goal_x_right], [GOAL_WIDTH/2, GOAL_WIDTH/2], [0, GOAL_HEIGHT],
            color='white', linewidth=4)

    # Right crossbar
    ax.plot([goal_x_right, goal_x_right], [-GOAL_WIDTH/2, GOAL_WIDTH/2], [GOAL_HEIGHT, GOAL_HEIGHT],
            color='white', linewidth=4)

    # 5. Draw penalty spots
    penalty_x_right = FIELD_LENGTH/2 - PENALTY_SPOT_DISTANCE
    ax.scatter([penalty_x_right], [0], [0.01], color='white', s=15)

    # 6. Draw colored regions - penalty areas
    PENALTY_AREA_DEPTH = 16.5
    PENALTY_AREA_HALF_WIDTH = 20.16

    # Right penalty area - light brown
    penalty_x_right_inner = FIELD_LENGTH/2 - PENALTY_AREA_DEPTH
    ax.plot_surface(np.array([[FIELD_LENGTH/2, penalty_x_right_inner],
                            [FIELD_LENGTH/2, penalty_x_right_inner]]),
                   np.array([[-PENALTY_AREA_HALF_WIDTH, -PENALTY_AREA_HALF_WIDTH],
                            [PENALTY_AREA_HALF_WIDTH, PENALTY_AREA_HALF_WIDTH]]),
                   np.array([[0.01, 0.01], [0.01, 0.01]]),
                   color=(222/255, 184/255, 135/255), alpha=1.0)

    # 7. Draw colored regions - goal areas (overlay on penalty areas)
    GOAL_AREA_DEPTH = 5.5
    GOAL_AREA_HALF_WIDTH = 9.16

    # Right goal area - gold/yellow
    goal_x_right_inner = FIELD_LENGTH/2 - GOAL_AREA_DEPTH
    ax.plot_surface(np.array([[FIELD_LENGTH/2, goal_x_right_inner],
                            [FIELD_LENGTH/2, goal_x_right_inner]]),
                   np.array([[-GOAL_AREA_HALF_WIDTH, -GOAL_AREA_HALF_WIDTH],
                            [GOAL_AREA_HALF_WIDTH, GOAL_AREA_HALF_WIDTH]]),
                   np.array([[0.02, 0.02], [0.02, 0.02]]),
                   color=(255/255, 215/255, 0/255), alpha=1.0)

    # Set labels and limits
    ax.set_xlabel('X Axis (meters) - Field Length')
    ax.set_ylabel('Y Axis (meters) - Field Width')
    ax.set_zlabel('Z Axis (meters) - Height')
    ax.set_xlim(0, FIELD_LENGTH/2 + 5)
    ax.set_ylim(-FIELD_WIDTH/2 - 5, FIELD_WIDTH/2 + 5)
    ax.set_zlim(0, 8)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    ax.set_title('Soccer Field 3D Model')
    ax.grid(True)

    return fig

def create_2d_soccer_field_matplotlib():
    """
    Create 2D top-down view of soccer field using matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 8))

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
    penalty_area_right = patches.Rectangle((FIELD_LENGTH/2 - 16.5, -20.16),
                                         16.5, 40.32,
                                         linewidth=2, edgecolor='white',
                                         facecolor=(222/255, 184/255, 135/255), alpha=1.0, label='Penalty Area')
    ax.add_patch(penalty_area_right)

    # 6. Draw goal areas - gold/yellow rectangles (overlay on penalty areas)
    goal_area_right = patches.Rectangle((FIELD_LENGTH/2 - 5.5, -9.16),
                                      5.5, 18.32,
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

    # Set labels and limits
    ax.set_xlabel('X Axis (meters) - Field Length')
    ax.set_ylabel('Y Axis (meters) - Field Width')
    ax.set_xlim(-5, FIELD_LENGTH/2 + 10)
    ax.set_ylim(-FIELD_WIDTH/2 - 10, FIELD_WIDTH/2 + 10)
    ax.set_title('Soccer Field 2D Plan View')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return fig

if __name__ == "__main__":
    # Create 3D visualization
    fig_3d = create_3d_soccer_field_matplotlib()
    fig_3d.savefig('/home/ubuntu/projects/mcm/training2/visual/soccer_field_3d_matplotlib.png',
                   dpi=300, bbox_inches='tight')
    print("3D matplotlib visualization saved as: soccer_field_3d_matplotlib.png")

    # Create 2D plan view
    fig_2d = create_2d_soccer_field_matplotlib()
    fig_2d.savefig('/home/ubuntu/projects/mcm/training2/visual/soccer_field_2d_matplotlib.png',
                   dpi=300, bbox_inches='tight')
    print("2D matplotlib plan view saved as: soccer_field_2d_matplotlib.png")

    # Show plots
    plt.show()
