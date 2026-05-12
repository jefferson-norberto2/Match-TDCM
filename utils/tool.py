import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from shapely.geometry import Polygon

######################################
#             for test
######################################

def rotated_rect(x_center, y_center, width, height, angle_degrees):
    """
    Generates the four vertex coordinates of a rotated rectangle.
    :param x_center: X-coordinate of the center point.
    :param y_center: Y-coordinate of the center point.
    :param width: Rectangle width.
    :param height: Rectangle height.
    :param angle_degrees: Rotation angle (in degrees).
    :return: List of the four vertex coordinates after rotation.
    """
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    # Define the relative coordinates of the four vertices before rotation
    corners = np.array([
        [width / 2, height / 2],   # Top-right
        [-width / 2, height / 2],  # Top-left
        [-width / 2, -height / 2], # Bottom-left
        [width / 2, -height / 2]   # Bottom-right
    ])

    # Rotate and translate the vertices
    rotated_corners = []
    for x, y in corners:
        x_rot = x * cos_theta - y * sin_theta
        y_rot = x * sin_theta + y * cos_theta
        rotated_x = x_center + x_rot
        rotated_y = y_center + y_rot
        rotated_corners.append((rotated_x, rotated_y))
    
    return rotated_corners


def getIOU(true_param, pred_param, th=36, tw=36):
    """
    Calculates the Intersection over Union (IoU) of rotated rectangles.
    :param true_param: Ground truth box parameters [x, y, scale, rotate]
    :param pred_param: Predicted box parameters [x, y, scale, rotate]
    :param th: Template height (used to calculate actual size)
    :param tw: Template width (used to calculate actual size)
    :return: IoU value
    """
    if len(pred_param) == 4:
        # Only one scale parameter
        pred_param = [pred_param[0], pred_param[1], pred_param[2], pred_param[2], pred_param[3]]
        true_param = [true_param[0], true_param[1], true_param[2], true_param[2], true_param[3]] 
    elif len(pred_param) == 3:
         # No scale parameter
        pred_param = [pred_param[0], pred_param[1],  1,  1, pred_param[2]]
        true_param = [true_param[0], true_param[1],  1,  1, true_param[2]] 
    
    true_param = np.array(true_param)
    pred_param = np.array(pred_param)
    
    # Parse parameters
    true_x, true_y, true_sX, true_sY, true_r = true_param
    pred_x, pred_y, pred_sX, pred_sY, pred_r = pred_param

    # Calculate actual width and height (assuming scale is a global scaling factor)
    true_width = true_sX * tw
    true_height = true_sY * th
    pred_width = pred_sX * tw
    pred_height = pred_sY * th

    # Generate rotated rectangle vertices
    true_poly = rotated_rect(true_x, true_y, true_width, true_height, -true_r)
    pred_poly = rotated_rect(pred_x, pred_y, pred_width, pred_height, -pred_r)

    # Create polygon objects
    poly1 = Polygon(true_poly)
    poly2 = Polygon(pred_poly)

    # Check polygon validity
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    # Calculate intersection and union areas
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection

    return intersection / union if union != 0 else 0.0


def get_center(center, points, threshold=15):
    """
    Mean calculation after removing outliers based on a fixed threshold.
    
    Args:
    center : Center point coordinates, array of shape (2,)
    points : Original point set, array of shape (n, 2)
    threshold : Outlier judgment threshold (Euclidean distance), defaults to 15
    
    Returns:
    mean : Mean after removing outliers, array of shape (2,)
    """
    if len(points) == 0:
        return center
    
    # Calculate the Euclidean distance from all points to the center point
    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    distances = np.sqrt(dx**2 + dy**2)
    
    # Filter outliers
    mask = distances <= threshold
    filtered_points = points[mask]
    
    # Handle the extreme case where all points are outliers
    if len(filtered_points) == 0:
        return center  # Return original center
    
    # Calculate the mean of valid points
    return np.mean(filtered_points, axis=0)


def draw_rotated_bbox(ax, param, color, th=36, tw=36, lineWidth=1):
    """
    Draws a rotated rectangle and center line on the specified Axes.
    """
    if len(param) == 4:
        # Only one scale parameter
        param = [param[0], param[1], param[2], param[2], param[3]]
    elif len(param) == 3:
        # No scale parameter
        param = [param[0], param[1], 1, 1, param[2]]
    
    param = [float(p) for p in param]
    
    x, y, scale_x, scale_y, rotate = param
    h, w = th * scale_y, tw * scale_x

    # Create a rectangle box
    rect = patches.Rectangle(
        (x - w/2, y - h/2),  # Bottom-left coordinates
        w, h,                # Width and height
        linewidth=lineWidth,
        edgecolor=color,
        facecolor='none'
    )

    # Center line
    line = plt.Line2D(
        [x, x + w/2],
        [y, y],
        color=color,
        linewidth=lineWidth,
        linestyle='-'
    )

    # Rotation transformation
    transform = Affine2D().rotate_deg_around(x, y, -rotate) + ax.transData
    rect.set_transform(transform)
    line.set_transform(transform)

    ax.add_patch(rect)
    ax.add_line(line)