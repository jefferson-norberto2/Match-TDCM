import cv2
import numpy as np
import os
import glob
import math
from PIL import Image

'''
Build DOTA template matching test set
'''

def draw_polygon(image, points, color, thickness=2):
    """Draw a polygon on the image"""
    int_points = np.array([[int(p[0]), int(p[1])] for p in points], dtype=np.int32)
    cv2.polylines(image, [int_points], isClosed=True, color=color, thickness=thickness)
    return image

def calculate_min_area_rect_angle(corners):
    """Calculate the rotation angle using the minimum bounding rectangle"""
    corners_array = np.array(corners, dtype=np.float32)
    rect = cv2.minAreaRect(corners_array)
    return rect[2]  # Return the rotation angle

def calculate_rotation_angle(corners):
    """Calculate rotation angle based on template coordinates - Improved version"""
    if len(corners) < 4:
        return 0
    
    # Method 1: Calculate angle using minimum bounding rectangle
    min_area_angle = calculate_min_area_rect_angle(corners)
    
    # Method 2: Calculate the angle of the two long edges and take the average
    edges = []
    for i in range(4):
        dx = corners[(i+1)%4][0] - corners[i][0]
        dy = corners[(i+1)%4][1] - corners[i][1]
        length = math.sqrt(dx**2 + dy**2)
        angle = math.degrees(math.atan2(dy, dx))
        edges.append((length, angle))
    
    # Sort by length, take the two longest edges
    edges.sort(key=lambda x: x[0], reverse=True)
    long_edge_angles = [edges[0][1], edges[1][1]]
    
    # Calculate average angle
    avg_angle = np.mean(long_edge_angles)
    
    # Select the most stable angle calculation method
    # Use the angle of the minimum bounding rectangle, but adjust to the -90 to 90 degree range
    if min_area_angle < -45:
        min_area_angle += 90
    elif min_area_angle > 45:
        min_area_angle -= 90
    
    return min_area_angle

def rotate_point(point, center, angle_deg):
    """Rotate point coordinates (around the center point, consistent with OpenCV rotation direction)"""
    angle_rad = np.deg2rad(angle_deg)
    x, y = point
    cx, cy = center
    
    # Translate point to origin
    x -= cx
    y -= cy
    
    # Rotate
    x_rot = x * np.cos(angle_rad) + y * np.sin(angle_rad)
    y_rot = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
    
    # Translate back to original position
    x_rot += cx
    y_rot += cy
    
    return (x_rot, y_rot)

def process_template_data(image, template_corners):
    """Process template data: rotate image and crop template - Improved version"""
    # Calculate rotation angle
    rotation_angle = calculate_rotation_angle(template_corners)
    
    # Calculate rotation center (using the center of the minimum bounding rectangle)
    corners_array = np.array(template_corners, dtype=np.float32)
    rect = cv2.minAreaRect(corners_array)
    center = rect[0]  # Use the center of the minimum bounding rectangle
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # Rotate the entire image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    # Calculate the four corner coordinates of the rotated template
    rotated_corners = [rotate_point(corner, center, rotation_angle) for corner in template_corners]
    
    # Find the bounding box of the rotated template (using the minimum bounding rectangle)
    rotated_corners_array = np.array(rotated_corners, dtype=np.float32)
    rect_rotated = cv2.minAreaRect(rotated_corners_array)
    box_rotated = cv2.boxPoints(rect_rotated)
    box_rotated = np.intp(box_rotated)
    
    # Extract bounding box coordinates
    x_coords = box_rotated[:, 0]
    y_coords = box_rotated[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Ensure the bounding box is within the image bounds
    x_min = max(0, x_min)
    x_max = min(image.shape[1], x_max)
    y_min = max(0, y_min)
    y_max = min(image.shape[0], y_max)
    
    # Calculate the width and height of the cropped template
    template_width = x_max - x_min
    template_height = y_max - y_min
    
    # Ensure the bounding box is valid
    if template_width <= 0 or template_height <= 0:
        print(f"Warning: Invalid bounding box ({x_min}, {y_min}, {x_max}, {y_max})")
        return None, None, None, None, None, None
    
    # Crop template
    cropped_template = rotated_image[y_min:y_max, x_min:x_max]
    
    # Calculate the four corner coordinates of the cropped region (coordinates in the rotated image)
    cropped_corners_rotated = [
        (x_min, y_min),  # Top-left
        (x_max, y_min),  # Top-right
        (x_max, y_max),  # Bottom-right
        (x_min, y_max)   # Bottom-left
    ]
    
    # Reverse rotate the four corner coordinates of the cropped region to get the coordinates in the original image
    inverse_rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
    
    # Convert the four corner coordinates of the cropped region back to the original image coordinates
    original_corners = []
    for corner in cropped_corners_rotated:
        point = np.array([corner[0], corner[1], 1])
        original_point = np.dot(inverse_rotation_matrix, point)
        original_corners.append(tuple(original_point))
    
    # Calculate the center point using the restored rectangle corners
    restored_center_x = np.mean([p[0] for p in original_corners])
    restored_center_y = np.mean([p[1] for p in original_corners])
    restored_center = (restored_center_x, restored_center_y)
    
    return cropped_template, rotation_angle, restored_center, template_width, template_height, original_corners

def process_images_batch(image_dir, label_dir, template_output_dir, output_dir=None):
    """Batch process images and labels"""
    # Ensure the output directory exists
    os.makedirs(template_output_dir, exist_ok=True)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create gt.txt file
    gt_file_path = os.path.join(template_output_dir, "gt.txt")
    gt_file = open(gt_file_path, "w", encoding="utf-8")
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image
    for image_path in image_files:
        print(f"Processing image: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot read image: {image_path}")
            continue
            
        img_height, img_width = image.shape[:2]
        print(f"Image size: {img_width}x{img_height}")
        
        # Build the corresponding label file path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, base_name + ".txt")
        
        # Check if the label file exists
        if not os.path.exists(label_path):
            print(f"Label file does not exist: {label_path}")
            continue
        
        # Copy image for drawing
        image_with_boxes = image.copy()
        
        # Read label file
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        valid_lines = lines[2:] # Skip the first two lines
        print(f"Found {len(valid_lines)} templates")
        
        # Template counter
        img_template_count = 0
        
        # Process each template
        for line_idx, line in enumerate(valid_lines):
            parts = line.strip().split()
            if len(parts) < 8:  # Only need 8 coordinate values
                print(f"Invalid label line: {line}")
                continue
            
            # Extract coordinates
            coords = list(map(float, parts[:8]))
            template_corners = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
            
            # Process template data
            cropped_template, rotation_angle, center, template_width, template_height, restored_corners = process_template_data(image, template_corners)
            
            if cropped_template is None:
                print(f"Template {line_idx+1} processing failed, skipping")
                continue
            
            # Check if the template size meets the requirements (18-72 pixels)
            if template_width < 18 or template_width > 72 or template_height < 18 or template_height > 72:
                continue
            
            # Draw the original template box on the image (red)
            image_with_boxes = draw_polygon(image_with_boxes, template_corners, (0, 0, 255), 2)
            
            # Draw the restored bounding box on the image (green)
            image_with_boxes = draw_polygon(image_with_boxes, restored_corners, (0, 255, 0), 2)
            
            # Create template filename
            img_template_count += 1
            template_filename = f"{base_name}_template_{img_template_count:04d}.png"
            template_path = os.path.join(template_output_dir, template_filename)
            
            # Save template image
            if len(cropped_template.shape) == 3 and cropped_template.shape[2] == 3:
                template_rgb = cv2.cvtColor(cropped_template, cv2.COLOR_BGR2RGB)
            else:
                template_rgb = cropped_template
            
            template_pil = Image.fromarray(template_rgb)
            template_pil.save(template_path)
            
            # Write to gt.txt file
            image_name = os.path.basename(image_path)
            center_x, center_y = center
            
            gt_line = f"{image_name} {template_filename} {center_x:.2f} {center_y:.2f} {template_width/36:.3f} {template_height/36:.3f} {rotation_angle:.2f}\n"
            gt_file.write(gt_line)
        
        # Save the drawn image
        if output_dir and img_template_count != 0:
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            
            # Convert BGR to RGB
            if len(image_with_boxes.shape) == 3 and image_with_boxes.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_with_boxes
            
            output_pil = Image.fromarray(image_rgb)
            output_pil.save(output_path)
            print(f"Saved drawn image: {output_path}")
        
        print("-" * 50)
    
    # Close gt.txt file
    gt_file.close()
    print(f"GT file saved: {gt_file_path}")

# Usage example
if __name__ == "__main__":
    # Set paths and parameters
    image_dir = "/workspace/zhouji/dataSets/DOTA/images/images"  # Replace with your image directory
    label_dir = "/workspace/zhouji/dataSets/DOTA/labelTxt" # Replace with your label directory
    template_output_dir = "/workspace/zhouji/dataSets/DOTA/templates" # Cropped template saving directory
    output_dir = "output_images_with_boxes" # Drawn image saving directory (optional)
    
    # Batch processing
    process_images_batch(image_dir, label_dir, template_output_dir)