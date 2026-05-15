import cv2, math
import numpy as np


def rotate_template(template, angle, scale_x, scale_y):
    """Rotates and scales the template, avoiding cropping (warp output size is extended by sqrt(2))."""
    th, tw = template.shape[:2]

    # Scale dimensions
    scaled_th = int(th * scale_y)
    scaled_tw = int(tw * scale_x)
    scaled_template = cv2.resize(template, (scaled_tw, scaled_th))
    mask = np.ones((scaled_th, scaled_tw), dtype=np.uint8)

    # Calculate the safe boundary size after scaling (diagonal bounding box)
    diagonal_ratio = math.sqrt(2)
    safe_h = int(scaled_th * diagonal_ratio)
    safe_w = int(scaled_tw * diagonal_ratio)

    # Ensure dimensions are even for easier center alignment later
    safe_h += safe_h % 2
    safe_w += safe_w % 2

    # New center
    new_center = (safe_w // 2, safe_h // 2)

    # Calculate affine matrix: from scaled template center -> new center
    old_center = (scaled_tw // 2, scaled_th // 2)
    M = cv2.getRotationMatrix2D(old_center, angle, 1.0)
    M[0, 2] += new_center[0] - old_center[0]
    M[1, 2] += new_center[1] - old_center[1]

    rotated_template = cv2.warpAffine(scaled_template, M, (safe_w, safe_h), borderValue=0)
    rotated_mask = cv2.warpAffine(mask, M, (safe_w, safe_h), borderValue=0)

    return rotated_mask, rotated_template


def extract_subpixel_patch(image, cx, cy, patch_w, patch_h):
    """Extracts a patch centered at (cx, cy) from the image, supporting subpixel precision."""
    dst_center = (patch_w / 2, patch_h / 2)

    # Construct translation matrix to move the target point to the center
    dx = dst_center[0] - cx
    dy = dst_center[1] - cy
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)

    patch = cv2.warpAffine(image, M, (patch_w, patch_h), flags=cv2.INTER_LINEAR, borderValue=0)
    return patch


def compute_masked_similarity(search_img, rotated_template, mask, cx, cy):
    """Calculates similarity only within the corresponding template region."""
    
    patch = extract_subpixel_patch(search_img, cx, cy, rotated_template.shape[1], rotated_template.shape[0])

    if len(patch.shape) == 3:
        masked_patch = patch * mask[..., np.newaxis]
        patch_pixels = masked_patch[mask > 0].flatten().astype(np.float32)
        template_pixels = rotated_template[mask > 0].flatten().astype(np.float32)
    else:
        masked_patch = patch * mask
        patch_pixels = masked_patch[mask > 0].astype(np.float32)
        template_pixels = rotated_template[mask > 0].astype(np.float32)

    dot_product = np.dot(patch_pixels, template_pixels)
    norm1 = np.linalg.norm(patch_pixels)
    norm2 = np.linalg.norm(template_pixels)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def refine_angle(search_img, template, pred_x, pred_y, pred_scale_x, pred_scale_y, pred_angle,
                         initial_range=15.0, step_size=1.5, imageShape=(224, 224), threshold=0.9):
    """Adaptive range angle refinement."""
    
    def search_angles(angle_range, step):
        """Searches for the best angle within a specified range."""
        angle_candidates = np.arange(
            pred_angle - angle_range, 
            pred_angle + angle_range + step, 
            step
        )
        
        best_score = -1
        best_angle = pred_angle
        
        for angle in angle_candidates:
            # Ensure the angle is in the [-180, 180] range
            normalized_angle = ((angle + 180) % 360) - 180
            
            mask, transformed_template = rotate_template(
                template, normalized_angle, 
                pred_scale_x, pred_scale_y
            )
            
            score = compute_masked_similarity(search_img, transformed_template, mask, pred_x, pred_y)
            
            if score > best_score:
                best_score = score
                best_angle = normalized_angle
        
        return best_angle, best_score
    
    # Stage 1: Search within the initial range
    best_angle, best_score = search_angles(initial_range, step_size)
    
    # If a result meeting the threshold is found in the initial range, return directly
    if best_score >= threshold:
        return best_angle, best_score
    
    # Stage 2: Expand to a full-range search
    # print(f"Initial search failed (score={best_score:.3f}), expanding to full range...")
    coarse_angle, coarse_score = search_angles(180.0, 10)
    if coarse_score > best_score:
        coarse_angle, coarse_score = search_angles(10, 1)
    
    # If the coarse search finds a better result
    if coarse_score > threshold:
        return coarse_angle, coarse_score
    
    # If the full-range search doesn't improve, return the original prediction
    return pred_angle, best_score


def refine_angle_bisection(search_img, template, pred_x, pred_y, pred_scale_x, pred_scale_y, pred_angle,
                           initial_range=20.0, coarse_threshold=2.0, fine_step=0.5, imageShape=(224, 224), threshold=0.9):
    """Angle refinement using bisection method (efficient version)."""

    def get_score(angle):
        """Calculates the matching score for a given angle."""
        normalized_angle = ((angle + 180) % 360) - 180
        mask, transformed_template = rotate_template(
            template,normalized_angle,
            pred_scale_x, pred_scale_y
        )
        return compute_masked_similarity(search_img, transformed_template, mask, pred_x, pred_y)

    # Initial left and right boundaries
    left = pred_angle - initial_range
    right = pred_angle + initial_range
    
    score_left = get_score(left)
    score_right = get_score(right)

    # Coarse search stage: Bisection
    while (right - left) > coarse_threshold:
        if score_left > score_right:
            right = (left + right) / 2
            score_right = get_score(right)
        else:
            left = (left + right) / 2
            score_left = get_score(left)

    # Fine search stage: Fine-grained traversal
    fine_angles = np.arange(left, right + fine_step, fine_step)
    best_score = -1
    best_angle = pred_angle

    for angle in fine_angles:
        score = get_score(angle)
        if score > best_score:
            best_score = score
            best_angle = angle
            
    if best_score > threshold:
        # If no result meets the threshold, return the original prediction
        return best_angle, best_score

    return refine_angle(search_img, template, pred_x, pred_y, pred_scale_x, pred_scale_y, pred_angle, 
                 initial_range=initial_range*2, step_size=1, imageShape=imageShape, threshold=threshold)


def refine_scale_x(search_img, template, pred_x, pred_y, pred_scale_x, pred_scale_y, refined_angle,
                   search_range=0.21, step_size=0.03, threshold=0.92):
    """Refines scaling in the X direction."""
    
    # Generate candidate scale values
    scale_candidates = np.arange(
        pred_scale_x - search_range,
        pred_scale_x + search_range + step_size,
        step_size
    )
    
    # Limit the scale range to a reasonable interval
    scale_candidates = scale_candidates[scale_candidates > 0.3]
    scale_candidates = scale_candidates[scale_candidates < 3.0]
    
    best_score = -1
    best_scale_x = pred_scale_x
    
    for scale_x in scale_candidates:
        mask, transformed_template = rotate_template(
            template, refined_angle, 
            scale_x, pred_scale_y
        )
        
        score = compute_masked_similarity(search_img, transformed_template, mask, pred_x, pred_y)
        
        if score > best_score:
            best_score = score
            best_scale_x = scale_x
    
    # If the improvement is not significant, return the original value
    if best_score < threshold:
        return pred_scale_x, best_score
    
    return best_scale_x, best_score


def refine_scale_y(search_img, template, pred_x, pred_y, refined_scale_x, pred_scale_y, refined_angle,
                   search_range=0.15, step_size=0.015,  threshold=0.92):
    """Refines scaling in the Y direction."""
    
    # Generate candidate scale values
    scale_candidates = np.arange(
        pred_scale_y - search_range,
        pred_scale_y + search_range + step_size,
        step_size
    )
    
    # Limit the scale range to a reasonable interval
    scale_candidates = scale_candidates[scale_candidates > 0.3]
    scale_candidates = scale_candidates[scale_candidates < 3.0]
    
    best_score = -1
    best_scale_y = pred_scale_y
    
    for scale_y in scale_candidates:
        mask, transformed_template = rotate_template(
            template, refined_angle, 
            refined_scale_x, scale_y
        )
        
        score = compute_masked_similarity(search_img, transformed_template, mask, pred_x, pred_y)
        
        if score > best_score:
            best_score = score
            best_scale_y = scale_y
    
    # If the improvement is not significant, return the original value
    if best_score < threshold:
        return pred_scale_y, best_score
    
    return best_scale_y, best_score


def refine_position(search_img, template, pred_x, pred_y, refined_scale_x, refined_scale_y, refined_angle,
                   search_range=0.5, step_size=0.1, threshold=0.95):
    """Refines position: Combined optimization of x and y."""
    
    # Generate offsets in x and y directions
    # From -0.9 to 0.9, step 0.3: [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]
    offsets = np.arange(-search_range, search_range + step_size, step_size)
    
    best_score = -1
    best_x = pred_x
    best_y = pred_y
    
    # Iterate through all position combinations
    for dx in offsets:
        for dy in offsets:
            # Calculate new position
            new_x = pred_x + dx
            new_y = pred_y + dy
            
            mask, transformed_template = rotate_template(
                template, refined_angle, 
                refined_scale_x, refined_scale_y
            )
            
            score = compute_masked_similarity(search_img, transformed_template, mask, new_x, new_y)
            
            if score > best_score:
                best_score = score
                best_x = new_x
                best_y = new_y
    
    # If the improvement is not significant, return the original values
    if best_score < threshold:
        return pred_x, pred_y, best_score
    
    return best_x, best_y, best_score