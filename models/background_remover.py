import cv2
import numpy as np

def remove_background(input_path, output_path, method="grabcut", **kwargs):
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Could not read image")
    
    original_shape = img.shape
    output = img.copy()
    
    details = {
        'method': method,
        'dimensions': f"{original_shape[1]}x{original_shape[0]}",
        'processing_info': {}
    }
    
    if method == "grabcut":
        output, info = _grabcut_removal(img, **kwargs)
        details['processing_info'] = info
    
    elif method == "edge":
        output, info = _edge_based_removal(img, **kwargs)
        details['processing_info'] = info
    
    elif method == "color":
        output, info = _color_based_removal(img, **kwargs)
        details['processing_info'] = info
    
    elif method == "contour":
        output, info = _contour_based_removal(img, **kwargs)
        details['processing_info'] = info
    
    elif method == "threshold":
        output, info = _threshold_removal(img, **kwargs)
        details['processing_info'] = info
    
    # Save output
    cv2.imwrite(output_path, output)
    
    return details


def _grabcut_removal(img, iterations=5, margin=10):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Define rectangle around center (assume subject is centered)
    h, w = img.shape[:2]
    rect = (margin, margin, w - margin*2, h - margin*2)
    
    # Apply GrabCut
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    
    # Create binary mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply mask to image
    output = img * mask2[:, :, np.newaxis]
    
    # Convert to RGBA with transparency
    b, g, r = cv2.split(output)
    alpha = mask2 * 255
    output = cv2.merge([b, g, r, alpha])
    
    foreground_pixels = np.sum(mask2)
    total_pixels = mask2.size
    
    return output, {
        'algorithm': 'GrabCut',
        'iterations': iterations,
        'foreground_percentage': f"{(foreground_pixels/total_pixels)*100:.1f}%",
        'margin': margin
    }


def _edge_based_removal(img, canny_low=50, canny_high=150, dilate_iter=3):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # Dilate edges to connect components
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=dilate_iter)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask from largest contour
    mask = np.zeros(img.shape[:2], np.uint8)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    # Apply mask
    b, g, r = cv2.split(img)
    alpha = mask
    output = cv2.merge([b, g, r, alpha])
    
    return output, {
        'algorithm': 'Edge Detection',
        'canny_thresholds': f"{canny_low}-{canny_high}",
        'contours_found': len(contours),
        'dilate_iterations': dilate_iter
    }


def _color_based_removal(img, bg_color=(255, 255, 255), threshold=40):
    # Convert BGR to RGB for comparison
    bg_color_bgr = (bg_color[2], bg_color[1], bg_color[0])
    
    # Calculate color difference
    diff = np.abs(img.astype(np.float32) - np.array(bg_color_bgr))
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Create mask
    mask = (dist > threshold).astype(np.uint8) * 255
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Apply Gaussian blur to soften edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Apply mask
    b, g, r = cv2.split(img)
    alpha = mask
    output = cv2.merge([b, g, r, alpha])
    
    removed_pixels = np.sum(mask == 0)
    total_pixels = mask.size
    
    return output, {
        'algorithm': 'Color-based',
        'target_color': f"RGB{bg_color}",
        'threshold': threshold,
        'background_removed': f"{(removed_pixels/total_pixels)*100:.1f}%"
    }


def _contour_based_removal(img, min_area=1000):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and create mask
    mask = np.zeros(img.shape[:2], np.uint8)
    valid_contours = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.drawContours(mask, [contour], -1, 255, -1)
            valid_contours += 1
    
    # Apply mask
    b, g, r = cv2.split(img)
    alpha = mask
    output = cv2.merge([b, g, r, alpha])
    
    return output, {
        'algorithm': 'Contour-based',
        'total_contours': len(contours),
        'valid_contours': valid_contours,
        'min_area': min_area
    }


def _threshold_removal(img, threshold_value=127):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding if threshold_value is 0
    if threshold_value == 0:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        method_used = "Otsu's automatic"
    else:
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        method_used = f"Manual ({threshold_value})"
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Apply mask
    b, g, r = cv2.split(img)
    alpha = mask
    output = cv2.merge([b, g, r, alpha])
    
    return output, {
        'algorithm': 'Threshold-based',
        'method': method_used,
        'threshold_value': threshold_value
    }