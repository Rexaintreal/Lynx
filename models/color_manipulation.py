import cv2
import numpy as np

def change_background_color(image_path, output_path, target_color=(255, 255, 255), threshold=50):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    # convert to RGB for processing 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # gET CORNER PIXELS TO ESTIMATE BACKGROUND COLOR
    h, w = img_rgb.shape[:2]
    corners = [
        img_rgb[0, 0],
        img_rgb[0, w-1],
        img_rgb[h-1, 0],
        img_rgb[h-1, w-1]
    ]
    bg_color = np.mean(corners, axis=0).astype(np.uint8)

    # Create mask for background
    diff = np.abs(img_rgb.astype(np.int16) - bg_color.astype(np.int16))
    mask = np.all(diff < threshold, axis=2).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # create output image
    result = img_rgb.copy()
    result[mask == 255] = target_color

    # Convert back to BGR fro saving
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)

    return {
        'original_bg_color': bg_color.tolist(),
        'new_bg_color': target_color,
        'pixels_changed': np.sum(mask == 255)
    }

def replace_color(image_path, output_path, source_color=(255, 0, 0), target_color=(0, 255, 0), tolerance=30):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to HSV for better color matching
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    source_hsv = cv2.cvtColor(np.uint8([[source_color[::-1]]]), cv2.COLOR_BGR2HSV)[0][0]

    # Create mask for soruce color
    lower_bound = np.array([max(0, source_hsv[0] - tolerance), 50, 50])
    upper_bound = np.array([min(179, source_hsv[0] + tolerance), 255, 255])
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)

    # Apply morphological operations 
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # replace color
    result = img_rgb.copy()
    result[mask > 0] = target_color

    # Convert back to BGR 
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)

    return {
        'source_color': source_color,
        'target_color': target_color,
        'pixels_changed': np.sum(mask > 0)
    }

def adjust_saturation(image_path, output_path, saturation_scale=1.5):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv = img_hsv.astype(np.float32)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation_scale, 0, 255)

    img_hsv = img_hsv.astype(np.uint8)
    result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, result)

    return {
        'saturation_scale': saturation_scale,
        'avg_saturation_before': np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]),
        'avg_saturation_after': np.mean(img_hsv[:, :, 1])
    }

def shift_hue(image_path, output_path, hue_shift=30):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
  # Convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    
    # Shift hue (OpenCV hue range is 0-179)
    hue_shift_cv = int(hue_shift / 2)  # Convert from 0-360 to 0-179 range
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift_cv) % 180
    
    # Convert back to BGR
    img_hsv = img_hsv.astype(np.uint8)
    result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
    cv2.imwrite(output_path, result)
    
    return {
        'hue_shift': hue_shift,
        'avg_hue_before': np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]) * 2,
        'avg_hue_after': np.mean(img_hsv[:, :, 0]) * 2
    }


def apply_color_manipulation(image_path, output_path, operation, **kwargs):
    if operation == 'background':
        target_color = kwargs.get('target_color', (255, 255, 255))
        threshold = kwargs.get('threshold', 50)
        return change_background_color(image_path, output_path, target_color, threshold)
    
    elif operation == 'replace':
        source_color = kwargs.get('source_color', (255, 0, 0))
        target_color = kwargs.get('target_color', (0, 255, 0))
        tolerance = kwargs.get('tolerance', 30)
        return replace_color(image_path, output_path, source_color, target_color, tolerance)
    
    elif operation == 'saturation':
        saturation_scale = kwargs.get('saturation_scale', 1.5)
        return adjust_saturation(image_path, output_path, saturation_scale)
    
    elif operation == 'hue':
        hue_shift = kwargs.get('hue_shift', 30)
        return shift_hue(image_path, output_path, hue_shift)
    
    else:
        raise ValueError(f"Unknown operation: {operation}")