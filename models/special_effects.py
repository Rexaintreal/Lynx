import cv2
import numpy as np

def green_screen_effect(image_path, output_path, bg_color=(0, 255, 0), new_bg_color=(255, 255, 255), threshold=40):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    bg_hsv = cv2.cvtColor(np.uint8([[bg_color[::-1]]]), cv2.COLOR_BGR2HSV)[0][0]
    
    lower_bound = np.array([max(0, bg_hsv[0] - threshold), 50, 50])
    upper_bound = np.array([min(179, bg_hsv[0] + threshold), 255, 255])
    
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    result = img_rgb.copy()
    result[mask > 0] = new_bg_color
    
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    
    return {
        'screen_color': bg_color,
        'new_background': new_bg_color,
        'pixels_replaced': np.sum(mask > 0)
    }

def color_pop_effect(image_path, output_path, keep_color=(255, 0, 0), tolerance=30):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    keep_hsv = cv2.cvtColor(np.uint8([[keep_color[::-1]]]), cv2.COLOR_BGR2HSV)[0][0]
    
    lower_bound = np.array([max(0, keep_hsv[0] - tolerance), 50, 50])
    upper_bound = np.array([min(179, keep_hsv[0] + tolerance), 255, 255])
    
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    result = gray_3channel.copy()
    result[mask > 0] = img[mask > 0]
    
    cv2.imwrite(output_path, result)
    
    return {
        'preserved_color': keep_color,
        'colored_pixels': np.sum(mask > 0),
        'grayscale_pixels': np.sum(mask == 0)
    }

def duotone_effect(image_path, output_path, shadow_color=(0, 0, 139), highlight_color=(255, 215, 0)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    normalized = gray.astype(np.float32) / 255.0
    
    shadow_color = np.array(shadow_color, dtype=np.float32)
    highlight_color = np.array(highlight_color, dtype=np.float32)
    
    result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    for i in range(3):
        result[:, :, i] = shadow_color[i] * (1 - normalized) + highlight_color[i] * normalized
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, result_bgr)
    
    return {
        'shadow_color': shadow_color.tolist(),
        'highlight_color': highlight_color.tolist(),
        'avg_intensity': np.mean(gray)
    }

def color_isolation_effect(image_path, output_path, isolate_color=(255, 0, 0), tolerance=30, desaturation=0.3):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    isolate_hsv = cv2.cvtColor(np.uint8([[isolate_color[::-1]]]), cv2.COLOR_BGR2HSV)[0][0]
    
    lower_bound = np.array([max(0, isolate_hsv[0] - tolerance), 50, 50])
    upper_bound = np.array([min(179, isolate_hsv[0] + tolerance), 255, 255])
    
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    result_hsv = img_hsv.copy().astype(np.float32)
    result_hsv[mask == 0, 1] = result_hsv[mask == 0, 1] * desaturation
    result_hsv[mask == 0, 2] = result_hsv[mask == 0, 2] * 0.9
    
    result_hsv = np.clip(result_hsv, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
    
    cv2.imwrite(output_path, result)
    
    return {
        'isolated_color': isolate_color,
        'isolated_pixels': np.sum(mask > 0),
        'desaturated_pixels': np.sum(mask == 0)
    }

def apply_special_effect(image_path, output_path, effect, **kwargs):
    if effect == 'greenscreen':
        bg_color = kwargs.get('bg_color', (0, 255, 0))
        new_bg_color = kwargs.get('new_bg_color', (255, 255, 255))
        threshold = kwargs.get('threshold', 40)
        return green_screen_effect(image_path, output_path, bg_color, new_bg_color, threshold)
    
    elif effect == 'colorpop':
        keep_color = kwargs.get('keep_color', (255, 0, 0))
        tolerance = kwargs.get('tolerance', 30)
        return color_pop_effect(image_path, output_path, keep_color, tolerance)
    
    elif effect == 'duotone':
        shadow_color = kwargs.get('shadow_color', (0, 0, 139))
        highlight_color = kwargs.get('highlight_color', (255, 215, 0))
        return duotone_effect(image_path, output_path, shadow_color, highlight_color)
    
    elif effect == 'isolation':
        isolate_color = kwargs.get('isolate_color', (255, 0, 0))
        tolerance = kwargs.get('tolerance', 30)
        desaturation = kwargs.get('desaturation', 0.3)
        return color_isolation_effect(image_path, output_path, isolate_color, tolerance, desaturation)
    
    else:
        raise ValueError(f"Unknown effect: {effect}")