import cv2
import numpy as np

def apply_filter(input_path, output_path, filter_type="none",**params):
    image = cv2.imread(input_path)

    if image is None:
        raise ValueError("Could not read image")
    
    # apply adjustments
    if filter_type == "adjustable":
        brightness = params.get("brightness", 100)
        contrast = params.get("contrast", 100)
        sepia_amount = params.get("sepia", 0)
        blur_amount = params.get("blur", 0)

        # Apply brightness and contrast adjustments
        image = adjust_brightness_contrast(image, brightness, contrast)

        # Apply blur effect
        if blur_amount > 0:
            kernel_size = int(blur_amount) * 2 + 1
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        # Apply Sepia effect
        if sepia_amount > 0:
            image = apply_sepia(image, sepia_amount / 100.0)
        
    # Presest filters
    elif filter_type == "grayscale":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == "sepia":
        image = apply_sepia(image, 1.0)
    
    elif filter_type == "invert":
        image = cv2.bitwise_not(image)
    
    elif filter_type == "cool":
        # Cool blue tone effect
        image = apply_cool_tone(image)
    
    elif filter_type == "vibrant":
        # Increase saturation for vibrant look
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
        hsv = hsv.astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif filter_type == "edge_detection":
        # Canny edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == "cartoon":
        image = apply_cartoon_effect(image)
    
    elif filter_type == "sketch":
        image = apply_sketch_effect(image)
    
    elif filter_type == "oil_painting":
        # Oil painting effect using bilateral filter
        image = cv2.bilateralFilter(image, 9, 75, 75)
        image = cv2.bilateralFilter(image, 9, 75, 75)
    
    elif filter_type == "sharpen":
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        image = cv2.filter2D(image, -1, kernel)
    
    elif filter_type == "emboss":
        kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]])
        image = cv2.filter2D(image, -1, kernel)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == "warm":
        # Warm orange/yellow tone
        image = apply_warm_tone(image)
    
    elif filter_type == "vintage":
        # Vintage effect (sepia + vignette + noise)
        image = apply_vintage_effect(image)
    
    elif filter_type == "none":
        pass  # No filter
    
    # Save the processed image
    cv2.imwrite(output_path, image)
    return output_path


def adjust_brightness_contrast(image, brightness=100, contrast=100):
    """
    Adjust brightness and contrast using OpenCV.
    brightness: 0-200 (100 is normal)
    contrast: 0-200 (100 is normal)
    """
    # Convert to float
    img_float = image.astype(np.float32)
    
    # Apply brightness (additive)
    brightness_factor = (brightness - 100) * 2.55
    img_float = img_float + brightness_factor
    
    # Apply contrast (multiplicative)
    contrast_factor = contrast / 100.0
    img_float = (img_float - 128) * contrast_factor + 128
    
    # Clip and convert back
    img_float = np.clip(img_float, 0, 255)
    return img_float.astype(np.uint8)


def apply_sepia(image, intensity=1.0):
    """Apply sepia tone filter"""
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    
    sepia_image = cv2.transform(image, kernel)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    
    # Blend with original based on intensity
    if intensity < 1.0:
        sepia_image = cv2.addWeighted(image, 1 - intensity, sepia_image, intensity, 0)
    
    return sepia_image


def apply_cool_tone(image):
    """Apply cool blue tone"""
    # Increase blue channel, decrease red
    b, g, r = cv2.split(image)
    b = np.clip(b.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
    r = np.clip(r.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


def apply_warm_tone(image):
    """Apply warm orange/yellow tone"""
    # Increase red and green, decrease blue
    b, g, r = cv2.split(image)
    r = np.clip(r.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


def apply_cartoon_effect(image):
    """Create cartoon-like effect"""
    # Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 9, 9)
    
    # Color quantization
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    result = result.reshape(image.shape)
    
    # Combine with edges
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(result, edges_colored)
    
    return cartoon


def apply_sketch_effect(image):
    """Create pencil sketch effect"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the image
    inverted = cv2.bitwise_not(gray)
    
    # Blur the inverted image
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    
    # Invert the blurred image
    inverted_blur = cv2.bitwise_not(blurred)
    
    # Create sketch by dividing
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    
    # Convert back to BGR
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    return sketch_bgr


def apply_vintage_effect(image):
    """Apply vintage/retro effect"""
    # Apply sepia
    image = apply_sepia(image, 0.8)
    
    # Create vignette
    rows, cols = image.shape[:2]
    
    # Create radial gradient mask
    X_resultant_kernel = cv2.getGaussianKernel(cols, cols / 2)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows / 2)
    
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()
    
    # Apply vignette
    for i in range(3):
        image[:, :, i] = image[:, :, i] * mask
    
    # Add slight noise for grain
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    # Reduce contrast slightly
    image = adjust_brightness_contrast(image, 100, 80)
    
    return image