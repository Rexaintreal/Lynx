import cv2
import numpy as np

def apply_filter(input_path, output_path, filter_type="none"):
    """Apply various filters to an image using OpenCV"""
    image = cv2.imread(input_path)
    
    if image is None:
        raise ValueError("Could not read image")
    
    if filter_type == "grayscale":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == "sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        image = cv2.transform(image, kernel)
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    elif filter_type == "invert":
        image = cv2.bitwise_not(image)
    
    elif filter_type == "cool":
        # Apply cool blue tone
        image = cv2.applyColorMap(image, cv2.COLORMAP_OCEAN)
    
    elif filter_type == "vibrant":
        # Increase saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(int) + 60, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif filter_type == "none":
        # No filter, keep original
        pass
    
    # Save the processed image
    cv2.imwrite(output_path, image)
    return output_path