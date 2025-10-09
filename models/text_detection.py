import cv2
import numpy as np
from typing import List, Tuple, Dict

def detect_text_regions(image_path: str, output_path: str, 
                       detection_method: str = "mser") -> Dict:
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    orig = image.copy()
    height, width = image.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    text_regions = []
    
    if detection_method == "mser":
        text_regions = detect_text_mser(gray, image)
    elif detection_method == "contour":
        text_regions = detect_text_contour(gray, image)
    else:  # basic
        text_regions = detect_text_basic(gray, image)
    
    # Draw bounding boxes on original image
    for (x, y, w, h) in text_regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.1, image, 0.9, 0, image)
    
    # Save result
    cv2.imwrite(output_path, image)
    
    # Prepare details
    details = {
        "method": detection_method,
        "total_regions": len(text_regions),
        "image_size": f"{width}x{height}",
        "regions": text_regions,
        "avg_region_size": calculate_avg_region_size(text_regions) if text_regions else 0
    }
    
    return details


def detect_text_mser(gray: np.ndarray, image: np.ndarray) -> List[Tuple[int, int, int, int]]:

    # Create MSER detector
    mser = cv2.MSER_create()
    mser.setMinArea(100)
    mser.setMaxArea(10000)
    
    # Detect regions
    regions, _ = mser.detectRegions(gray)
    
    # Convert regions to bounding boxes
    bboxes = []
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    
    for hull in hulls:
        x, y, w, h = cv2.boundingRect(hull)
        # Filter based on aspect ratio 
        aspect_ratio = w / float(h) if h > 0 else 0
        if 0.1 < aspect_ratio < 10 and w > 10 and h > 10:
            bboxes.append((x, y, w, h))
    
    # Merge overlapping boxes
    bboxes = merge_boxes(bboxes)
    
    return bboxes


def detect_text_contour(gray: np.ndarray, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter based on size and aspect ratio
        aspect_ratio = w / float(h) if h > 0 else 0
        if w > 20 and h > 10 and 0.5 < aspect_ratio < 20:
            bboxes.append((x, y, w, h))
    
    return bboxes


def detect_text_basic(gray: np.ndarray, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter based on size
        if w > 30 and h > 15:
            bboxes.append((x, y, w, h))
    
    return bboxes


def merge_boxes(boxes: List[Tuple[int, int, int, int]], 
                overlap_thresh: float = 0.3) -> List[Tuple[int, int, int, int]]:
    if len(boxes) == 0:
        return []
    
    # Convert to box format
    boxes_array = np.array([(x, y, x+w, y+h) for x, y, w, h in boxes])
    
    # Simple grouping based on proximity
    merged = []
    used = set()
    
    for i, box in enumerate(boxes):
        if i in used:
            continue
        
        x, y, w, h = box
        group = [box]
        used.add(i)
        
        for j, other_box in enumerate(boxes):
            if j in used or i == j:
                continue
            
            ox, oy, ow, oh = other_box
            
            # Check if boxes are close
            if (abs(x - ox) < 50 and abs(y - oy) < 30):
                group.append(other_box)
                used.add(j)
        
        # Merge group into single box
        if group:
            min_x = min(b[0] for b in group)
            min_y = min(b[1] for b in group)
            max_x = max(b[0] + b[2] for b in group)
            max_y = max(b[1] + b[3] for b in group)
            merged.append((min_x, min_y, max_x - min_x, max_y - min_y))
    
    return merged


def calculate_avg_region_size(regions: List[Tuple[int, int, int, int]]) -> int:
    if not regions:
        return 0
    total_area = sum(w * h for x, y, w, h in regions)
    return total_area // len(regions)


def extract_text_basic(image_path: str, output_path: str) -> Dict:
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing for better text visibility
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)
    
    # Convert back to BGR for consistency
    result = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    # Save result
    cv2.imwrite(output_path, result)
    
    # Detect text regions for statistics
    text_regions = detect_text_basic(gray, image)
    
    details = {
        "preprocessed": True,
        "total_regions": len(text_regions),
        "image_size": f"{image.shape[1]}x{image.shape[0]}",
        "note": "Image preprocessed for better text readability"
    }
    
    return details