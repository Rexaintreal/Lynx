import cv2
import numpy as np
from typing import Tuple, List, Optional

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4,2), dtype="float32")

    # sum and diff to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)] 
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate width of new image
    widthA = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    widthB = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    maxWidth = max(int(widthA), int(widthB))

    # Calculate height of a new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # destination points for prespective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Calculate prespective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def detect_document_edges(image_path: str) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    # Convert to grayscale and apply gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(blurred, 75, 200)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Fidn document contour 
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is not None:
        # Scale back to original size
        screenCnt = screenCnt.reshape(4, 2) * ratio
    
    return screenCnt, orig, edged


def enhance_text_readability(image: np.ndarray) -> np.ndarray:
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply adaptive thresholding for better text visibility
    enhanced = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )   

    # Denoise
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10 , 7, 21)

    return enhanced  


def convert_to_bw(image: np.ndarray, threshold: int = 127) -> np.ndarray:

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return bw

def remove_shadows(image: np.ndarray) -> np.ndarray:
    # Convert to grayscale 
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: 
        gray = image.copy()

    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))

    # blur to smooth
    bg = cv2.medianBlur(dilated, 21)

    # Calculate difference
    diff = 255 - cv2.absdiff(gray, bg)

    # Noralize
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return norm

def scan_document(image_path: str, output_path: str, 
                 mode: str = "enhanced") -> dict:
    # Detect document edges
    document_contour, original, edged = detect_document_edges(image_path)

    if document_contour is None:
        # If no document detected, process the whole image
        print("Warning: Could not detect document edges. Processing entire image.")
        warped = original.copy()
        detected_edges = False

    else:
        # Apply prespective transform (dewrap)
        warped = four_point_transform(original, document_contour)
        detected_edges = True

    # Apply processing mode
    if mode == "enhanced":
        result = enhance_text_readability(warped)
        # Convert back to BGR for consistency
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif mode == "bw":
        result = convert_to_bw(warped)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif mode == "shadow_removed":
        result = remove_shadows(warped)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    else:  # original
        result = warped

        # Save result
    cv2.imwrite(output_path, result)
    
    # Prepare details
    details = {
        "edges_detected": detected_edges,
        "mode": mode,
        "original_size": f"{original.shape[1]}x{original.shape[0]}",
        "processed_size": f"{result.shape[1]}x{result.shape[0]}",
        "perspective_corrected": detected_edges
    }
    
    return details