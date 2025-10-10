import cv2
import numpy as np
import easyocr
import os

# Intialize easyOCR reader globally
reader = None

def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)

    return reader

def preprocess_captcha(img):

    # Conver to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Morphological operations to clean up the captcha
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # Invert if background is dark
    if np.mean(morph) < 127:
        morph = cv2.bitwise_not(morph)

    return morph

def remove_lines(img):

    # Detect hortizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (255, 255, 255), 2)

    # Detect Vertical Lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (255, 255, 255), 2)
    
    return img

def solve_captcha(image_path, output_path, preprocessing_level='medium'):
    # Load img
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    height, width = img.shape[:2]
    original_img = img.copy()

    # Apply preprocessing based on level
    if preprocessing_level == 'light':
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
    elif preprocessing_level == 'medium':
        processed = preprocess_captcha(img)
    else: # heavy
        processed = preprocess_captcha(img)
        processed = remove_lines(processed)

    # Save processed image for visualization
    cv2.imwrite(output_path, processed)

    # get the ocr reader
    ocr_reader = get_reader()

    # Perform OCR on processed image
    results = ocr_reader.readtext(output_path, detail=1)

    # Also try on the original img
    results_original = ocr_reader.readtext(image_path, detail=1)

    # Combine and filter results
    all_results = results + results_original

    # Sort by confidence and remove duplicates
    seen_texts = set()
    
    unique_results = []
    for bbox, text, confidence in sorted(all_results, key=lambda x: x[2], reverse=True):
        text_clean = text.strip().lower()
        if text_clean and text_clean not in seen_texts and confidence > 0.1:
            seen_texts.add(text_clean)
            unique_results.append({
                'text': text.strip(),
                'confidence': round(confidence * 100, 1),
                'source': 'processed' if (bbox, text, confidence) in results else 'original'
            })

    # Create visualization image
    viz_img = original_img.copy()

    if unique_results:
        # Draw the most confident result
        best_result = unique_results[0]

        # Add semi-transparent overlay
        overlay = viz_img.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, viz_img, 0.7, 0, viz_img)

        #  Add solved text
        label = f"SOLVED: {best_result['text']}"
        cv2.putText(
            viz_img, label, (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            viz_img, label, (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA
        )

    # Sve visualization
    viz_filename = output_path.replace('.', '_viz.')
    cv2.imwrite(viz_filename, viz_img)

    # Determine best solution
    if unique_results:
        best_solution = unique_results[0]['text']
        best_confidence = unique_results[0]['confidence']
        status = 'success' if best_confidence > 50 else 'low_confidence'
    else:
        best_solution = None
        best_confidence = 0
        status = 'failed'
    
    return {
        'status': status,
        'solution': best_solution,
        'confidence': best_confidence,
        'all_candidates': unique_results[:5],  # Top 5 candidates
        'preprocessing_used': preprocessing_level,
        'total_candidates': len(unique_results),
        'image_size': f"{width}x{height}",
        'processed_image': os.path.basename(output_path),
        'viz_image': os.path.basename(viz_filename)
    }