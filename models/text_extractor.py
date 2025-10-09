# IF TEXT EXTRACTION IS NOT WORKING PLEASE INSTALL EASY OCR ITS A ONE TIME INSTALL FOR AROUND 300 MBS 

import cv2 
import numpy as np
import easyocr
import os 

# We are initializing easyOCR reader globally to avoid reloading on every call

reader = None

def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have CUDA (NVIDIA GPU MAKES 3-5X faster idk that was in the docs)

    return reader

def extract_text_from_image(image_path, output_path, draw_boxes=True, confidence_threshold=0.25):
    # loading image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    height, width = img.shape[:2]

    # get easyOCR reader

    ocr_reader = get_reader()

    # Perform OCR
    results = ocr_reader.readtext(image_path)

    # Filter by confidence 
    filtered_results = [
        result for result in results
        if result[2] >= confidence_threshold
    ]

    # Preapre output image
    output_img = img.copy()

    # Extract text and draw boces
    extracted_texts = []
    total_confidence = 0

    for detection in filtered_results:
        bbox, text, confidence = detection

        # Store text info
        extracted_texts.append({
            'text': text,
            'confidence': round(confidence * 100, 1)
        })
        total_confidence += confidence

        if draw_boxes: 
            # Convert bbox to integer coordinates
            points = np.array(bbox, dtype=np.int32)

            # draw rectangles
            cv2.polylines(output_img, [points], True, (0, 255, 0), 2)

            # Add text label with background
            label = f"{text} {confidence * 100:.1f}%"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)


            # Draw background rectangle for text
            cv2.rectangle(
                output_img,
                (points[0][0], points[0][1] - text_h - 10),
                (points[0][0] + text_w + 5, points[0][1]),
                (0, 255, 0),
                -1
            )

            # Draw text
            cv2.putText(
                output_img,
                label,
                (points[0][0] + 2, points[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

    # Save output image 
    cv2.imwrite(output_path, output_img)

    # Prepare full text
    full_text = ' '.join([item['text'] for item in extracted_texts])
    
    # Calculate average confidence
    avg_confidence = (total_confidence / len(filtered_results) * 100) if filtered_results else 0
    
    return {
        'total_texts': len(filtered_results),
        'extracted_texts': extracted_texts,
        'full_text': full_text,
        'avg_confidence': round(avg_confidence, 1),
        'image_size': f"{width}x{height}",
        'filtered_count': len(results) - len(filtered_results),
        'note': f"Using EasyOCR with {confidence_threshold*100:.0f}% confidence threshold"
    }