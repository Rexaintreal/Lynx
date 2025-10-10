import cv2
import numpy as np
import easyocr
import os

def extract_numberplates(input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Could not read image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edges = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    plates = []
    reader = easyocr.Reader(['en'], gpu=False)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            if 2 < aspect_ratio < 6 and w > 80 and h > 25:
                plate_region = image[y:y+h, x:x+w]
                ocr_result = reader.readtext(plate_region)

                if ocr_result:
                    for res in ocr_result:
                        text = res[1].strip().replace(" ", "")
                        if len(text) >= 5:
                            plates.append(text)
                            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(image, text, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)

    return {
        "detected_plates": list(set(plates)),
        "output_path": output_path
    }
