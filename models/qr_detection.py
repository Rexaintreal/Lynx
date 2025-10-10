import cv2
import os
from datetime import datetime

def detect_qr_from_image(image_path: str, output_dir: str = "uploads") -> dict:
    try: 
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "message": "Invalid image file"}
        
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(image)

        if points is not None and data:
            # Draw the bounding box
            points = points[0]
            for i in range(len(points)):
                pt1 = tuple(map(int, points[i]))
                pt2 = tuple(map(int, points[(i + 1) % len(points)]))
                cv2.line(image, pt1, pt2, (0, 255, 0), 3)

            # Save the processed image
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"qr_result_{timestamp}.jpg")
            cv2.imwrite(output_path, image)

            return {
                "success": True,
                "data": data,
                "image_path": output_path, 
                "message": "QR Code detected successfully."
            }
        
        else:
            return {"success": False, "message": "No QR code detected."}
        
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}