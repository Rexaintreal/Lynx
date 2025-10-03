import cv2
import numpy as np
import os

def detect_objects(input_path, output_path):
    # Paths to MobileNet-SSD files
    models_dir = os.path.join("models", "pretrained")
    prototxt_path = os.path.join(models_dir, "MobileNetSSD_deploy.prototxt")
    model_path = os.path.join(models_dir, "MobileNetSSD_deploy.caffemodel")
    
    # MobileNet-SSD class labels
    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]
    
    # Generate random colors for each class
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
    # Load the model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # Read image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image at {input_path}")
    
    height, width = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )
    
    net.setInput(blob)
    detections = net.forward()
    
    object_counts = {}
    detections_list = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter by confidence threshold (50%)
        if confidence > 0.5:
            # Get class ID and label
            class_id = int(detections[0, 0, i, 1])
            label = CLASSES[class_id]
            
            if label == "background":
                continue
            
            # Count objects by type
            if label in object_counts:
                object_counts[label] += 1
            else:
                object_counts[label] = 1
            
            detections_list.append({
                'label': label,
                'confidence': round(confidence * 100, 2)
            })
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            
            color = COLORS[class_id].tolist()
            
            # Draw bounding box
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            
            label_text = f"{label}: {confidence:.2f}"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                2
            )
            
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            # Draw background rectangle for text
            cv2.rectangle(
                image,
                (startX, y - text_height - 5),
                (startX + text_width, y),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image,
                label_text,
                (startX, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
    
    # Save processed image
    cv2.imwrite(output_path, image)
    
    return {
        'total_objects': len(detections_list),
        'object_counts': object_counts,
        'detections': detections_list
    }