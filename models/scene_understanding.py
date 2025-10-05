import cv2
import numpy as np
import os

def analyze_scene(input_path, output_path):
        #path to mobilenet ssd files
    models_dir = os.path.join("models", "pretrained")
    prototxt_path = os.path.join(models_dir, "MobileNetSSD_deploy.prototxt")
    model_path = os.path.join(models_dir, "MobileNetSSD_deploy.caffemodel")
    
    # MobileNetSSD class labels
    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]
    
    # Indoor/Outdoor indicators
    OUTDOOR_OBJECTS = {"aeroplane", "bicycle", "bird", "boat", "bus", "car", "cow", "horse", "sheep", "train"}
    INDOOR_OBJECTS = {"bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"}
    
    # Generate colors for each class
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
    # Load the model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # Read image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image at {input_path}")
    
    height, width = image.shape[:2]
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Sky detection
    upper_region = hsv[:int(height * 0.3), :]
    blue_mask = cv2.inRange(upper_region, np.array([100, 50, 50]), np.array([130, 255, 255]))
    sky_ratio = np.sum(blue_mask > 0) / blue_mask.size
    
    # Green detection 
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    green_ratio = np.sum(green_mask > 0) / (height * width)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (height * width)
    
    # Brightness analysis
    brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0])
    
    # Object detection
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
    outdoor_score = 0
    indoor_score = 0
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            label = CLASSES[class_id]
            
            if label == "background":
                continue
            
            # Count objects
            if label in object_counts:
                object_counts[label] += 1
            else:
                object_counts[label] = 1
            
            # Score for indoor/outdoor
            if label in OUTDOOR_OBJECTS:
                outdoor_score += 1
            if label in INDOOR_OBJECTS:
                indoor_score += 1
            
            detections_list.append({
                'label': label,
                'confidence': round(float(confidence) * 100, 2)
            })
            
            # Draw bounding boxes
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            
            color = COLORS[class_id].tolist()
            
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            
            label_text = f"{label}: {confidence:.2f}"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                2
            )
            
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            cv2.rectangle(
                image,
                (startX, y - text_height - 5),
                (startX + text_width, y),
                color,
                -1
            )
            
            cv2.putText(
                image,
                label_text,
                (startX, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
    
    # Determine scene type using multiple factors
    scene_factors = {
        'sky': sky_ratio > 0.15,
        'green': green_ratio > 0.2,
        'edge_density': edge_density > 0.1,
        'brightness': brightness > 120,
        'objects': outdoor_score > indoor_score
    }
    
    outdoor_indicators = sum(scene_factors.values())
    
    if outdoor_indicators >= 3:
        scene_type = "outdoor"
        confidence = float(min(95, 60 + (outdoor_indicators * 7)))
    elif outdoor_indicators <= 1:
        scene_type = "indoor"
        confidence = float(min(95, 60 + ((5 - outdoor_indicators) * 7)))
    else:
        if outdoor_score > indoor_score:
            scene_type = "outdoor"
            confidence = float(55 + (outdoor_score * 5))
        else:
            scene_type = "indoor"
            confidence = float(55 + (indoor_score * 5))
    
    # Add scene type label to image
    scene_label = f"Scene: {scene_type.upper()} ({int(confidence)}%)"
    label_bg_color = (46, 125, 50) if scene_type == "outdoor" else (158, 90, 35)
    
    cv2.rectangle(image, (10, 10), (10 + 250, 50), label_bg_color, -1)
    cv2.putText(
        image,
        scene_label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    # Save processed image
    cv2.imwrite(output_path, image)
    
    # Calculate spatial distribution
    spatial_dist = {
        'top': 0,
        'middle': 0,
        'bottom': 0,
        'left': 0,
        'center': 0,
        'right': 0
    }
    
    for i in range(detections.shape[2]):
        detection_conf = detections[0, 0, i, 2] = detections[0, 0, i, 2]
        if detection_conf > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if CLASSES[class_id] != "background":
                box = detections[0, 0, i, 3:7]
                center_y = (box[1] + box[3]) / 2
                center_x = (box[0] + box[2]) / 2
                
                # Vertical position
                if center_y < 0.33:
                    spatial_dist['top'] += 1
                elif center_y < 0.66:
                    spatial_dist['middle'] += 1
                else:
                    spatial_dist['bottom'] += 1
                
                # Horizontal position
                if center_x < 0.33:
                    spatial_dist['left'] += 1
                elif center_x < 0.66:
                    spatial_dist['center'] += 1
                else:
                    spatial_dist['right'] += 1
    
    return {
        'scene_type': scene_type,
        'scene_confidence': int(confidence),
        'total_objects': len(detections_list),
        'object_counts': object_counts,
        'detections': detections_list,
        'spatial_distribution': spatial_dist,
        'scene_analysis': {
            'sky_detected': bool(scene_factors['sky']),
            'vegetation_detected': bool(scene_factors['green']),
            'edge_density': float(edge_density),
            'brightness_level': float(brightness)
        }
    }