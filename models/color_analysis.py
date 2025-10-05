import cv2 
import numpy as np
import os 
from collections import Counter

def analyze_colors(input_path, output_path):
    # read image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read the image at {input_path}")
    
    height, width = image.shape[:2]

    # converting to RGB for display 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape image to be a list of pixels
    pixels = image_rgb.reshape(-1, 3)

    # using K-means clustering to find the dominant colors
    from sklearn.cluster import KMeans

    n_colors = 10
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Get the colors and their frequencies 
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    # count occurrences of each cluster 
    label_counts = Counter(labels)
    total_pixels = len(labels)

    # creating Color Paletter with percentages 
    color_palette = []
    for i in range(n_colors):
        count = label_counts[i]
        percentage = (count / total_pixels) * 100
        color_palette.append({
            'rgb': colors[i].tolist(),
            'hex': '#{:02x}{:02x}'.format(colors[i][0], colors[i][1], colors[i][2]),
            'percentage': round(percentage, 2),
            'pixel_count': int(count)
        })
    
    # sorting by percentage (decening order)
    color_palette.sort(key=lambda x: x['percentage'], reverse=True)

    # Calculate color histograms for RGB channels
    hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
    
    # Normalize histograms
    hist_r = (hist_r / hist_r.max() * 100).flatten().tolist()
    hist_g = (hist_g / hist_g.max() * 100).flatten().tolist()
    hist_b = (hist_b / hist_b.max() * 100).flatten().tolist()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Average hue, saturation, value
    avg_hue = float(np.mean(hsv[:, :, 0]))
    avg_saturation = float(np.mean(hsv[:, :, 1]))
    avg_value = float(np.mean(hsv[:, :, 2]))

    # Determining dominant color category
    if avg_saturation < 50:
        if avg_value < 85:
            color_mood = "Dark & Muted"
        elif avg_value > 170:
            color_mood = "Bright & Neutral"
        else:
            color_mood = "Neutral"
    else:
        if avg_value < 85:
            color_mood = "Dark & Vibrant"
        elif avg_value > 170:
            color_mood = "Bright & Vibrant"
        else:
            color_mood = "Colorful"

    
     # Calculate color temperature (warm vs cool)
    # Based on red vs blue dominance
    red_mean = float(np.mean(image_rgb[:, :, 0]))
    blue_mean = float(np.mean(image_rgb[:, :, 2]))
    
    if red_mean > blue_mean + 20:
        color_temperature = "Warm"
    elif blue_mean > red_mean + 20:
        color_temperature = "Cool"
    else:
        color_temperature = "Neutral"
    
    # Create visualization image with color bars
    bar_height = 60
    palette_image = np.zeros((bar_height * len(color_palette[:5]), width, 3), dtype=np.uint8)
    
    for idx, color_info in enumerate(color_palette[:5]):
        color = color_info['rgb']
        y_start = idx * bar_height
        y_end = (idx + 1) * bar_height
        palette_image[y_start:y_end, :] = color
        
        # Add text with color info
        text = f"{color_info['percentage']:.1f}% - {color_info['hex']}"
        
        # Choose text color based on brightness
        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        
        cv2.putText(
            palette_image,
            text,
            (20, y_start + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2
        )
    
    # Combine original image with palette
    # Rezise
    if palette_image.shape[1] != image_rgb.shape[1]:
        palette_image = cv2.resize(palette_image, (image_rgb.shape[1], palette_image.shape[0]))
    
    combined = np.vstack([image_rgb, palette_image])
    
    # Converting back to BGR for saving
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, combined_bgr)
    
    return {
        'color_palette': color_palette,
        'color_mood': color_mood,
        'color_temperature': color_temperature,
        'avg_hue': round(avg_hue, 1),
        'avg_saturation': round(avg_saturation, 1),
        'avg_value': round(avg_value, 1),
        'histograms': {
            'red': [round(v, 2) for v in hist_r],
            'green': [round(v, 2) for v in hist_g],
            'blue': [round(v, 2) for v in hist_b]
        },
        'dimensions': {
            'width': width,
            'height': height,
            'total_pixels': int(width * height)
        }
    }