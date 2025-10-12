import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import easyocr

# Set up multiple OCR readers for different languages
try:
    # Main reader: just English (safest, works with everything)
    reader_primary = easyocr.Reader(['en'], gpu=False)
    
    # Language-specific readers (each with English pair)
    reader_hindi = easyocr.Reader(['hi', 'en'], gpu=False)
    reader_russian = easyocr.Reader(['ru', 'en'], gpu=False)
    reader_chinese = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    reader_arabic = easyocr.Reader(['ar', 'en'], gpu=False)
    reader_japanese = easyocr.Reader(['ja', 'en'], gpu=False)
    reader_korean = easyocr.Reader(['ko', 'en'], gpu=False)
    
    READERS_AVAILABLE = True
    print("EasyOCR initialized successfully with 7 language readers!")
except Exception as e:
    print(f"❌ Warning: Couldn't load EasyOCR - {e}")
    READERS_AVAILABLE = False


def analyze_location(input_path, output_path):
    
    # Load the image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Could not read image")
    
    original_shape = img.shape
    output_img = img.copy()
    
    # Set up our results dictionary
    results = {
        'dimensions': f"{original_shape[1]}x{original_shape[0]}",
        'exif_data': {},
        'architecture': {},
        'vegetation': {},
        'text_analysis': {},
        'color_palette': {},
        'climate_indicators': {},
        'objects_detected': {},
        'road_features': {},
        'probable_regions': []
    }
    
    # Run all our analysis functions
    results['exif_data'] = _extract_gps_data(input_path)  # Check for GPS in metadata
    results['architecture'] = _analyze_architecture(img)  # Look at buildings
    results['vegetation'] = _analyze_vegetation(img)  # Check plants/greenery
    results['text_analysis'] = _analyze_text(img)  # Read any text/signs
    results['color_palette'] = _analyze_color_palette(img)  # Analyze colors
    results['climate_indicators'] = _analyze_climate(img)  # Check sky/weather
    results['objects_detected'] = _detect_location_objects(img)  # Find objects
    results['road_features'] = _analyze_road_features(img)  # Look at roads
    results['probable_regions'] = _predict_regions(results)  # Make our guess!
    
    # Create a cool annotated version of the image
    output_img = _create_annotated_image(img, results)
    cv2.imwrite(output_path, output_img)
    
    return results


def _extract_gps_data(image_path):
    try:
        image = Image.open(image_path)
        exif = image._getexif()
        
        if not exif:
            return {'has_gps': False, 'message': 'No EXIF data found'}
        
        gps_data = {}
        for tag, value in exif.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == 'GPSInfo':
                for gps_tag in value:
                    gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                    gps_data[gps_tag_name] = value[gps_tag]
        
        if gps_data:
            return {
                'has_gps': True,
                'data': gps_data,
                'message': 'GPS coordinates found in EXIF'
            }
        else:
            return {'has_gps': False, 'message': 'No GPS data in EXIF'}
    
    except Exception as e:
        return {'has_gps': False, 'message': f'Error reading EXIF: {str(e)}'}


def _analyze_architecture(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find edges (buildings have lots of straight lines)
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect lines - buildings are full of them!
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    vertical_lines = 0
    horizontal_lines = 0
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Horizontal lines
            if angle < 10 or angle > 170:
                horizontal_lines += 1
            # Vertical lines
            elif 80 < angle < 100:
                vertical_lines += 1
    
    # More lines = more buildings
    total_lines = vertical_lines + horizontal_lines
    
    if total_lines > 200:
        style = "Dense Urban"
        confidence = 75
    elif total_lines > 100:
        style = "Suburban/Mixed"
        confidence = 65
    elif total_lines > 50:
        style = "Rural/Low-density"
        confidence = 60
    else:
        style = "Natural/Minimal structures"
        confidence = 55
    
    return {
        'style': style,
        'confidence': confidence,
        'vertical_lines': int(vertical_lines),
        'horizontal_lines': int(horizontal_lines),
        'structure_density': 'High' if total_lines > 150 else 'Medium' if total_lines > 50 else 'Low'
    }


def _analyze_vegetation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define what counts as "green" (vegetation)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Count green pixels
    green_pixels = np.sum(green_mask > 0)
    total_pixels = img.shape[0] * img.shape[1]
    vegetation_percentage = (green_pixels / total_pixels) * 100
    
    # Check how "lush" the green is (bright green vs dull green)
    if vegetation_percentage > 0:
        green_regions = cv2.bitwise_and(hsv, hsv, mask=green_mask)
        avg_saturation = np.mean(green_regions[green_mask > 0][:, 1]) if green_pixels > 0 else 0
    else:
        avg_saturation = 0
    
    # Guess the climate based on vegetation
    if vegetation_percentage > 40 and avg_saturation > 100:
        climate = "Tropical/Subtropical"
    elif vegetation_percentage > 25:
        climate = "Temperate"
    elif vegetation_percentage > 10:
        climate = "Mediterranean/Semi-arid"
    else:
        climate = "Arid/Desert or Urban"
    
    return {
        'coverage_percentage': round(vegetation_percentage, 2),
        'climate_zone': climate,
        'vegetation_density': 'Dense' if vegetation_percentage > 30 else 'Moderate' if vegetation_percentage > 15 else 'Sparse',
        'lushness': 'High' if avg_saturation > 100 else 'Medium' if avg_saturation > 50 else 'Low'
    }


def _analyze_text(img):
    try:
        if not READERS_AVAILABLE:
            return {
                'text_detected': False,
                'script': 'OCR unavailable',
                'region_hint': 'EasyOCR not initialized',
                'sample_text': 'OCR unavailable',
                'detected_languages': []
            }
        
        # Try reading text with all our OCR readers
        all_results = []
        
        # Start with the main reader (English, Hindi, Russian)
        try:
            results_primary = reader_primary.readtext(img)
            all_results.extend(results_primary)
        except:
            pass
        
        # If we didn't find much text, try the other language readers
        if len(all_results) < 3:
            try:
                results_chinese = reader_chinese.readtext(img)
                all_results.extend(results_chinese)
            except:
                pass
            
            try:
                results_arabic = reader_arabic.readtext(img)
                all_results.extend(results_arabic)
            except:
                pass
            
            try:
                results_japanese = reader_japanese.readtext(img)
                all_results.extend(results_japanese)
            except:
                pass
            
            try:
                results_korean = reader_korean.readtext(img)
                all_results.extend(results_korean)
            except:
                pass
        
        if not all_results:
            return {
                'text_detected': False,
                'script': 'None',
                'region_hint': 'No text detected',
                'sample_text': 'No text detected',
                'detected_languages': []
            }
        
        # Combine all the text we found
        full_text = ' '.join([result[1] for result in all_results])
        
        detected_scripts = set()
        detected_languages = []
        
        # Check Unicode ranges to identify scripts
        has_devanagari = any(0x0900 <= ord(char) <= 0x097F for char in full_text)  # Hindi
        has_bengali = any(0x0980 <= ord(char) <= 0x09FF for char in full_text)
        has_tamil = any(0x0B80 <= ord(char) <= 0x0BFF for char in full_text)
        has_telugu = any(0x0C00 <= ord(char) <= 0x0C7F for char in full_text)
        has_gujarati = any(0x0A80 <= ord(char) <= 0x0AFF for char in full_text)
        has_kannada = any(0x0C80 <= ord(char) <= 0x0CFF for char in full_text)
        has_malayalam = any(0x0D00 <= ord(char) <= 0x0D7F for char in full_text)
        
        # Other major scripts
        has_cyrillic = any(0x0400 <= ord(char) <= 0x04FF for char in full_text)  # Russian
        has_arabic = any(0x0600 <= ord(char) <= 0x06FF for char in full_text)
        has_chinese = any(0x4E00 <= ord(char) <= 0x9FFF for char in full_text)
        has_japanese_hiragana = any(0x3040 <= ord(char) <= 0x309F for char in full_text)
        has_japanese_katakana = any(0x30A0 <= ord(char) <= 0x30FF for char in full_text)
        has_korean = any(0xAC00 <= ord(char) <= 0xD7AF for char in full_text)
        has_thai = any(0x0E00 <= ord(char) <= 0x0E7F for char in full_text)
        
        script = "Latin"
        region_hint = "Western countries/Global"
        
        # Check Indian scripts first
        if has_devanagari:
            detected_scripts.add("Devanagari (Hindi/Marathi/Nepali)")
            detected_languages.append("Hindi")
            script = "Devanagari"
            region_hint = "India (North/Central), Nepal"
        
        if has_bengali:
            detected_scripts.add("Bengali/Assamese")
            detected_languages.append("Bengali")
            script = "Bengali"
            region_hint = "India (East), Bangladesh"
        
        if has_tamil:
            detected_scripts.add("Tamil")
            detected_languages.append("Tamil")
            script = "Tamil"
            region_hint = "India (Tamil Nadu), Sri Lanka"
        
        if has_telugu:
            detected_scripts.add("Telugu")
            detected_languages.append("Telugu")
            script = "Telugu"
            region_hint = "India (Andhra Pradesh, Telangana)"
        
        if has_gujarati:
            detected_scripts.add("Gujarati")
            detected_languages.append("Gujarati")
            script = "Gujarati"
            region_hint = "India (Gujarat)"
        
        if has_kannada:
            detected_scripts.add("Kannada")
            detected_languages.append("Kannada")
            script = "Kannada"
            region_hint = "India (Karnataka)"
        
        if has_malayalam:
            detected_scripts.add("Malayalam")
            detected_languages.append("Malayalam")
            script = "Malayalam"
            region_hint = "India (Kerala)"
        
        # Check other major scripts
        if has_cyrillic:
            detected_scripts.add("Cyrillic")
            detected_languages.append("Russian")
            script = "Cyrillic"
            region_hint = "Russia, Eastern Europe, Central Asia"
        
        if has_arabic:
            detected_scripts.add("Arabic")
            detected_languages.append("Arabic")
            script = "Arabic"
            region_hint = "Middle East, North Africa"
        
        if has_chinese:
            detected_scripts.add("Chinese")
            detected_languages.append("Chinese")
            script = "Chinese"
            region_hint = "China, Taiwan, Singapore"
        
        if has_japanese_hiragana or has_japanese_katakana:
            detected_scripts.add("Japanese")
            detected_languages.append("Japanese")
            script = "Japanese"
            region_hint = "Japan"
        
        if has_korean:
            detected_scripts.add("Korean")
            detected_languages.append("Korean")
            script = "Korean"
            region_hint = "South Korea, North Korea"
        
        if has_thai:
            detected_scripts.add("Thai")
            detected_languages.append("Thai")
            script = "Thai"
            region_hint = "Thailand"
        
        indian_scripts = [s for s in detected_scripts if any(x in s for x in ['Devanagari', 'Bengali', 'Tamil', 'Telugu', 'Gujarati', 'Kannada', 'Malayalam'])]
        if len(indian_scripts) > 1:
            region_hint = "India (Multiple regions)"
        
        return {
            'text_detected': True,
            'script': script,
            'scripts_detected': list(detected_scripts) if detected_scripts else ['Latin'],
            'region_hint': region_hint,
            'sample_text': full_text[:200] if full_text else "Text detected but not readable",
            'detected_languages': detected_languages if detected_languages else ['English'],
            'confidence': round(np.mean([result[2] for result in all_results]) * 100, 1) if all_results else 0
        }
    
    except Exception as e:
        return {
            'text_detected': False,
            'script': 'Unknown',
            'region_hint': 'Unable to analyze',
            'sample_text': 'Error during text detection',
            'error': str(e),
            'detected_languages': []
        }


def _analyze_color_palette(img):
    # Make it smaller for faster processing
    small = cv2.resize(img, (100, 100))
    pixels = small.reshape(-1, 3).astype(np.float32)
    
    # Use k-means to find the 5 most common colors
    k = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = centers.astype(np.uint8)
    
    # Check if the image is warm or cool toned
    avg_color = np.mean(img, axis=(0, 1))
    b, g, r = avg_color
    
    if r > b + 20:
        temperature = "Warm (Desert/Mediterranean)"
    elif b > r + 20:
        temperature = "Cool (Northern regions)"
    else:
        temperature = "Neutral (Temperate)"
    
    # Check how vibrant/saturated the colors are
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_saturation = np.mean(hsv[:, :, 1])
    
    vibrancy = "High" if avg_saturation > 100 else "Medium" if avg_saturation > 50 else "Low"
    
    return {
        'dominant_colors': centers.tolist(),
        'temperature': temperature,
        'vibrancy': vibrancy,
        'avg_saturation': round(float(avg_saturation), 2)
    }


def _analyze_climate(img):
    # Just check the top third of the image (usually sky)
    top_third = img[:img.shape[0]//3, :]
    
    hsv = cv2.cvtColor(top_third, cv2.COLOR_BGR2HSV)
    
    # Look for blue sky
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_percentage = (np.sum(blue_mask > 0) / blue_mask.size) * 100
    
    # Check how bright/dark it is
    gray_top = cv2.cvtColor(top_third, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_top)
    
    # Make a guess about the weather
    if blue_percentage > 30:
        sky_condition = "Clear/Sunny"
        weather_hint = "Good weather, possibly tropical or Mediterranean"
    elif avg_brightness > 180:
        sky_condition = "Overcast/Cloudy"
        weather_hint = "Temperate or Northern climate"
    elif avg_brightness < 100:
        sky_condition = "Dark/Stormy"
        weather_hint = "Bad weather or nighttime"
    else:
        sky_condition = "Partly Cloudy"
        weather_hint = "Mixed weather conditions"
    
    return {
        'sky_condition': sky_condition,
        'blue_sky_percentage': round(blue_percentage, 2),
        'weather_hint': weather_hint,
        'avg_brightness': round(float(avg_brightness), 2)
    }


def _detect_location_objects(img):
    #  pretty basic - could use YOLO for better results
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find edges and contours
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count large objects (cars, buildings, etc.)
    large_objects = sum(1 for cnt in contours if cv2.contourArea(cnt) > 1000)
    
    object_density = "High" if large_objects > 50 else "Medium" if large_objects > 20 else "Low"
    
    return {
        'object_count': large_objects,
        'density': object_density,
        'hint': f"{large_objects} large objects detected - " + 
                ("urban area" if large_objects > 50 else "suburban/rural" if large_objects > 20 else "natural/remote")
    }


def _analyze_road_features(img):
    """Check road markings - yellow lines (Americas) vs white lines (Europe/Asia)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Look for white lines
    _, white_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_pixels = np.sum(white_thresh == 255)
    
    # Look for yellow lines (common in USA/Americas)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pixels = np.sum(yellow_mask > 0)
    
    has_roads = white_pixels > 1000 or yellow_pixels > 500
    
    # Figure out what kind of markings we found
    if yellow_pixels > white_pixels and yellow_pixels > 1000:
        marking_type = "Yellow markings (Americas)"
    elif white_pixels > 5000:
        marking_type = "White markings (Europe/Asia)"
    else:
        marking_type = "No clear road markings"
    
    return {
        'roads_detected': has_roads,
        'marking_type': marking_type,
        'white_pixels': int(white_pixels),
        'yellow_pixels': int(yellow_pixels)
    }


def _predict_regions(results):
    predictions = []
    
    # Start with a score for each region (higher = more likely)
    scores = {
        'Northern Europe': 0,
        'Southern Europe': 0,
        'North America': 0,
        'South America': 0,
        'East Asia': 0,
        'Southeast Asia': 0,
        'Middle East': 0,
        'Africa': 0,
        'Oceania': 0,
        'India': 0,
        'Russia/Central Asia': 0
    }
    
    # Text/script is the BIGGEST clue
    text = results['text_analysis']
    
    if text.get('detected_languages'):
        for lang in text['detected_languages']:
            if lang in ['Hindi', 'Bengali', 'Tamil', 'Telugu', 'Gujarati', 'Kannada', 'Malayalam']:
                scores['India'] += 50
                scores['Southeast Asia'] += 5
    
    # Other scripts
    if text['script'] == 'Cyrillic':
        scores['Russia/Central Asia'] += 45
        scores['Northern Europe'] += 10
    elif text['script'] == 'Arabic':
        scores['Middle East'] += 45
        scores['Africa'] += 20
    elif text['script'] == 'Chinese':
        scores['East Asia'] += 45
    elif text['script'] == 'Japanese':
        scores['East Asia'] += 50
    elif text['script'] == 'Korean':
        scores['East Asia'] += 50
    elif text['script'] == 'Thai':
        scores['Southeast Asia'] += 45
    
    # Weather clues (boost if no text detected)
    climate = results['climate_indicators']
    climate_weight = 15 if not text.get('text_detected', False) else 10
    
    if 'Clear' in climate['sky_condition']:
        scores['Southern Europe'] += climate_weight
        scores['Middle East'] += climate_weight
        scores['Oceania'] += climate_weight
        scores['India'] += int(climate_weight * 0.8)
    elif 'Overcast' in climate['sky_condition']:
        scores['Northern Europe'] += climate_weight + 5
        scores['North America'] += climate_weight
    
    # Vegetation clues (boost if no text detected)
    veg = results['vegetation']
    veg_weight = 30 if not text.get('text_detected', False) else 20
    
    if 'Tropical' in veg['climate_zone']:
        scores['Southeast Asia'] += veg_weight
        scores['South America'] += veg_weight - 5
        scores['Africa'] += veg_weight - 10
        scores['India'] += veg_weight - 5
    elif 'Arid' in veg['climate_zone']:
        scores['Middle East'] += veg_weight - 5
        scores['Africa'] += veg_weight - 10
        scores['Oceania'] += veg_weight - 15
        scores['India'] += veg_weight - 15
    elif 'Temperate' in veg['climate_zone']:
        scores['Northern Europe'] += veg_weight - 10
        scores['North America'] += veg_weight - 15
        scores['East Asia'] += veg_weight - 15
    
    # Road markings are a good clue
    road = results['road_features']
    if 'Yellow' in road['marking_type']:
        scores['North America'] += 22
        scores['South America'] += 18
        scores['East Asia'] += 8
    elif 'White' in road['marking_type']:
        scores['Northern Europe'] += 15
        scores['Southern Europe'] += 12
        scores['East Asia'] += 10
        scores['India'] += 12
        scores['Southeast Asia'] += 8
    
    # Architecture density
    arch = results['architecture']
    if arch['structure_density'] == 'High':
        scores['East Asia'] += 12
        scores['North America'] += 10
        scores['Northern Europe'] += 10
        scores['India'] += 10
    elif arch['structure_density'] == 'Low':
        scores['Oceania'] += 8
        scores['Africa'] += 8
    
    # Color temperature
    color = results['color_palette']
    if 'Warm' in color['temperature']:
        scores['Middle East'] += 10
        scores['India'] += 10
        scores['Africa'] += 8
        scores['Southern Europe'] += 8
    elif 'Cool' in color['temperature']:
        scores['Northern Europe'] += 12
        scores['Russia/Central Asia'] += 10
    
    # Sort regions by score (highest first)
    sorted_regions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Take the top 5 regions that have at least some points
    top_regions = [r for r in sorted_regions if r[1] > 0][:5]
    total_score = sum(score for _, score in top_regions)
    
    # Convert scores to percentages
    if total_score > 0:
        for region, score in top_regions:
            percentage = int((score / total_score) * 100)
            # Only show regions with at least 5% confidence
            if percentage >= 5:
                predictions.append({
                    'region': region,
                    'confidence': percentage,
                    'score': score
                })
    
    # If we have no good guesses, say so
    if not predictions:
        predictions.append({
            'region': 'Insufficient data for prediction',
            'confidence': 0,
            'score': 0
        })
    
    return predictions


def _create_annotated_image(img, results):
    output = img.copy()
    h, w = output.shape[:2]
    
    # Create a dark overlay at the top for text
    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    
    # Write our findings on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = 35
    
    cv2.putText(output, "Location Analysis", (20, y_pos), font, 0.7, (0, 255, 0), 2)
    y_pos += 30
    
    # Show the top region guess
    if results['probable_regions']:
        top_region = results['probable_regions'][0]
        text = f"Top Match: {top_region['region']} ({top_region['confidence']}%)"
        cv2.putText(output, text, (20, y_pos), font, 0.5, (255, 255, 255), 1)
        y_pos += 25
    
    # Show climate
    climate = results['vegetation']['climate_zone']
    cv2.putText(output, f"Climate: {climate}", (20, y_pos), font, 0.5, (255, 255, 255), 1)
    y_pos += 25
    
    # Show architecture style
    arch = results['architecture']['style']
    cv2.putText(output, f"Architecture: {arch}", (20, y_pos), font, 0.5, (255, 255, 255), 1)
    
    return output