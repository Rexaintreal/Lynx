import cv2
import numpy as np

def generate_ascii_art(input_path, output_path, method="standard", **kwargs):
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Could not read image")
    
    original_shape = img.shape

    details = {
        'method': method,
        'dimensions': f"{original_shape[1]}x{original_shape[0]}",
        'processing_info': {},
        'ascii_text': ''
    }

    if method == "standard":
        ascii_text, output_img, info = _standard_ascii(img, **kwargs)
        details['ascii_text'] = ascii_text
        details['processing_info'] = info

    elif method == "detailed":
        ascii_text, output_img, info = _detailed_ascii(img, **kwargs)
        details['ascii_text'] = ascii_text
        details['processing_info'] = info

    elif method == "block":
        ascii_text, output_img, info = _block_ascii(img, **kwargs)
        details['ascii_text'] = ascii_text
        details['processing_info'] = info

    elif method == "edge":
        ascii_text, output_img, info = _edge_ascii(img, **kwargs)
        details['ascii_text'] = ascii_text
        details['processing_info'] = info
    
    elif method == "braille":
        ascii_text, output_img, info = _braille_ascii(img, **kwargs)
        details['ascii_text'] = ascii_text
        details['processing_info'] = info

    # Save output image
    cv2.imwrite(output_path, output_img)

    return details

def _standard_ascii(img, width=100, invert=False):
    # ASCII charcter from dark to light
    ascii_chars = " .:-=+*#%@"
    if invert:
        ascii_chars = ascii_chars[::-1]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image
    height, original_width = gray.shape
    aspect_ratio = height / original_width
    new_height = int(width * aspect_ratio * 0.55) 

    resized = cv2.resize(gray, (width, new_height))

    # Generate ASCII
    ascii_lines = []
    for row in resized:
        line = ""
        for pixel in row:
            # map pixel value to ascii character
            char_index = int(pixel / 255 * (len(ascii_chars) - 1))
            line += ascii_chars[char_index]
        ascii_lines.append(line)

    ascii_text = "\n".join(ascii_lines)

    # Create output image
    output_img = _text_to_image(ascii_text, font_scale=0.3)

    return ascii_text, output_img, {
        'algorithm': 'Standard ASCII',
        'character_Set': ascii_chars,
        'width': width,
        'height': new_height,
        'total_characters': len(ascii_text.replace('\n', ''))
    }

def _detailed_ascii(img, width=120, invert=False):
    # Extended ASCII characters
    ascii_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    if invert:
        ascii_chars = ascii_chars[::-1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, original_width = gray.shape
    aspect_ratio = height / original_width
    new_height = int(width * aspect_ratio * 0.55)

    resized = cv2.resize(gray, (width, new_height))

    # Apply slight sharpening for better detail
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    ascii_lines = []
    for row in sharpened:
        line = ""
        for pixel in row:
            char_index = int(pixel / 255 * (len(ascii_chars) - 1))
            line += ascii_chars[char_index]
        ascii_lines.append(line)

    ascii_text = "\n".join(ascii_lines)
    output_img = _text_to_image(ascii_text, font_scale=0.25)

    return ascii_text, output_img, {
        'algorithm': 'Detailed ASCII',
        'character_set_size': len(ascii_chars),
        'width': width,
        'height': new_height,
        'enhancement': 'Sharpened'
    }

def _block_ascii(img, width=80, invert=False):
    # Block characters
    ascii_chars = " ░▒▓█"
    if invert:
        ascii_chars = ascii_chars[::-1]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, original_width = gray.shape
    aspect_ratio = height / original_width
    new_height = int(width * aspect_ratio * 0.55)

    resized = cv2.resize(gray, (width, new_height))

    ascii_lines = []
    for row in resized:
        line = ""
        for pixel in row:
            char_index = int(pixel / 255 * (len(ascii_chars) - 1))
            line += ascii_chars[char_index]
        ascii_lines.append(line)
    
    ascii_text = "\n".join(ascii_lines)
    output_img = _text_to_image(ascii_text, font_scale=0.4)

    return ascii_text, output_img, {
        'algorithm': 'Block ASCII',
        'character_set': ascii_chars,
        'width': width,
        'height': new_height,
        'style': 'Unicode blocks'
    }

def _edge_ascii(img, width=100, threshold1=100, threshold2=200):
    ascii_chars = " .-=+*#@"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, original_width = gray.shape
    aspect_ratio = height / original_width
    new_height = int(width * aspect_ratio * 0.55)

    resized = cv2.resize(gray, (width, new_height))

    # Apply edge detection
    edges = cv2.Canny(resized, threshold1, threshold2)

    ascii_lines = []
    for row in edges:
        line = ""
        for pixel in row:
            if pixel > 0:
                line += ascii_chars[-1] 
            else:
                brightness_idx = int(resized[len(ascii_lines), len(line)] / 255 * (len(ascii_chars) - 2))
                line += ascii_chars[brightness_idx]
        ascii_lines.append(line)
    ascii_text = "\n".join(ascii_lines)
    output_img = _text_to_image(ascii_text, font_scale=0.3)
    
    return ascii_text, output_img, {
        'algorithm': 'Edge-based ASCII',
        'edge_detection': 'Canny',
        'thresholds': f"{threshold1}-{threshold2}",
        'width': width,
        'height': new_height
    }

def _braille_ascii(img, width=80, invert=False):
    # Braille patterns for different densities
    braille_chars = " ⠁⠃⠇⠏⠟⠿⣿"
    if invert:
        braille_chars = braille_chars[::-1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, orginal_width = gray.shape
    aspect_ratio = height / orginal_width
    new_height = int(width * aspect_ratio * 0.55)

    resized = cv2.resize(gray, (width, new_height))

    ascii_lines = []
    for row in resized:
        line = ""
        for pixel in row:
            char_index = int(pixel / 255 * (len(braille_chars) - 1))
            line += braille_chars[char_index]
        ascii_lines.append(line)

    ascii_text = "\n".join(ascii_lines)
    output_img = _text_to_image(ascii_text, font_scale=0.35)
    
    return ascii_text, output_img, {
        'algorithm': 'Braille Pattern',
        'character_set': 'Unicode Braille',
        'width': width,
        'height': new_height,
        'style': 'Braille dots'
    }

def _text_to_image(text, font_scale=0.3, font_thickness=1):

    lines = text.split('\n')

    # Calculate image size
    font = cv2.FONT_HERSHEY_PLAIN
    char_width = int(font_scale * 10)
    char_height = int(font_scale * 15)

    img_width = max(len(line) for line in lines) * char_width + 20
    img_height = len(lines) * char_height + 20

    # Create white background
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Draw text
    y = char_height
    for line in lines:
        cv2.putText(img, line, (10, y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        y += char_height
    
    return img