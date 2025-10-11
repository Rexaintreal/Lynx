import cv2

def pixelate_image(input_path, output_path, block_size=10):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Image not found or invalid format")
    
    h, w = img.shape[:2]

    # Calculate scaled-down size based on block size
    temp = cv2.resize(img, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(output_path, pixelated)
    return {"block_size": block_size, "output_path": output_path}
