import cv2
import numpy as np

def pixelate_image(input_path, output_path, block_size=10, mode="normal"):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Invalid image path or format.")
    
    h, w = img.shape[:2]

    # Basic pixelation
    temp = cv2.resize(img, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    # Pixel art effect
    if mode == "pixel_art":
        Z = pixelated.reshape((-1, 3))
        Z = np.float32(Z)
        K = 8 # retro
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        pixelated = res.reshape(pixelated.shape)

    # Game stripes
    elif mode == "game_stripes":
        stripe_height = max(1, block_size // 3)
        for y in range(0, h, stripe_height * 2):
            pixelated[y:y + stripe_height] = pixelated[y:y + stripe_height] * 0.6

    cv2.imwrite(output_path, pixelated)
    return {"block_size": block_size, "mode": mode, "output_path": output_path}