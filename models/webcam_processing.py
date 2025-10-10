import cv2
import numpy as np
import base64

# Load Haarcascades for face detection
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError("Could not load the face cascade classifier.")

def detect_faces_realtime(frame):
    # convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles aroudn faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # add label
        cv2.putText(frame, f"Face", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return frame , len(faces)
 
def apply_filter_realtime(frame, filter_type="none", **params):
    if filter_type == "grayscale":
        frmae = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    elif filter_type == "sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        
        frame = cv2.transform(frame, kernel)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    elif filter_type == "invert":
        frame = cv2.bitwise_not(frame)
    
    elif filter_type == "blur":
        blur_amount = params.get("blur", 5)
        kernel_size = int(blur_amount) * 2 + 1
        frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    elif filter_type == "edge":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == "cartoon":
        # simple realtime cartoon effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

        # Bilateral filter for color
        color = cv2.bilateralFilter(frame, 9, 75, 75)

        # combining
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(color, edges_colored)

    elif filter_type == "sketch":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        inverted_blur = cv2.bitwise_not(blurred)
        sketch = cv2.divide(gray, inverted_blur, scale=256.0)
        frame = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == "cool":
        b, g, r = cv2.split(frame)
        b = np.clip(b.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        r = np.clip(r.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
        frame = cv2.merge([b, g, r])
    
    elif filter_type == "warm":
        b, g, r = cv2.split(frame)
        r = np.clip(r.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        g = np.clip(g.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
        frame = cv2.merge([b, g, r])
    
    return frame

def process_frame(frame_data, mode="detect", filter_type="none"):
    try:
        # Decode base64 image
        img_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"success": False, "error": "Could not decode frame"}
        
        # Process based on mode
        if mode == "detect":
            frame, face_count = detect_faces_realtime(frame)
            metadata = {"faces": face_count}
        elif mode == "filter":
            frame = apply_filter_realtime(frame, filter_type)
            metadata = {"filter": filter_type}
        elif mode == "both":
            # apply yhe filter first then detect the faces in the frame
            frame = apply_filter_realtime(frame, filter_type)
            frame, face_count = detect_faces_realtime(frame)
            metadata = {"faces": face_count, "filter": filter_type}
        else:
            metadata = {}

        # encode back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "frame": f"data:image/jpeg;base64,{frame_base64}",
            "metadata": metadata
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
    
def save_frame(frame_data, output_path):
    try:
        # decode base64 image
        img_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode (nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return False
        cv2.imwrite(output_path, frame)
        return True
    
    except Exception as e:
        print(f"Error saving frame: {e}")
        return False