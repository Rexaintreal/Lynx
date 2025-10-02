import cv2

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError("Could not load the face cascade classifier.")

def detect_faces(image_path: str, output_path: str) -> int:
    # Read Image
    image = cv2.imread(image_path)
    
    # Check if the image was read successfully
    if image is None:
        print(f"Error: Could not read image at {image_path}. Skipping.")
        return 0 

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # Equalize histogram for better detection in varying lighting
    gray = cv2.equalizeHist(gray) 

    # Detect Faces
    # Tuned parameters for potentially better results
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,         
        minNeighbors=5,          
        minSize=(30, 30),       
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw Rectangles
    RECT_COLOR = (0, 255, 0)  
    RECT_THICKNESS = 2        

    for (x, y, w, h) in faces:
        # Draw on the original color image
        cv2.rectangle(image, (x, y), (x + w, y + h), RECT_COLOR, RECT_THICKNESS)  

    # Save Result
    try:
        cv2.imwrite(output_path, image)
    except cv2.error as e:
        print(f"Error: Could not save image to {output_path}. Details: {e}")
        return 0 # Indicate failure or no faces counted

    # Return Face Count
    print(f"Detected {len(faces)} face(s) and saved result to {output_path}")
    return len(faces)