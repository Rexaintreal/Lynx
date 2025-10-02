import cv2
import os

BASE_DIR = os.path.dirname(__file__)   # models/
PRETRAINED_DIR = os.path.join(BASE_DIR, "pretrained")

# Face detection model
FACE_PROTO = os.path.join(PRETRAINED_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(PRETRAINED_DIR, "opencv_face_detector_uint8.pb")

# Age model
AGE_PROTO = os.path.join(PRETRAINED_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(PRETRAINED_DIR, "age_net.caffemodel")

# Gender model
GENDER_PROTO = os.path.join(PRETRAINED_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(PRETRAINED_DIR, "gender_net.caffemodel")

# Load networks
faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
ageNet = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
genderNet = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

# Categories
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


def recognize_faces(image_path: str, output_path: str):
    # Load image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Prepare input blob for face detection
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False, crop=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    people = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # filter weak detections
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            # Extract face
            face = image[max(0, y1-15):min(y2+15, h-1),
                         max(0, x1-15):min(x2+15, w-1)]

            # Blob for age & gender
            face_blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Predict gender
            genderNet.setInput(face_blob)
            gender_preds = genderNet.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]

            # Predict age
            ageNet.setInput(face_blob)
            age_preds = ageNet.forward()
            age = AGE_BUCKETS[age_preds[0].argmax()]

            # Draw rectangle + label
            label = f"{gender}, {age}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            people.append({"age": age, "gender": gender})

    # Save output
    cv2.imwrite(output_path, image)

    return people
