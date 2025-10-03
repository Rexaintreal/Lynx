import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from models.face_detection import detect_faces
from models.face_recognition import recognize_faces
from models.filters import apply_filter
from models.object_detection import detect_objects
import base64
import time

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/face-detection", methods=["GET", "POST"])
def face_detection():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Processed output filename
            output_filename = "processed_" + filename
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

            # Run detection
            num_faces = detect_faces(filepath, output_path)

            return render_template(
                "face.html",
                filename=output_filename,
                faces=num_faces
            )
    return render_template("face.html")


@app.route("/face-recognition", methods=["GET", "POST"])
def face_recognition():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Processed output filename
            output_filename = "recog_" + filename
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

            # Run recognition 
            people = recognize_faces(filepath, output_path)

            return render_template(
                "facerecog.html",
                filename=output_filename,
                people=people
            )
    return render_template("facerecog.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/filters", methods=["GET"])
def filters():
    return render_template("filters.html", filename=None)


@app.route("/filters/apply", methods=["POST"])
def apply_filter_route():
    """Apply OpenCV filter to uploaded image"""
    try:
        data = request.get_json()
        
        # Get the base64 image data
        img_data = data["image"].split(",")[1]
        filter_type = data.get("filter", "none")
        
        # Get slider values for adjustable filter
        brightness = data.get("brightness", 100)
        contrast = data.get("contrast", 100)
        sepia = data.get("sepia", 0)
        blur = data.get("blur", 0)
        
        # Generate unique filename with timestamp
        timestamp = str(int(time.time() * 1000))
        filename = f"filtered_{filter_type}_{timestamp}.jpg"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        # Decode and save input image
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(img_data))
        
        # Apply filter via OpenCV
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
        
        # If preset filter selected, use that
        if filter_type != "adjustable":
            apply_filter(filepath, output_path, filter_type)
        else:
            # Use adjustable filter with slider values
            apply_filter(filepath, output_path, "adjustable",
                        brightness=int(brightness),
                        contrast=int(contrast),
                        sepia=int(sepia),
                        blur=int(blur))
        
        # Return URL to processed image
        return jsonify({
            "success": True,
            "url": url_for("uploaded_file", filename=output_filename)
        })
        
    except Exception as e:
        print(f"Error applying filter: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/object-detection", methods=["GET", "POST"])
def object_detection():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template(
                "object_detection.html",
                error="No file part in the request."
            )

        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "object_detection.html",
                error="No file selected. Please upload an image."
            )

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Processed output filename
            output_filename = "objects_" + filename
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

            try:
                # Run object detection
                results = detect_objects(filepath, output_path)

                return render_template(
                    "object_detection.html",
                    filename=output_filename,
                    total_objects=results['total_objects'],
                    object_counts=results['object_counts'],
                    detections=results['detections']
                )

            except FileNotFoundError as e:
                # Missing YOLO files
                return render_template(
                    "object_detection.html",
                    error=str(e)
                )
            except Exception as e:
                print(f"Error during object detection: {e}")
                import traceback
                traceback.print_exc()
                return render_template(
                    "object_detection.html",
                    error="An error occurred during object detection. Please try again."
                )
        else:
            return render_template(
                "object_detection.html",
                error="Invalid file type. Please upload PNG, JPG, or JPEG."
            )

    return render_template("object_detection.html")



if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)