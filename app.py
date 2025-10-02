import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from models.face_detection import detect_faces
from models.face_recognition import recognize_faces

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

            # Run recognition (returns list of dicts with Age & Gender)
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


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
