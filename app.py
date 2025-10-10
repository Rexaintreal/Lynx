import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from models.face_detection import detect_faces
from models.face_recognition import recognize_faces
from models.filters import apply_filter
from models.object_detection import detect_objects
from models.scene_understanding import analyze_scene
from models.color_analysis import analyze_colors
from models.color_manipulation import apply_color_manipulation
from models.special_effects import apply_special_effect
from models.document_scanner import scan_document
from models.text_detection import detect_text_regions, extract_text_basic
from models.text_extractor import extract_text_from_image
from models.webcam_processing import process_frame, save_frame
from models.qr_detection import detect_qr_from_image
from models.numberplate_detection import extract_numberplates
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

@app.route("/scene-understanding", methods=["GET", "POST"])
def scene_understanding():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template(
                "scene_understanding.html",
                error="No file part in the request."
            )

        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "scene_understanding.html",
                error="No file selected. Please upload an image."
            )

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Processed output filename
            output_filename = "scene_" + filename
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
            

            try:
                # Run scene analysis
                results = analyze_scene(filepath, output_path)

                return render_template(
                    "scene_understanding.html",
                    filename=output_filename,
                    scene_type=results['scene_type'],
                    scene_confidence=results['scene_confidence'],
                    total_objects=results['total_objects'],
                    object_counts=results['object_counts'],
                    detections=results['detections'],
                    spatial_distribution=results['spatial_distribution'],
                    scene_analysis=results['scene_analysis']
                )

            except FileNotFoundError as e:
                return render_template(
                    "scene_understanding.html",
                    error=str(e)
                )
            except Exception as e:
                print(f"Error during scene analysis: {e}")
                import traceback
                traceback.print_exc()
                return render_template(
                    "scene_understanding.html",
                    error="An error occurred during scene analysis. Please try again."
                )
        else:
            return render_template(
                "scene_understanding.html",
                error="Invalid file type. Please upload PNG, JPG, or JPEG."
            )

    return render_template("scene_understanding.html")


@app.route("/color-analysis", methods=["GET", "POST"])
def color_analysis():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template(
                "color_analysis.html",
                error="No file part in the request."
            )

        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "color_analysis.html",
                error="No file selected. Please upload an image."
            )

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Processed output filename
            output_filename = "color_" + filename
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

            try:
                # Run color analysis
                results = analyze_colors(filepath, output_path)

                return render_template(
                    "color_analysis.html",
                    filename=output_filename,
                    color_palette=results['color_palette'],
                    color_mood=results['color_mood'],
                    color_temperature=results['color_temperature'],
                    avg_hue=results['avg_hue'],
                    avg_saturation=results['avg_saturation'],
                    avg_value=results['avg_value'],
                    histograms=results['histograms'],
                    dimensions=results['dimensions']
                )

            except Exception as e:
                print(f"Error during color analysis: {e}")
                import traceback
                traceback.print_exc()
                return render_template(
                    "color_analysis.html",
                    error="An error occurred during color analysis. Please try again."
                )
        else:
            return render_template(
                "color_analysis.html",
                error="Invalid file type. Please upload PNG, JPG, or JPEG."
            )

    return render_template("color_analysis.html")

@app.route("/color-manipulation", methods=["GET", "POST"])
def color_manipulation():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template(
                "color_manipulation.html",
                error="No file part in the request."
            )
        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "color_manipulation.html",
                error="No file selected. Please upload on image."
            )
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # get operation type
            operation = request.form.get("operation", "saturation")

            # Processed output filename
            output_filename = f"manipulated_{operation}_{filename}"
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

            try:
                # Prepare kwargs based on operation type
                kwargs = {}

                if operation == "background":
                    # Parse hex color to RGB
                    bg_color_hex = request.form.get("bg_color", "#ffffff")
                    bg_color_rgb = tuple(int(bg_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    kwargs['target_color'] = bg_color_rgb
                    kwargs['threshold'] = int(request.form.get("bg_threshold", 50))
                elif operation == "replace":
                    # Parse source and target colors
                    source_hex = request.form.get("source_color", "#ff0000")
                    target_hex = request.form.get("target_color", "#00ff00")
                    source_rgb = tuple(int(source_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    target_rgb = tuple(int(target_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    kwargs['source_color'] = source_rgb
                    kwargs['target_color'] = target_rgb
                    kwargs['tolerance'] = int(request.form.get("tolerance", 30))
                
                elif operation == "saturation":
                    kwargs['saturation_scale'] = float(request.form.get("saturation", 1.5))
                
                elif operation == "hue":
                    kwargs['hue_shift'] = int(request.form.get("hue_shift", 30))
                
                # Run color manipulation
                details = apply_color_manipulation(filepath, output_path, operation, **kwargs)
                
                return render_template(
                    "color_manipulation.html",
                    filename=output_filename,
                    operation=operation,
                    details=details
                )

            except Exception as e:
                print(f"Error during color manipulation: {e}")
                import traceback
                traceback.print_exc()
                return render_template(
                    "color_manipulation.html",
                    error="An error occurred during color manipulation. Please try again."
                )
            
        else:
            return render_template(
                "color_manipulation.html",
                error="Invalid file type. Please uplaod PNG, JPG, or JPEG."
            )
            
    return render_template("color_manipulation.html")

@app.route("/special-effects", methods=["GET", "POST"])
def special_effects():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template(
                "special_effects.html",
                error="No file part in the request."
            )
        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "special_effects.html",
                error="No file selected. Please upload an image."
            )
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            effect = request.form.get("effect", "colorpop")

            output_filename = f"effect_{effect}_{filename}"
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

            try:
                kwargs = {}

                if effect == "greenscreen":
                    screen_hex = request.form.get("screen_color", "#00ff00")
                    new_bg_hex = request.form.get("new_bg_color", "#ffffff")
                    screen_rgb = tuple(int(screen_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    new_bg_rgb = tuple(int(new_bg_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    kwargs['bg_color'] = screen_rgb
                    kwargs['new_bg_color'] = new_bg_rgb
                    kwargs['threshold'] = int(request.form.get("gs_threshold", 40))
                
                elif effect == "colorpop":
                    pop_hex = request.form.get("pop_color", "#ff0000")
                    pop_rgb = tuple(int(pop_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    kwargs['keep_color'] = pop_rgb
                    kwargs['tolerance'] = int(request.form.get("pop_tolerance", 30))
                
                elif effect == "duotone":
                    shadow_hex = request.form.get("shadow_color", "#00008b")
                    highlight_hex = request.form.get("highlight_color", "#ffd700")
                    shadow_rgb = tuple(int(shadow_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    highlight_rgb = tuple(int(highlight_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    kwargs['shadow_color'] = shadow_rgb
                    kwargs['highlight_color'] = highlight_rgb
                
                elif effect == "isolation":
                    isolate_hex = request.form.get("isolate_color", "#ff0000")
                    isolate_rgb = tuple(int(isolate_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    kwargs['isolate_color'] = isolate_rgb
                    kwargs['tolerance'] = int(request.form.get("iso_tolerance", 30))
                    kwargs['desaturation'] = float(request.form.get("desaturation", 0.3))
                
                details = apply_special_effect(filepath, output_path, effect, **kwargs)
                
                return render_template(
                    "special_effects.html",
                    filename=output_filename,
                    effect=effect,
                    details=details
                )

            except Exception as e:
                print(f"Error during special effect: {e}")
                import traceback
                traceback.print_exc()
                return render_template(
                    "special_effects.html",
                    error="An error occurred during effect processing. Please try again."
                )
            
        else:
            return render_template(
                "special_effects.html",
                error="Invalid file type. Please upload PNG, JPG, or JPEG."
            )
            
    return render_template("special_effects.html")

@app.route("/document-scanner", methods=["GET", "POST"])
def document_scanner():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template(
                "document_scanner.html",
                error="No file part in the request."
            )
        
        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "document_scanner.html",
                error="No file selected. Please upload an image."
            )
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Get scan mode from form
            mode = request.form.get("mode", "enhanced")

            # Output filename
            output_filename = f"scanned_{mode}_{filename}"
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

            try:
                # Run document scanning
                details = scan_document(filepath, output_path, mode=mode)
                
                return render_template(
                    "document_scanner.html",
                    filename=output_filename,
                    details=details
                )

            except Exception as e:
                print(f"Error during document scanning: {e}")
                import traceback
                traceback.print_exc()
                return render_template(
                    "document_scanner.html",
                    error="An error occurred during document scanning. Please try again."
                )
        else:
            return render_template(
                "document_scanner.html",
                error="Invalid file type. Please upload PNG, JPG, or JPEG."
            )
    
    return render_template("document_scanner.html")

@app.route("/text-detection", methods=["GET", "POST"])
def text_detection():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template(
                "text_detection.html",
                error="No file part in the request."
            )
        
        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "text_detection.html",
                error="No file selected. Please upload an image."
            )
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Get detection method from form
            method = request.form.get("method", "mser")

            # Output filename
            output_filename = f"text_{method}_{filename}"
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

            try:
                # Run text detection
                if method == "extract":
                    details = extract_text_basic(filepath, output_path)
                else:
                    details = detect_text_regions(filepath, output_path, detection_method=method)
                
                return render_template(
                    "text_detection.html",
                    filename=output_filename,
                    details=details
                )

            except Exception as e:
                print(f"Error during text detection: {e}")
                import traceback
                traceback.print_exc()
                return render_template(
                    "text_detection.html",
                    error="An error occurred during text detection. Please try again."
                )
        else:
            return render_template(
                "text_detection.html",
                error="Invalid file type. Please upload PNG, JPG, or JPEG."
            )
    
    return render_template("text_detection.html")

@app.route("/text-extractor", methods=["GET", "POST"])
def text_extractor():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template(
                "text_extractor.html",
                error="No file part in the request."
            )
        
        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "text_extractor.html",
                error="No file selected. Please upload an image."
            )
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Get options from the from
            draw_boxes = request.form.get("draw_boxes", "true") == "true"
            confidence = float(request.form.get("confidence", 25)) / 100  # Convert to 0-1 range

            # Output filename
            output_filename = f"extracted_{filename}"
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

            try:
                # Run text extraction
                details = extract_text_from_image(
                    filepath,
                    output_path,
                    draw_boxes=draw_boxes,
                    confidence_threshold=confidence
                )

                return render_template(
                    "text_extractor.html",
                    filename=output_filename,
                    details=details
                )
            
            except Exception as e:
                print(f"Error during text extraction: {e}")
                import traceback
                traceback.print_exc()
                return render_template(
                    "text_extractor.html",
                    error="An error occurred during text extraction. Please try again."
                )
        else:
            return render_template(
                "text_extractor.html",
                error="Invalid file type. Please upload PNG, JPG, or JPEG."
            )
    
    return render_template("text_extractor.html")

@app.route("/webcam-live", methods=["GET"])
def webcam_live():
    return render_template("webcam_live.html")

@app.route("/webcam-live/process", methods=["POST"])
def webcam_process():
    try:
        data = request.get_json()

        frame_data = data.get("frame")
        mode = data.get("mode", "detect")
        filter_type = data.get("filter", "none")

        if not frame_data:
            return jsonify({
                "success": False,
                "error": "No frame data provided"
            }), 400
        
        # Process the frame
        result = process_frame(frame_data, mode=mode, filter_type=filter_type)

        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing webcam frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/webcam-live/capture", methods=["POST"])
def webcam_capture():
    try:
        data = request.get_json()
        frame_data = data.get("frame")

        if not frame_data:
            return jsonify({
                "success": False,
                "error": "No frame data provided"
            }), 400
        
        # Generate unique filename
        timestamp = str(int(time.time() * 1000))
        filename = f"webcam_capture_{timestamp}.jpg"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Save the image
        if save_frame(frame_data, filepath):
            return jsonify({
                "success": True,
                "filename": filename,
                "url": url_for("uploaded_file", filename=filename)
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to save frame"
            }), 500
    except Exception as e:
        print(f"Error capturing frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
@app.route("/qr_detection", methods=["GET", "POST"])
def qr_detection():
    qr_result = None
    if request.method == "POST":
        if "image" not in request.files:
            qr_result = {"success": False, "message": "No file uploaded."}
        else:
            file = request.files["image"]
            if file.filename == "":
                qr_result = {"success": False, "message": "No file selected."}
            else:
                filename = secure_filename(file.filename)
                upload_path = os.path.join("uploads", filename)
                file.save(upload_path)

                qr_result = detect_qr_from_image(upload_path, "uploads")
                
                if qr_result['success']:
                    qr_result['image_filename'] = os.path.basename(qr_result['image_path'])

    return render_template("qr_detection.html", result=qr_result)

@app.route("/numberplate_detection", methods=["GET", "POST"])
def numberplate_detection():
    if request.method == "POST":
        # Check for file in request
        if "file" not in request.files:
            return render_template(
                "numberplate_detection.html",
                error="No file part in the request."
            )
        
        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "numberplate_detection.html",
                error="No file selected. Please upload an image."
            )

        # Validate and save input file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(input_path)

            # Output path setup
            output_filename = f"plate_{filename}"
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

            try:
                result = extract_numberplates(input_path, output_path)

                return render_template(
                    "numberplate_detection.html",
                    filename=result["output_path"],  
                    detected_plates=result["detected_plates"]
                )


            except Exception as e:
                print(f"Error during number plate detection: {e}")
                import traceback
                traceback.print_exc()
                return render_template(
                    "numberplate_detection.html",
                    error="An error occurred while processing the image. Please try again."
                )
        else:
            return render_template(
                "numberplate_detection.html",
                error="Invalid file type. Please upload PNG, JPG, or JPEG."
            )

    return render_template("numberplate_detection.html", filename=None)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)