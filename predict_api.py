from flask import Flask, render_template, Response, request, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
import base64
import json
import uuid
import argparse
import os
import sys
import boto3
from flask_cors import CORS
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.checks import cv2, print_args
from utils.general import update_options
import logging

# Initialize paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Configure logging (example)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')

# **Create Argument Parsers**
yolo_parser = argparse.ArgumentParser(description="YOLO Model Arguments")
flask_parser = argparse.ArgumentParser(description="Flask App Arguments")

# **Add YOLO Arguments**
yolo_parser.add_argument('--model','--weights', type=str, default=ROOT / 'yolov8n.pt', help='model path or triton URL')
yolo_parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='source directory for images or videos')
yolo_parser.add_argument('--conf','--conf-thres', type=float, default=0.5, help='object confidence threshold for detection')
yolo_parser.add_argument('--iou', '--iou-thres', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
yolo_parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='image size as scalar or (h, w) list, i.e. (640, 480)')
yolo_parser.add_argument('--half', action='store_true', help='use half precision (FP16)')
yolo_parser.add_argument('--device', default='', help='device to run on, i.e. cuda device=0/1/2/3 or device=cpu')
yolo_parser.add_argument('--show','--view-img', default=False, action='store_true', help='show results if possible')
yolo_parser.add_argument('--save', action='store_true', help='save images with results')
yolo_parser.add_argument('--save_txt','--save-txt', action='store_true', help='save results as .txt file')
yolo_parser.add_argument('--save_conf', '--save-conf', action='store_true', help='save results with confidence scores')
yolo_parser.add_argument('--save_crop', '--save-crop', action='store_true', help='save cropped images with results')
yolo_parser.add_argument('--show_labels','--show-labels', default=True, action='store_true', help='show labels')
yolo_parser.add_argument('--show_conf', '--show-conf', default=True, action='store_true', help='show confidence scores')
yolo_parser.add_argument('--max_det','--max-det', type=int, default=300, help='maximum number of detections per image')
yolo_parser.add_argument('--vid_stride', '--vid-stride', type=int, default=1, help='video frame-rate stride')
yolo_parser.add_argument('--stream_buffer', '--stream-buffer', default=False, action='store_true', help='buffer all streaming frames (True) or return the most recent frame (False)')
yolo_parser.add_argument('--line_width', '--line-thickness', default=None, type=int, help='The line width of the bounding boxes. If None, it is scaled to the image size.')
yolo_parser.add_argument('--visualize', default=False, action='store_true', help='visualize model features')
yolo_parser.add_argument('--augment', default=False, action='store_true', help='apply image augmentation to prediction sources')
yolo_parser.add_argument('--agnostic_nms', '--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
yolo_parser.add_argument('--retina_masks', '--retina-masks', default=False, action='store_true', help='whether to plot masks in native resolution')
yolo_parser.add_argument('--classes', type=list, help='filter results by class, i.e. classes=0, or classes=[0,2,3]') # 'filter by class: --classes 0, or --classes 0 2 3')
yolo_parser.add_argument('--boxes', default=True, action='store_false', help='Show boxes in segmentation predictions')
yolo_parser.add_argument('--exist_ok', '--exist-ok', action='store_true', help='existing project/name ok, do not increment')
yolo_parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
yolo_parser.add_argument('--name', default='exp', help='save results to project/name')
yolo_parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

# **Add Flask Arguments**
flask_parser.add_argument('--raw_data', '--raw-data', default=ROOT / 'data/raw', help='save raw images to data/raw')
flask_parser.add_argument('--port', default=5000, type=int, help='port deployment')

# **Parse Arguments**
yolo_opt, yolo_unknown = yolo_parser.parse_known_args()
flask_opt, flask_unknown = flask_parser.parse_known_args()

# Load model (Ensemble is not supported)
MODEL_PATH = "./KINOKO/yoloresult/okashi23/weights/best.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    logging.error(f"Failed to load YOLO model: {e}")
    # Handle the error (e.g., exit the application)
    sys.exit(1)

# Initialize Flask API and set default configurations
#app = Flask(__name__)
# Serve static files from the Vue.js 'dist' folder
app = Flask(__name__, static_folder='./dist/', static_url_path='')
CORS(app)
# Set a default RAW_DATA path to prevent KeyError
app.config['RAW_DATA'] = Path(flask_opt.raw_data)
app.config['RAW_DATA'].mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# DATABASE SETUP WITH FLASK_SQLALCHEMY AND AWS RDS CONFIG
# --------------------------------------------------------------------

# Set the SQLALCHEMY_DATABASE_URI.
#DB connection
#DB_URI = 'postgresql://postgres:chimera1112@database-2.c7ceu88uuynd.us-west-1.rds.amazonaws.com:5432/kinokodb'
#app.config["SQLALCHEMY_DATABASE_URI"] = DB_URI
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")

#from ssm_parameter_store import EC2ParameterStore
#store = EC2ParameterStore(prefix='/poc')
# access a parameter under /poc/yolov8DB
#DB_URI = store['yolov8db']
#print(DB_URI)









ssm = boto3.client('ssm', region_name='us-west-1')



## It's working !!!
## Getting de DB_URI as a "string" from SSM parameter store

parameter = ssm.get_parameter(Name='/poc/yolov8db', WithDecryption=False)

DB_URI = parameter['Parameter']['Value']
print(DB_URI)



## It's working and I think it's better
## Getting a stringList from SSM parameter store

dictParameters = ssm.get_parameters(Names=['/poc/predict'], WithDecryption=False)
#print(dictParameters)

p = str( {p['Value'] for p in dictParameters['Parameters']} )[1:-1].replace("'", "").split(",")
#print(type(p))
#print(p)

DB_URI = 'postgresql://' + p[0] + ':' + p[1] + '@database-2.' + p[2] + '.us-west-1.rds.amazonaws.com:5432/'+ p[3]
print(DB_URI)




# Initialize SQLAlchemy instance
app.config["SQLALCHEMY_DATABASE_URI"] = DB_URI
db = SQLAlchemy(app)

# Create a model to store the count
class MushroomCount(db.Model):
    __tablename__ = "mushroom_count"
    id = db.Column(db.Integer, primary_key=True)
    kinoko_count = db.Column(db.Integer, default=0)
    takenoko_count = db.Column(db.Integer, default=0)

# Create/update the database tables if they do not exist
with app.app_context():
    db.create_all()

# Helper function to update counts in the database
def update_mushroom_count(kinoko_found, takenoko_found):
    count = MushroomCount.query.first()
    if not count:
        count = MushroomCount(kinoko_count=5, takenoko_count=5)
        db.session.add(count)
        db.session.commit()
    count.kinoko_count += kinoko_found
    count.takenoko_count += takenoko_found
    db.session.commit()
    logging.info(
        f"Updated counts: kinoko +{kinoko_found}, takenoko +{takenoko_found}"
    )

# --------------------------------------------------------------------
# END DATABASE SETUP
# --------------------------------------------------------------------

# Global variable for model
# model = None  # Remove this line, model is initialized above
STATIC_FOLDER = Path("data/final")
STATIC_FOLDER.mkdir(exist_ok=True)
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Set a default RAW_DATA path to prevent KeyError
RAW_DATA_FOLDER = Path("data/raw")
RAW_DATA_FOLDER.mkdir(exist_ok=True)
app.config['RAW_DATA'] = RAW_DATA_FOLDER

# Your model loading code here
MODEL_PATH = "./KINOKO/yoloresult/okashi23/weights/best.pt"

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    logging.error(f"Failed to load YOLO model: {e}")
    # Handle the error (e.g., exit the application)
    sys.exit(1)

confidence_threshold = 0.5 #Define the confidence threshold.

# NEW Route to fetch current counts from the database
@app.route("/counts", methods=["GET"])
def get_counts():
    count = MushroomCount.query.first()
    if not count:
        return json.dumps({"kinoko": 0, "takenoko": 0})
    return json.dumps({
        "kinoko": count.kinoko_count,
        "takenoko": count.takenoko_count
    })

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'myfile' not in request.files:
        return jsonify({"error": "No file part"}), 400

    uploaded_file = request.files['myfile']

    if uploaded_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if uploaded_file:
        # Save the raw image
        raw_image_path = app.config['RAW_DATA'] / uploaded_file.filename
        uploaded_file.save(raw_image_path)

        # Run prediction
        results = model(str(raw_image_path))

        # Extract class counts
        kinoko_count = 0
        takenoko_count = 0
        for r in results: #Iterate through all predictions in case there are several.
            for box in r.boxes:
                confidence = float(box.conf[0])  # Extract confidence as a float
                class_id = int(box.cls[0])      # Extract class ID as an integer

                if confidence > confidence_threshold:
                    if r.names[class_id] == 'kinoko':
                        kinoko_count += 1
                    elif r.names[class_id] == 'takenoko':
                        takenoko_count += 1
        # Process the image and get the path
        image_path = process_image(str(raw_image_path), results, model.names)
        if not image_path:
            return jsonify({"error": "Error processing image"}), 500

        # Create the response
        response_data = {
            "results": {
                "class": {
                    "kinoko": kinoko_count,
                    "takenoko": takenoko_count
                }
            },
            "img": request.url_root + image_path # Use absolute URL
        }
        print(response_data)
        # Update the counts in the database
        update_mushroom_count(kinoko_count, takenoko_count)
        count_updated = True
        return jsonify(response_data)

    return jsonify({"error": "No file uploaded or invalid request"}), 400

def process_image(image_path, results, class_names):
    try:
        # Let YOLO draw the annotations using its default settings.
        # If there are multiple results, you can choose one (here we use the last).
        img = None
        for r in results:
            img = r.plot()
        if img is None:  # Fallback in case no results were processed
            img = cv2.imread(image_path)

        # Save the processed image with a unique name.
        output_image_name = f"{uuid.uuid4()}.jpg"
        output_image_path = app.config["STATIC_FOLDER"] / output_image_name
        cv2.imwrite(str(output_image_path), img)

        return f"data/final/{output_image_name}"  # Return path relative to root.
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    
# def process_image(image_path, results, class_names):
#     try:
#         img = cv2.imread(image_path)
#         # Iterate over each result (each predicted box)
#         for r in results: #Iterate through all predictions in case there are several.
#             for box in r.boxes:
#                 b = box.xyxy[0].cpu().numpy().astype(int)  # Get box coordinates
#                 c = int(box.cls)  # Get class ID

#                 label = f'{class_names[c]} {box.conf[0]:.2f}'
#                 cv2.rectangle(img, b[:2], b[2:], (0, 255, 0), 2)  # Draw rectangle
#                 cv2.putText(img, label, (b[0], b[1] - 10), 0.9, (0, 255, 0), 2)  # Draw label

#         # Save the processed image with a unique name
#         output_image_name = f"{uuid.uuid4()}.jpg"
#         output_image_path = app.config['STATIC_FOLDER'] / output_image_name
#         cv2.imwrite(str(output_image_path), img)
#         # for result in results:
#         # if opt.save_txt:
#         #     result_json = json.loads(result.tojson())
#         #     yield json.dumps({'results': result_json, 'source': str(opt.source)})
#         # else:
#         #     im0 = cv2.imencode('.jpg', result.plot())[1].tobytes()

#         return f'data/final/{output_image_name}'  # Path relative to root
#     except Exception as e:
#         print(f"Error processing image: {e}")
#         return None


# @app.route('/predict', methods=['GET', 'POST'])
# def predict_endpoint():
#     global yolo_opt

#     # Process request parameters and file upload
#     if request.method == 'POST':
#         uploaded_file = request.files.get('myfile')
#         save_txt = request.form.get('save_txt', 'F')  # Default to 'F' if not provided

#         if uploaded_file:
#             # Save the uploaded file and update options accordingly.
#             source = Path(__file__).parent / app.config['RAW_DATA'] / uploaded_file.filename
#             uploaded_file.save(source)
#             yolo_opt.source = source
#         else:
#             # If no file was provided update the option from query parameters.
#             yolo_opt.source, _ = update_options(request)

#         yolo_opt.save_txt = True if save_txt == 'T' else False

#     elif request.method == 'GET':
#         yolo_opt.source, yolo_opt.save_txt = update_options(request)

#     # Run prediction. We assume the model returns an iterator/list of results.
#     results = list(model(**vars(yolo_opt), stream=True))
#     if not results:
#         return jsonify({"error": "No prediction was made."}), 400

#     # Take the first result (adjust if needed)
#     result = results[0]

#     # Depending on the flag "save_txt", we either return classification details only
#     # or we also include a base64 encoded image produced by drawing on the image.
#     # if yolo_opt.save_txt:
#     #     result_json = json.loads(result.tojson())
#     #     image_base64 = None
#     # else:
#         # Plot the image with annotations
#     # img = result.plot()
#     # success, encoded_image = cv2.imencode('.jpg', img)
#     # if not success:
#     #     return jsonify({"error": "Failed to encode image."}), 500

#     #     # Convert the image bytes to a base64 string.
#     # image_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
#     # # Try to extract additional details if available.
#     # result_json = json.loads(result.tojson()) if hasattr(result, "tojson") else {}

#     # # Build the JSON response matching what your Vue frontend expects.
#     # response_data = {
#     #     "results": result_json,
#     #     "image": image_base64,
#     #     "source": str(yolo_opt.source)
#     # }
#     # print(response_data)

#     # return jsonify(response_data)
#     img = result.plot()
#     success, encoded_image = cv2.imencode('.jpg', img)

#     if not success:
#         return jsonify({"error": "Failed to encode image."}), 500

#     # Generate a unique filename
#     image_filename = f"{uuid.uuid4()}.jpg"
#     image_path = app.config['STATIC_FOLDER'] / image_filename

#     # Save the image to the static folder
#     cv2.imwrite(str(image_path), img)

#     # Create a URL to the image
#     image_url = request.url_root + 'static/' + image_filename # request.url_root = The root URL of the application. This is useful if you’re generating content that may live outside the application itself

#     # Create your JSON response
#     response_data = {
#         "results": {
#             "class": {
#                 "kinoko": kinoko_count,
#                 "takenoko": takenoko_count
#             }
#         },
#         "img": image_url
#     }

#     return jsonify(response_data)


# def predict(opt):
#     results = model(**vars(opt), stream=True)

#     for result in results:
#         if opt.save_txt:
#             result_json = json.loads(result.tojson())
#             print(result_json)
#             yield json.dumps({'results': result_json, 'source': str(opt.source)})
#         else:
#             im0 = cv2.imencode('.jpg', result.plot())[1].tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + im0 + b'\r\n')
#             print(final_image)
#     return final_image

# @app.route('/predict', methods=['GET', 'POST'])
# def predict_endpoint():
#     response_data = {
#             "results": {
#               "class": "detected_class"
#             },
#             "source": str(yolo_opt.source)
#         }

#     return jsonify(response_data)

# @app.route('/predict', methods=['GET', 'POST'])
# def predict_endpoint():
#     global yolo_opt

#     # Process request parameters and file upload
#     if request.method == 'POST':
#         uploaded_file = request.files.get('myfile')
#         save_txt = request.form.get('save_txt', 'F')  # Default to 'F' if not provided

#         if uploaded_file:
#             # Save the uploaded file and update options accordingly.
#             source = Path(__file__).parent / app.config['RAW_DATA'] / uploaded_file.filename
#             uploaded_file.save(source)
#             yolo_opt.source = source
#         else:
#             # If no file was provided update the option from query parameters.
#             yolo_opt.source, _ = update_options(request)

#         yolo_opt.save_txt = True if save_txt == 'T' else False

#     elif request.method == 'GET':
#         yolo_opt.source, yolo_opt.save_txt = update_options(request)

#     # Run prediction. We assume the model returns an iterator/list of results.
#     results = list(model(**vars(yolo_opt), stream=True))
#     if not results:
#         return jsonify({"error": "No prediction was made."}), 400

#     # Take the first result (adjust if needed)
#     result = results[0]

#     # Depending on the flag "save_txt", we either return classification details only
#     # or we also include a base64 encoded image produced by drawing on the image.
#     # if yolo_opt.save_txt:
#     #     result_json = json.loads(result.tojson())
#     #     image_base64 = None
#     # else:
#         # Plot the image with annotations
#     img = result.plot()
#     success, encoded_image = cv2.imencode('.jpg', img)
#     if not success:
#         return jsonify({"error": "Failed to encode image."}), 500

    #     # Convert the image bytes to a base64 string.
    # image_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    # # Try to extract additional details if available.
    # result_json = json.loads(result.tojson()) if hasattr(result, "tojson") else {}

#     response_data = {
#             "results": {
#               "class": "detected_class"
#             },
#             "source": str(yolo_opt.source)
#         }
    # response_data = {
    #     "results": result_json,
    #     "image": image_base64,
    #     "source": str(yolo_opt.source)
    # }
    # response_data = {
    #         "results": {
    #           "class": "kinoko"
    #         },
    #         "source": encoded_image
    #     }
    # print("aqui")
    # print(response_data)
    # return jsonify(response_data)

#@app.route('/')
#def index():
#    return render_template('index.html')
# Serve static files from the Vue.js 'dist' folder
# Catch-all route to serve Vue frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')
    
@app.route('/data/final/<path:filename>')
def serve_final_image(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

# @app.route('/predict', methods=['GET', 'POST'])
# def video_feed():
#     global yolo_opt
#     if request.method == 'POST':
#         uploaded_file = request.files.get('myfile')
#         save_txt = request.form.get('save_txt', 'F')  # Default to 'F' if save_txt is not provided

#         if uploaded_file:
#             source = Path(__file__).parent / app.config['RAW_DATA'] / uploaded_file.filename
#             uploaded_file.save(source)
#             yolo_opt.source = source  # Use yolo_opt
#         else:
#             yolo_opt.source, _ = update_options(request) # Use yolo_opt

#         yolo_opt.save_txt = True if save_txt == 'T' else False # Use yolo_opt

#     elif request.method == 'GET':
#         yolo_opt.source, yolo_opt.save_txt = update_options(request) # Use yolo_opt
#     print(predict(yolo_opt))
#     return Response(predict(yolo_opt))


if __name__ == '__main__':

    # print used arguments
    print("YOLO Options:", vars(yolo_opt))
    print("Flask Options:", vars(flask_opt))

    # Get port to deploy
    port = flask_opt.port

    # Update RAW_DATA in app.config with the command line argument
    app.config['RAW_DATA'] = Path(flask_opt.raw_data)
    app.config['RAW_DATA'].mkdir(parents=True, exist_ok=True)

    # Run app
    app.run(host='0.0.0.0', port=port, debug=False) # Don't use debug=True, model will be loaded twice