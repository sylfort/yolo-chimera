from flask import Flask, render_template, Response, request
import json
import argparse
import os
import sys
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
yolo_parser.add_argument('--conf','--conf-thres', type=float, default=0.25, help='object confidence threshold for detection')
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
MODEL_PATH = "./yolov8n.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    logging.error(f"Failed to load YOLO model: {e}")
    # Handle the error (e.g., exit the application)
    sys.exit(1)

# Initialize Flask API and set default configurations
app = Flask(__name__)
# Set a default RAW_DATA path to prevent KeyError
app.config['RAW_DATA'] = Path(flask_opt.raw_data)
app.config['RAW_DATA'].mkdir(parents=True, exist_ok=True)

# Global variable for model
# model = None  # Remove this line, model is initialized above

def predict(opt):
    results = model(**vars(opt), stream=True)

    for result in results:
        if opt.save_txt:
            result_json = json.loads(result.tojson())
            yield json.dumps({'results': result_json, 'source': str(opt.source)})
        else:
            im0 = cv2.imencode('.jpg', result.plot())[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + im0 + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def video_feed():
    global yolo_opt
    if request.method == 'POST':
        uploaded_file = request.files.get('myfile')
        save_txt = request.form.get('save_txt', 'F')  # Default to 'F' if save_txt is not provided

        if uploaded_file:
            source = Path(__file__).parent / app.config['RAW_DATA'] / uploaded_file.filename
            uploaded_file.save(source)
            yolo_opt.source = source  # Use yolo_opt
        else:
            yolo_opt.source, _ = update_options(request) # Use yolo_opt

        yolo_opt.save_txt = True if save_txt == 'T' else False # Use yolo_opt

    elif request.method == 'GET':
        yolo_opt.source, yolo_opt.save_txt = update_options(request) # Use yolo_opt

    return Response(predict(yolo_opt), mimetype='multipart/x-mixed-replace; boundary=frame')


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
    app.run(host='1.0.0.0', port=port, debug=False) # Don't use debug=True, model will be loaded twice

