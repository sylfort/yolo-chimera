# Japanese Sweets Image Classifier - Backend API (Kinoko vs Takenoko)

[![GitHub Repo](https://img.shields.io/badge/Backend%20Repo-GitHub-blue?logo=github)](https://github.com/sylfort/yolo-chimera)
[![Frontend Repo](https://img.shields.io/badge/Frontend%20Repo-GitHub-lightgrey?logo=github)](https://github.com/sylfort/chimera-frontend)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](http://ec2-54-215-114-190.us-west-1.compute.amazonaws.com)

This repository contains the Python/Flask backend API for the Japanese Sweets Image Classifier project. It handles image uploads, performs machine learning inference using YOLOv8, interacts with a PostgreSQL database, and serves results to the [Vue.js frontend application](https://github.com/sylfort/chimera-frontend).

[View the Live Demo](http://ec2-54-215-114-190.us-west-1.compute.amazonaws.com) | [Visit the Frontend Repository](https://github.com/sylfort/chimera-frontend)

## Overview

This backend application forms the core logic of the YOLO-Chimera project. It exposes a RESTful API that allows a client application (like the companion [Vue.js frontend](https://github.com/sylfort/chimera-frontend)) to upload images of Japanese sweets ("Kinoko no Yama" and "Takenoko no Sato").

The backend then:
1.  Receives the uploaded image.
2.  Performs object detection using a fine-tuned YOLOv8 model to identify and count the sweets.
3.  Updates persistent counters for each sweet type in an AWS RDS PostgreSQL database using SQLAlchemy.
4.  Annotates the original image with bounding boxes around detected sweets.
5.  Returns the detection counts and a URL to the processed image back to the client.
6.  Provides an endpoint to query the total accumulated counts from the database for the leaderboard display.

The entire backend stack is designed for deployment on AWS, leveraging EC2, RDS, and SSM, managed via Terraform and configured to run efficiently, potentially within free-tier limits.

## Features (Backend Perspective)

*   **RESTful API Endpoints:** Provides `/predict` for image processing and `/counts` for retrieving leaderboard data.
*   **YOLOv8 Inference:** Integrates a custom-trained Ultralytics YOLOv8 model for accurate sweet detection.
*   **Database Persistence:** Uses Flask-SQLAlchemy to interact with a PostgreSQL database (AWS RDS) for storing and retrieving leaderboard counts.
*   **Image Processing:** Annotates images with detection results using OpenCV.
*   **Secure Configuration:** Utilizes AWS SSM Parameter Store (via Boto3) to securely load database credentials.
*   **Static File Serving:** Configured (when deployed with Nginx) to serve the static frontend build files (`./dist` from the frontend repo) and the generated annotated images (`/data/final`).

## Tech Stack (Backend)

*   **Machine Learning:**
    *   [Ultralytics YOLOv8](https://ultralytics.com/) (`yolov8n` fine-tuned on custom dataset)
    *   PyTorch (via Ultralytics)
    *   OpenCV (`opencv-python` via Ultralytics)
*   **Backend Framework & Libraries:**
    *   [Python](https://www.python.org/) 3.8+
    *   [Flask](https://flask.palletsprojects.com/): Micro web framework.
    *   [Flask-SQLAlchemy](https://flask-sqlalchemy.palletsprojects.com/): ORM for database interaction.
    *   [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html): AWS SDK for Python.
    *   [Flask-CORS](https://flask-cors.readthedocs.io/): Handles Cross-Origin Resource Sharing.
*   **Database:**
    *   [PostgreSQL](https://www.postgresql.org/) (hosted on AWS RDS)
*   **Deployment & Infrastructure:**
    *   [AWS (Amazon Web Services)](https://aws.amazon.com/): EC2, RDS, S3.
    *   [Nginx](https://nginx.org/): Web server and reverse proxy.
    *   [uWSGI](https://uwsgi-docs.readthedocs.io/): Application server gateway interface for Flask.
    *   [Terraform](https://www.terraform.io/): Infrastructure as Code.
    *   Linux Ubuntu 24.04 LTS
    *   Bash Scripting

## Architecture & API Interaction

1.  **Frontend Request:** The [Vue.js frontend](https://github.com/sylfort/chimera-frontend) sends HTTP requests to this backend API deployed on AWS EC2 (behind Nginx).
2.  **Nginx Routing:** Nginx receives the request.
    *   If it's an API call (e.g., `/predict`, `/counts`), Nginx reverse-proxies it to the uWSGI server.
    *   If it's a request for a static file (root `/`, `/index.html`, JS, CSS, frontend images), Nginx serves it directly from the frontend's build directory (`./dist`).
    *   If it's a request for a processed image (`/data/final/...`), Nginx serves the static image file.
3.  **uWSGI & Flask:** uWSGI receives the API request from Nginx and forwards it to the Flask application (`predict_api.py`).
4.  **`/predict` Endpoint (POST):**
    *   Flask receives the image file.
    *   Saves the raw image temporarily to `/data/raw`.
    *   Calls the YOLOv8 model for inference.
    *   Counts detections above the confidence threshold.
    *   Connects to RDS (using URI from SSM via Boto3) and updates counts in the `mushroom_count` table via SQLAlchemy.
    *   Annotates the image using OpenCV/YOLO's `plot()` method.
    *   Saves the annotated image to the `/data/final` static directory.
    *   Returns a JSON response `{ "results": {"class": {"kinoko": X, "takenoko": Y}}, "img": "<URL_to_annotated_image>" }`.
5.  **`/counts` Endpoint (GET):**
    *   Flask receives the request.
    *   Queries the RDS database via SQLAlchemy for the latest totals in `mushroom_count`.
    *   Returns a JSON response `{ "kinoko": TotalX, "takenoko": TotalY }`.
6.  **Database:** AWS RDS PostgreSQL instance stores the `mushroom_count` table.
7.  **Configuration:** AWS SSM Parameter Store securely holds the database connection string, retrieved by Flask using Boto3 at startup.

## API Endpoints

*   `POST /predict`
    *   **Request:** `multipart/form-data` with image file attached under the key `myfile`.
    *   **Response:** `application/json`
        ```json
        {
          "results": {
            "class": {
              "kinoko": <int: count in image>,
              "takenoko": <int: count in image>
            }
          },
          "img": "<string: absolute URL to the annotated image>"
        }
        ```
    *   **Side Effects:** Updates counts in the database, saves raw and annotated images locally on the server (`/data/raw`, `/data/final`).
*   `GET /counts`
    *   **Request:** None
    *   **Response:** `application/json`
        ```json
        {
          "kinoko": <int: total count from DB>,
          "takenoko": <int: total count from DB>
        }
        ```
*   `GET /data/final/<filename>`
    *   Serves the static annotated image file generated by a `/predict` call from the `./data/final` directory.
*   `GET /` and `GET /<path:path>`
    *   Serves the `index.html` and other static assets for the Vue.js frontend (typically from the `./dist` directory when deployed).

## Setup and Local Development

These instructions are for running the **backend API server** locally.

1.  **Prerequisites:**
    *   Python 3.8+ and Pip
    *   PostgreSQL Server (running locally or on cloud)
    *   AWS Account and configured AWS CLI *or* method to provide DB URI locally (e.g., environment variables).
    *   Git

2.  **Clone this Repository:**
    ```bash
    git clone https://github.com/sylfort/yolo-chimera.git
    cd yolo-chimera
    ```

3.  **Backend Setup:**
    *   Create and activate a virtual environment:
        ```bash
        python -m venv venv
        # On Windows: venv\Scripts\activate
        # On macOS/Linux: source venv/bin/activate
        ```
    *   Install Python dependencies:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Database Setup:**
    *   Ensure your PostgreSQL server is running.
    *   Create a database (e.g., `kinokodb`).
    *   Configure the database connection URI. The application expects this via AWS SSM Parameter `/poc/yolov8db`. For local development:
        *   **Option A (AWS SSM):** Ensure your local environment has AWS credentials configured (`aws configure`) that can access the parameter in `us-west-1`.
        *   **Option B (Environment Variable - Recommended for Local):**
            1.  Modify `predict_api.py` to read the DB URI from an environment variable instead of SSM. Comment out the `boto3` and SSM parts and add something like:
                ```python
                # import os
                # DB_URI = os.environ.get("DATABASE_URL", "postgresql://user:pass@host:port/dbname") # Provide a default or ensure it's set
                # app.config["SQLALCHEMY_DATABASE_URI"] = DB_URI
                ```
            2.  Set the `DATABASE_URL` environment variable before running the app:
                ```bash
                # Example (Linux/macOS)
                export DATABASE_URL="postgresql://your_user:your_password@localhost:5432/kinokodb"
                # Example (Windows CMD)
                set DATABASE_URL="postgresql://your_user:your_password@localhost:5432/kinokodb"
                # Example (Windows PowerShell)
                $env:DATABASE_URL="postgresql://your_user:your_password@localhost:5432/kinokodb"
                ```
    *   The application will attempt to create the `mushroom_count` table on first run if it doesn't exist.

5.  **Run the Backend API Server:**
    ```bash
    # Ensure virtual environment is active and DB_URI is configured
    python predict_api.py --port 5000
    ```
    The backend API should now be running on `http://localhost:5000`.

6.  **Running the Frontend (Separately):**
    *   To interact with the UI, you need to clone, set up, and run the [Frontend Application](https://github.com/sylfort/chimera-frontend) in a separate terminal.
    *   Make sure the frontend is configured to point to your locally running backend API URL (`http://localhost:5000` by default, potentially via its `.env.development.local` file).

## Usage

*   **Via Frontend:** The primary way to use this backend is through the [companion Vue.js frontend](https://github.com/sylfort/chimera-frontend). Follow the setup instructions in that repository to run the UI, which will then make requests to this backend API.
*   **Direct API Testing:** You can test the API endpoints directly using tools like `curl`, Postman, or Insomnia.
    *   Example `POST /predict`: Send a POST request to `http://localhost:5000/predict` with `multipart/form-data`, including a file field named `myfile` containing your image.
    *   Example `GET /counts`: Send a simple GET request to `http://localhost:5000/counts`.

## Model Training

*   Based on Ultralytics YOLOv8n (`yolov8n.pt`).
*   Fine-tuned on a custom, manually annotated dataset of Kinoko no Yama and Takenoko no Sato images.
*   Annotation format: YOLO bounding box (`.txt` files located in `KINOKO/labels/`).
*   Training details and configuration: See `KINOKO/yoloresult/okashi23/args.yaml`.
*   The resulting trained weights are located at `KINOKO/yoloresult/okashi23/weights/best.pt`.
*   Goal: Improve detection precision for these specific objects compared to the base model.

## Deployment

*   **Infrastructure:** AWS EC2 (compute), AWS RDS (database), AWS SSM (secrets).
*   **Orchestration:** Terraform manages the AWS resources.
*   **Web Server:** Nginx serves static frontend files (from `./dist` copied to the server) and acts as a reverse proxy.
*   **Application Server:** uWSGI runs the Flask application (`predict_api.py`).
*   **Process:**
    1.  Terraform provisions EC2, RDS, security groups, etc.
    2.  Backend code (this repo) is deployed to EC2.
    3.  Frontend code ([frontend repo](https://github.com/sylfort/chimera-frontend)) is built (`npm run build`) and the resulting `dist` directory is copied to the EC2 instance (e.g., to `/home/ubuntu/yolo-chimera/dist`).
    4.  Nginx is configured to serve static files from the `dist` directory and proxy API requests (`/predict`, `/counts`) to uWSGI (listening on a socket like `/wsgi_apptest/flask_with_nginx.sock`).
    5.  uWSGI is configured (e.g., via `flask_with_nginx_UWSGI.ini`) to run the Flask app defined in `predict_api.py` (likely using a WSGI entry point file).
    6.  The Flask app connects to RDS using credentials fetched from SSM.

## Future Improvements (Backend)

*   Implement more robust error handling for API requests and database operations.
*   Add API authentication/authorization if needed for future features.
*   Optimize model inference speed (e.g., ONNX Runtime, TensorRT if GPU is available).
*   Consider asynchronous task processing (e.g., Celery) for predictions if they become slow under load.
*   Refactor database interaction into a separate service layer for better organization.
*   Add unit and integration tests for API endpoints and core logic.
*   **Explore Segmentation/Feature Extraction:** As suggested, investigate adding image segmentation (using YOLOv8-seg) for precise outlines or feature extraction (shape, color, texture from bounding boxes/masks) for deeper analysis.

---
