# Japanese Sweets Image Classifier - Backend API (Kinoko vs Takenoko)

[![GitHub Repo](https://img.shields.io/badge/Backend%20Repo-GitHub-lightgrey?logo=github)](https://github.com/sylfort/yolo-chimera)
[![Frontend Repo](https://img.shields.io/badge/Frontend%20Repo-GitHub-blue?logo=github)](https://github.com/sylfort/chimera-frontend)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](http://ec2-54-215-114-190.us-west-1.compute.amazonaws.com)

This repository contains the Python/Flask backend API for the Japanese Sweets Image Classifier project. It handles image uploads, performs machine learning inference using YOLOv8, interacts with a PostgreSQL database, and serves results to the [Vue.js frontend application](https://github.com/sylfort/chimera-frontend).

[View the Live Demo](http://ec2-54-215-114-190.us-west-1.compute.amazonaws.com) | [Visit the Frontend Repository](https://github.com/sylfort/chimera-frontend)

## Overview

<p align="center">  <!-- Optional: align="center" or align="left" -->
  <img src="https://github.com/user-attachments/assets/2c001062-8bcd-41eb-b193-b557c92ce604" alt="Kansai Ben Quest image" width="650">
</p>

This backend application forms the core logic of the Image Classifier project. It exposes a RESTful API that allows a client application (like the companion [Vue.js frontend](https://github.com/sylfort/chimera-frontend)) to upload images of Japanese sweets ("Kinoko no Yama" and "Takenoko no Sato").

The backend then:
1.  Receives the uploaded image.
2.  Performs object detection using a fine-tuned YOLOv8 model to identify and count the sweets.
3.  Updates persistent counters for each sweet type in an AWS RDS PostgreSQL database using SQLAlchemy.
4.  Annotates the original image with bounding boxes around detected sweets.
5.  Returns the detection counts and a URL to the processed image back to the client.
6.  Provides an endpoint to query the total accumulated counts from the database for the leaderboard display.

The entire backend stack is designed for deployment on AWS, leveraging EC2, RDS, and SSM, managed via Terraform and configured to run efficiently within free-tier limits.

## Features (Backend Perspective)

*   **RESTful API Endpoints:** Provides `/predict` for image processing and `/counts` for retrieving leaderboard data.
*   **YOLOv8 Inference:** Integrates a custom-trained Ultralytics YOLOv8 model for accurate sweet detection.
*   **Database Persistence:** Uses Flask-SQLAlchemy to interact with a PostgreSQL database (AWS RDS) for storing and retrieving leaderboard counts.
*   **Image Processing:** Annotates images with detection results using OpenCV.
*   **Secure Configuration:** Utilizes AWS SSM Parameter Store to securely load database credentials.
*   **Static File Serving:** Nginx configured to serve the static frontend build files (`./dist` from the frontend repo) and the generated annotated images.

## Tech Stack (Backend)

*   **Machine Learning:**
    *   [Ultralytics YOLOv8](https://ultralytics.com/) (`yolov8n` fine-tuned on custom dataset)
    *   PyTorch (via Ultralytics)
    *   OpenCV (`opencv-python` via Ultralytics)
*   **Backend Framework & Libraries:**
    *   [Python](https://www.python.org/)
    *   [Flask](https://flask.palletsprojects.com/): Micro web framework.
    *   [Flask-SQLAlchemy](https://flask-sqlalchemy.palletsprojects.com/): ORM for database interaction.
    *   [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html): AWS SDK for Python.
    *   [Flask-CORS](https://flask-cors.readthedocs.io/): Handles Cross-Origin Resource Sharing.
*   **Database:**
    *   [PostgreSQL](https://www.postgresql.org/) (hosted on AWS RDS)
*   **Deployment & Infrastructure:**
    *   [AWS (Amazon Web Services)](https://aws.amazon.com/): EC2, RDS, S3.
    *   [Nginx](https://nginx.org/): Web server and reverse proxy.
    *   [uWSGI](https://uwsgi-docs.readthedocs.io/): Application server gateway interface for Python.
    *   [Terraform](https://www.terraform.io/): Infrastructure as Code.
    *   Linux Ubuntu 24.04 LTS
    *   Bash Scripting

## Deployment

*   **Infrastructure:** AWS EC2 (compute), AWS RDS (database), AWS SSM (secrets).
*   **Orchestration:** Terraform manages the AWS resources.
*   **Web Server:** Nginx serves static frontend files (from `./dist` copied to the server) and acts as a reverse proxy.
*   **Application Server:** uWSGI runs workers with the Flask application.
*   **Process:**
    1.  Terraform provisions EC2, RDS, security groups, etc.
    2.  Backend code (this repo) is deployed to EC2.
    3.  Frontend code ([frontend repo](https://github.com/sylfort/chimera-frontend)) is built (`npm run build`) and the resulting `dist` directory is deployed to the EC2 instance via GitHub actions.
    4.  Nginx is configured to serve static files from the `dist` directory and proxy API requests to uWSGI.
    5.  uWSGI is configured to run multiple workers of the Flask app.
    6.  The Flask app connects to RDS using credentials fetched from SSM.

## Future Improvements (Backend)

*   Implement more robust error handling for API requests and database operations.
*   Add API authentication/authorization if needed for future features.
*   Consider asynchronous task processing (e.g., Celery) for predictions if they become slow under load.
*   Refactor database interaction into a separate service layer for better organization.
*   Add unit and integration tests for API endpoints and core logic.
*   **Explore Segmentation/Feature Extraction:** Add image segmentation (using YOLOv8-seg) for precise outlines or feature extraction (shape, color, texture from bounding boxes/masks) for deeper analysis.

---
