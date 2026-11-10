# AI-Powered Security Monitoring System

A Python-based security monitoring project using Flask, OpenCV, and face recognition. The application is intended for local demonstration and experimentation.

## Features

- Face recognition using the `face_recognition` and OpenCV libraries
- Flask-based web interface
- Audio alert support
- Detection event logging and snapshot storage
- Log summary generation

## Technology Stack

- Python
- Flask
- OpenCV
- face-recognition
- NumPy
- Pygame

## Project Structure

```text
AI_Security_Project/
├── app.py
├── face_detection.py
├── summary_generator.py
├── alert.mp3
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── authorized_faces/
│   └── .gitkeep
└── logs/
    └── .gitkeep
```

## Setup

Use a Python environment compatible with the installed `face-recognition` / `dlib` dependencies. These packages can require additional native build tools, especially on Windows.

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add authorized face images to the `authorized_faces/` folder as required by the application.
4. Run the application:

   ```bash
   python app.py
   ```

5. Open the local Flask address printed by the application in your browser.

## Important security and privacy notes

- This repository is a demonstration project, not a production-ready security system.
- The original application source has been kept unchanged as requested. It contains hardcoded login credentials and a placeholder Flask secret key. **Do not deploy this version publicly or use it to protect real systems.** Replace those values in the source and implement proper authentication and secret management before any real deployment.
- Face images, generated snapshots, and runtime logs are not included in this repository. Keep any biometric data private and obtain appropriate consent before collecting or processing it.
- Face recognition can produce false matches and missed detections. Do not rely on it as the sole basis for security decisions.

## Notes

The `.env.example` file is a configuration reference only. The unchanged original `app.py` does not currently load these environment variables.
