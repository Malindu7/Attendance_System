# Attendance_System
# 🎓 Face Verification Attendance System

An automated, real-time attendance system built for university use. The system uses a live webcam feed to detect and verify student faces using deep learning, then records verified attendance directly into a MongoDB database — no manual sign-ins required.

> Developed for **CS31092 – Digital Image Processing** | General Sir John Kotelawala Defence University (KDU) | Faculty of Computing | Group 04

---

## 📸 Demo Overview

| Step | What Happens |
|---|---|
| 1. Camera opens | Live webcam feed streams to browser |
| 2. Face detected | HOG algorithm locates face in frame |
| 3. Face encoded | ResNet-34 CNN generates a 128-D embedding vector |
| 4. Face matched | Euclidean distance compared to registered students |
| 5. Attendance logged | Verified name + subject + timestamp saved to MongoDB |

---

## ✨ Features

- 🎥 **Real-time face detection** — HOG-based face localization on live webcam stream
- 🧠 **Deep learning verification** — ResNet-34 CNN (128-dimensional face embeddings)
- ✅ **Automatic attendance marking** — no manual input needed
- 🔁 **Duplicate prevention** — 60-second deduplication window prevents multiple records
- 🌐 **Web-based interface** — accessible from any browser via Flask
- 🗄️ **MongoDB database** — flexible NoSQL storage for attendance logs
- 👤 **Admin registration workflow** — capture student face photos directly through the app
- 📋 **Dashboard** — view and filter attendance records by subject
- 💻 **Windows compatible** — DirectShow camera fix included

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.x | Core backend language |
| Flask | Web framework and routing |
| OpenCV (cv2) | Camera capture, resizing, frame drawing |
| face_recognition | Face detection and 128-D encoding (wraps dlib's ResNet-34) |
| dlib | Underlying C++ ML engine for face landmark detection |
| NumPy | Vectorized distance calculations |
| MongoDB | NoSQL database for attendance records |
| PyMongo | Python–MongoDB connector |

---

## 🏗️ System Architecture

```
Webcam Frame
     │
     ▼
Resize to 25%  ──────────────────────── (speeds up detection)
     │
     ▼
BGR → RGB Conversion ─────────────────── (OpenCV → face_recognition format)
     │
     ▼
HOG Face Detection ───────────────────── (locate face bounding boxes)
     │
     ▼
ResNet-34 Face Encoding ──────────────── (generate 128-D embedding vector)
     │
     ▼
Euclidean Distance Matching ──────────── (compare to stored encodings)
     │
     ▼
Threshold Check (≤ 0.50) ─────────────── (verified or unknown)
     │
     ▼
MongoDB Attendance Log ───────────────── (name + subject + timestamp)
```

---

## 📁 Project Structure

```
Attendance_System/
│
├── app_windows_v2.py       # Main application (Windows-compatible entry point)
├── templates/              # Jinja2 HTML templates
│   ├── index.html          # Home page — student selects subject
│   ├── verify.html         # Live camera verification page
│   ├── login.html          # Admin login
│   ├── register.html       # Student face registration
│   ├── preprocessing.html  # Preprocessing view
│   └── dashboard.html      # Attendance records dashboard
├── static/                 # CSS, JS, and static assets
├── images/                 # Student face images — stored locally only (not on GitHub)
├── requirements.txt        # Python dependencies
└── README.md
```

> ⚠️ **Privacy Notice:** The `images/` folder is excluded from this repository via `.gitignore` to protect student biometric data. Images are stored locally only.

---

## ⚙️ Installation & Setup (Windows)

### Prerequisites
- Python 3.9+
- CMake
- Visual Studio C++ Build Tools (required by dlib)
- MongoDB Community Server
- Webcam

### Step-by-Step

**1. Clone the repository:**
```bash
git clone https://github.com/Malindu7/Attendance_System.git
cd Attendance_System
```

**2. Install CMake and dlib dependencies:**

Download and install:
- CMake: https://cmake.org/download/
- Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/

**3. Create and activate a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate
```

**4. Install Python dependencies:**
```bash
pip install cmake dlib face-recognition flask opencv-python numpy pymongo
```

**5. Start MongoDB:**
```bash
# Run CMD as Administrator
net start MongoDB
```

**6. Create the images folder:**
```bash
mkdir images
```

**7. Run the application:**
```bash
python app_windows_v2.py
```

**8. Open your browser and go to:**
```
http://127.0.0.1:5001
```

---

## 🌐 Application Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Home page — student selects subject |
| `/verify` | GET | Live camera attendance page |
| `/verify_feed` | GET | MJPEG camera stream for verification |
| `/login` | GET/POST | Admin authentication |
| `/register_page` | GET | Admin student registration page |
| `/register_feed` | GET | MJPEG stream for registration preview |
| `/capture_face` | POST | Captures photo and saves to images/ |
| `/dashboard` | GET | Attendance logs filtered by subject |

---

## 🧪 Face Matching — Tolerance Explained

The system uses a **tolerance threshold of 0.50** (Euclidean distance):

| Tolerance | Effect |
|---|---|
| 0.40 | Strict — may reject valid matches in different lighting |
| **0.50 (Used)** | Balanced — accurate under normal conditions |
| 0.60 | Lenient — higher risk of false positives |

---

## 🗄️ Database Schema (MongoDB)

Each attendance record is stored as a document:

```json
{
  "name":      "JOHN_DOE_D/BCS/24/0010",
  "subject":   "Python",
  "timestamp": "2025-03-17T10:45:32"
}
```

---

## ✅ Completed Features

- [x] Real-time MJPEG camera streaming in the browser
- [x] Face encoding using ResNet-34 CNN
- [x] Live HOG face detection and Euclidean distance verification
- [x] Attendance logging to MongoDB with subject tagging
- [x] 60-second duplicate prevention
- [x] Admin login and student registration workflow
- [x] Dashboard with subject-based filtering
- [x] Windows compatibility (DirectShow + reloader fix)

## 🔜 Planned Improvements

- [ ] Multi-frame confirmation before marking attendance (reduce false positives)
- [ ] Replace HOG with MTCNN or SSD for better detection accuracy
- [ ] Export attendance as Excel/CSV from the dashboard
- [ ] Anti-spoofing module to prevent photo-based attacks
- [ ] Multi-subject session management and lecturer controls

---

## 👨‍💻 Team — Group 04

| Name | Index Number |
|---|---|
| DMMH Dissanayake | D/BCE/24/0007 |
| HDT Gajanayaka | D/BCS/24/0010 |
| OM Kavishan | D/DBA/23/0012 |

---

## 🏫 Project Context

> **Module:** CS31092 – Digital Image Processing
> **University:** General Sir John Kotelawala Defence University (KDU)
> **Faculty:** Faculty of Computing
> **Year:** 4th Year, 7th Semester

---

## 📄 License

This project is developed for academic purposes only at KDU.

