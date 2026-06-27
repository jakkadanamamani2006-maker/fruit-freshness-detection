🍎 Fruit Freshness Detection using Deep Learning

📌 Project Overview

Fruit Freshness Detection is an AI-powered Machine Learning project that classifies fruits as Fresh or Rotten using Deep Learning and Computer Vision techniques.

The application is developed using TensorFlow/Keras for image classification and Streamlit for creating an interactive web application.

This project helps in:

- Food quality monitoring
- Reducing food waste
- Smart agriculture applications
- Automated fruit inspection systems

---

🚀 Features

✅ Upload fruit images
✅ Detect Fresh or Rotten fruit
✅ Real-time prediction using CNN model
✅ Confidence score display
✅ Interactive Streamlit web application
✅ Attractive animated UI
✅ Automatic model downloading using Google Drive

---

🛠️ Technologies Used

Technology| Purpose
Python| Programming Language
TensorFlow| Deep Learning Framework
Keras| Model Building
Streamlit| Web Application
NumPy| Numerical Computation
OpenCV| Image Processing
PIL (Pillow)| Image Handling
gdown| Download model from Google Drive
Git| Version Control
GitHub| Repository Hosting

---

💻 Software Requirements

Software| Version
Python| 3.11.9
Git| 2.54.1
TensorFlow| 2.x
Streamlit| Latest
VS Code| Recommended IDE

---

📦 Python Libraries Used

tensorflow
streamlit
numpy
pillow
gdown
opencv-python

---

📂 Project Structure

fruit-freshness-detection/
│
├── app.py
├── fruit_freshness_model.keras
├── labels.txt
├── requirements.txt
├── README.md

---

⚙️ Installation and Setup

Step 1: Clone Repository

git clone https://github.com/jakkadanamamani2006-maker/fruit-freshness-detection.git

---

Step 2: Move into Project Folder

cd fruit-freshness-detection

---

Step 3: Install Required Libraries

pip install -r requirements.txt

---

Step 4: Run Streamlit Application

streamlit run app.py

---

🌐 Streamlit Web Application

The project includes an interactive Streamlit web application for real-time fruit freshness detection.

🔹 Functionalities

- Upload fruit images ("jpg", "jpeg", "png")
- Predict whether the fruit is Fresh or Rotten
- Display confidence percentage
- Animated UI with floating fruit emojis
- Responsive interface

---

🧠 Machine Learning Model

Model Details

Parameter| Description
Model Type| CNN (Convolutional Neural Network)
Framework| TensorFlow / Keras
Input Size| 224 x 224
Output Classes| 6 Classes

---

🧪 Prediction Classes

The model predicts the following classes:

- Fresh Apple 🍎
- Fresh Banana 🍌
- Fresh Orange 🍊
- Rotten Apple ❌
- Rotten Banana ❌
- Rotten Orange ❌

---

🔄 Working Procedure

1. User uploads fruit image
2. Image resized to "224 x 224"
3. Image converted into NumPy array
4. CNN model processes image
5. Prediction generated
6. Result displayed with confidence score

---

🎨 User Interface Features

- Green gradient background
- Floating fruit animations
- Modern responsive design
- Progress bar for confidence score
- Eco-friendly themed interface

---

📥 Automatic Model Download

The trained ".keras" model is automatically downloaded from Google Drive using "gdown".

gdown.download(
    "Google Drive Model Link",
    MODEL_PATH,
    quiet=False
)

---

📊 Confidence Score Calculation

The prediction confidence is calculated using:

confidence = np.max(prediction) * 100

This shows how confident the model is about the prediction result.

---

📸 Output Screenshots

Home Page

(Add Screenshot Here)

Prediction Result

(Add Screenshot Here)

---

🎯 Applications

- Smart Agriculture
- Food Industries
- Supermarkets
- Fruit Quality Inspection
- Inventory Management
- Automated Food Monitoring Systems

---

🔮 Future Enhancements

- Add more fruit categories
- Improve model accuracy
- Real-time camera detection
- Cloud deployment
- Mobile application support

---

👩‍🎓 Student Details

Detail| Information
Student Name| J. Amani
Project Type| Machine Learning Mini Project
Domain| Artificial Intelligence & Deep Learning
Tools Used| Python, TensorFlow, Streamlit, GitHub

---

📚 Project Domain

- Machine Learning
- Deep Learning
- Computer Vision
- Artificial Intelligence

---

🙏 Acknowledgement

I would like to express my sincere gratitude to my faculty members and project guide for their valuable guidance and support throughout the development of this project.

---

👩‍💻 Author

J. Amani

Machine Learning & IoT Enthusiast

---

🔗 GitHub Repository

https://github.com/jakkadanamamani2006-maker/fruit-freshness-detection