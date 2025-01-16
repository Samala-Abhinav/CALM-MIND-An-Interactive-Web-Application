from flask import *
import mysql.connector, joblib, random, string, base64, pickle
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_mail import Mail, Message
import pandas as pd 
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import os,random,shutil
import cv2
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
import math
import os
from PIL import Image

app = Flask(__name__)
app.secret_key = 'yoga' 

# Initialize Flask-Mail
mail = Mail(app)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='yoga'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data
    


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']

        # img = request.files['img']
        # binary_data = img.read()

        if password == c_password:
            query = "SELECT email FROM users"
            exist_data = retrivequery2(query)
            exist_email_list = [i[0] for i in exist_data]

            if email not in exist_email_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)

                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID already exists!")
        return render_template('register.html', message="Confirm password does not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT * FROM users WHERE email = %s"
        values = (email,)
        user_data = retrivequery1(query, values)

        if user_data:
            if password == user_data[0][3]:
                session["user_id"] = user_data[0][0]
                session["user_name"] = user_data[0][1]
                session["user_email"] = user_data[0][2]

                return redirect("/home")
            return render_template('login.html', message="Invalid Password!!")
        return render_template('login.html', message="This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')






### yoga module

def recommend_poses(mood, u, sigma, vt, top_n=3):
    mood_idx = mood_to_idx[mood]
    mood_latent_features = np.dot(u[mood_idx, :], sigma)
    pose_scores = np.dot(mood_latent_features, vt)

    top_pose_indices = np.argsort(pose_scores)[::-1][:top_n]
    recommended_poses = [poses[idx] for idx in top_pose_indices]

    return recommended_poses


df = pd.read_csv("dataset/Recommendation_yoga_data.csv")

moods = df['Mood Before'].unique()
poses = df['Yoga Practice'].unique()

mood_to_idx = {mood: idx for idx, mood in enumerate(moods)}
pose_to_idx = {pose: idx for idx, pose in enumerate(poses)}

interaction_matrix = csr_matrix((len(mood_to_idx), len(pose_to_idx)), dtype=int)
interaction_matrix_float = interaction_matrix.astype('float32')

for _, row in df.iterrows():
    mood_idx = mood_to_idx[row['Mood Before']]
    pose_idx = pose_to_idx[row['Yoga Practice']]
    interaction_matrix_float[mood_idx, pose_idx] += 1

num_features = min(interaction_matrix_float.shape) - 1  # Choose a suitable number of features
u, sigma, vt = svds(interaction_matrix_float, k=num_features)
sigma_diag_matrix = np.diag(sigma)


@app.route('/yoga1', methods=['POST','GET'])
def yoga1():
    if request.method=='POST':
        mood = request.form['mood']

        recommended_poses = recommend_poses(mood, u, sigma_diag_matrix, vt, top_n=3)

        main_folder_path = os.path.join(os.getcwd(),'data')
        to_copy_path = os.path.join(os.getcwd(),'static','img')

        files_to_send = []
        for folder in recommended_poses:
            to_search_folder = os.path.join(main_folder_path,folder)
            for temp in os.walk(to_search_folder):
                single_file_name = random.choice(temp[2])
                files_to_send.append((f'/static/img/{single_file_name}',folder))
                shutil.copyfile(os.path.join(to_search_folder,single_file_name) , os.path.join(to_copy_path,single_file_name))
                break

        return render_template('yoga2.html', files = files_to_send, mood = mood)
    return render_template('yoga1.html')


import os
import cv2
import numpy as np
import mediapipe as mp


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Second point (vertex)
    c = np.array(c)  # Third point
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Function to provide corrections based on expected ranges
def provide_correction_statements(detected_angles, expected_ranges):
    corrections = []
    for joint, angle in detected_angles.items():
        min_angle, max_angle = expected_ranges[joint]
        if not (min_angle <= angle <= max_angle):
            corrections.append(f"{joint}: Adjust to be within {min_angle}° to {max_angle}° (Current: {angle:.2f}°)")
    return corrections

@app.route('/yoga2', methods=["POST", "GET"])
def yoga2():
    prd_result = ""
    annotated_image_filename = ""
    
    if request.method == "POST":
        user_id = session["user_id"]
        user_name = session["user_name"]
        user_email = session["user_email"]

        pose_name = request.form.get("pose_name")
        mood = request.form['mood']
        myfile = request.files['img']
        fn = myfile.filename

        # Ensure the filename is secure
        from werkzeug.utils import secure_filename
        fn = secure_filename(fn)

        # Define paths
        saved_images_path = os.path.join('static', 'saved_images')
        if not os.path.exists(saved_images_path):
            os.makedirs(saved_images_path)
        mypath = os.path.join(saved_images_path, fn)
        myfile.save(mypath)

        image_path = mypath
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initialize Mediapipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

        # Process the image to detect pose landmarks
        results = pose.process(image_rgb)

        # Define expected angle ranges for a specific pose
        expected_angle_ranges = {
            "Shoulder Angle": (90, 120),  # Example: expected range for the shoulder
            "Elbow Angle": (150, 180),
            "Hip Angle": (100, 140),
            "Knee Angle": (90, 110),
        }

        # Check if landmarks are detected
        if results.pose_landmarks:
            # Extract pose landmarks
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape  # Get image dimensions for scaling

            # Extract coordinates of key points
            points = {
                "left_shoulder": [int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                                  int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)],
                "left_elbow": [int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w),
                              int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h)],
                "left_wrist": [int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w),
                              int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h)],
                "left_hip": [int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h)],
                "left_knee": [int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * w),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h)],
                "left_ankle": [int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * w),
                               int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h)],
            }

            # Calculate angles
            detected_angles = {
                "Shoulder Angle": calculate_angle(points["left_elbow"], points["left_shoulder"], points["left_hip"]),
                "Elbow Angle": calculate_angle(points["left_wrist"], points["left_elbow"], points["left_shoulder"]),
                "Hip Angle": calculate_angle(points["left_shoulder"], points["left_hip"], points["left_knee"]),
                "Knee Angle": calculate_angle(points["left_hip"], points["left_knee"], points["left_ankle"]),
            }

            # Provide correction statements
            corrections = provide_correction_statements(detected_angles, expected_angle_ranges)

            # Draw points
            for key, point in points.items():
                cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)  # Draw points

            # Draw lines connecting joints
            cv2.line(image, tuple(points["left_shoulder"]), tuple(points["left_elbow"]), (255, 0, 0), 2)
            cv2.line(image, tuple(points["left_elbow"]), tuple(points["left_wrist"]), (255, 0, 0), 2)
            cv2.line(image, tuple(points["left_shoulder"]), tuple(points["left_hip"]), (255, 0, 0), 2)
            cv2.line(image, tuple(points["left_hip"]), tuple(points["left_knee"]), (255, 0, 0), 2)
            cv2.line(image, tuple(points["left_knee"]), tuple(points["left_ankle"]), (255, 0, 0), 2)

            # Annotate angles on the image
            for joint, angle in detected_angles.items():
                if "Elbow" in joint:
                    key_point = "left_elbow"
                elif "Hip" in joint:
                    key_point = "left_hip"
                elif "Knee" in joint:
                    key_point = "left_knee"
                else:
                    key_point = "left_shoulder"  # Default case
                if key_point in points:
                    cv2.putText(image, f"{int(angle)}\u00B0", tuple(points[key_point]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Save the annotated image
            annotated_image_filename = f"annotated_{fn}"
            annotated_image_path = os.path.join(saved_images_path, annotated_image_filename)
            cv2.imwrite(annotated_image_path, image)

            # Prepare feedback
            if corrections:
                prd_result = corrections
            else:
                prd_result = "Pose is within the expected range for all angles!"
            
            with open(mypath, 'rb') as f:
                uploaded_img_binary_data = f.read()

            with open(annotated_image_path, 'rb') as f:
                corrected_img_binary_data = f.read()


            # from datetime import datetime
            # # Get current date and time
            # now = datetime.now()
            # # Extract date in YYYY-MM-DD format
            # current_date = now.strftime("%Y-%m-%d")      

            query = "INSERT INTO dashboard (name, email, mood, yoga, uploaded_img, corrected_img, feedback) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            values = (user_name, user_email, mood, pose_name, uploaded_img_binary_data, corrected_img_binary_data, str(prd_result))
            executionquery(query, values)

        else:
            prd_result = "No pose landmarks detected in the image."

        # Release Mediapipe resources
        pose.close()
        return render_template('yoga3.html', feedback = prd_result, image_name = annotated_image_filename if annotated_image_filename else fn)

    



@app.route('/dashboard', methods=["POST","GET"])
def dashboard():
    user_email = session["user_email"]
    if request.method=="POST":
        frm_date = request.form['frm_date']
        to_date = request.form['to_date']

        query = "SELECT * FROM dashboard WHERE email = %s AND (date BETWEEN %s AND %s)"
        values = (user_email, frm_date, to_date)
        dashboard_data = retrivequery1(query, values)

    else:
        query = "SELECT * FROM dashboard WHERE email = %s"
        values = (user_email,)
        dashboard_data = retrivequery1(query, values)

    dashboard_list = []
    for item in dashboard_data:
        dashboard_list.append({
            'id': item[0],
            'user_name': item[1],
            'user_email': item[2],
            'mood_name': item[3],
            'yoga_name': item[4],
            'uploaded_img': base64.b64encode(item[5]).decode('utf-8'),
            'corrected_img': base64.b64encode(item[6]).decode('utf-8'),
            'feedback': item[7],
            'date': item[8]
        })

    return render_template('dashboard.html', dashboard_data = dashboard_list)







@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')





if __name__ == '__main__':
    app.run(debug = True)