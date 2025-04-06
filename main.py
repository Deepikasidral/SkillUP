import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import random
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mysql.connector
from datetime import datetime
import hashlib
import base64
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq


# Set page configuration
st.set_page_config(
    page_title="STEM Navigator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom dark theme CSS
st.markdown("""
<style>
    .main {
        background-color: #121212;
        color: #FFFFFF;
    }
    .stButton button {
        background-color: #8c52ff;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 8px 16px;
    }
    .stButton button:hover {
        background-color: #7340d4;
    }
    .stTextInput, .stNumberInput {
        background-color: #2c2c2c;
        border-radius: 5px;
        border: 1px solid #444444;
        color: white;
    }
    .stSelectbox select {
        background-color: #2c2c2c;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #1E1E1E;
    }
    h1, h2, h3 {
        color: #8c52ff;
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        background-color: #1E1E1E;
        margin-bottom: 20px;
        border: 1px solid #333333;
    }
    .highlight {
        color: #8c52ff;
        font-weight: bold;
    }
    .progress-container {
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #2c2c2c;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .bot-message {
        background-color: #3c2c5c;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #333333;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #8c52ff;
    }
    .metric-label {
        font-size: 14px;
        color: #AAAAAA;
    }
</style>
""", unsafe_allow_html=True)

# Database connection function
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="deepika@1711",
            database="stem_navigator"
        )
        return conn
    except mysql.connector.Error as err:
        st.error(f"Database Error: {err}")
        return None

# Initialize database if it doesn't exist
def initialize_database():
    try:
        # Connect to MySQL server
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="deepika@1711"
        )
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS stem_navigator")
        cursor.execute("USE stem_navigator")
        
        # Create users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            full_name VARCHAR(100) NOT NULL,
            grade INT NOT NULL,
            learning_style VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create assessments table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            subject VARCHAR(50) NOT NULL,
            score FLOAT NOT NULL,
            difficulty FLOAT NOT NULL,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)
        
        # Create learning_paths table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_paths (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            subject VARCHAR(50) NOT NULL,
            current_module VARCHAR(100) NOT NULL,
            progress FLOAT DEFAULT 0.0,
            difficulty_level FLOAT DEFAULT 0.5,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)
        
        # Create collaborative_challenges table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS collaborative_challenges (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(100) NOT NULL,
            description TEXT NOT NULL,
            subject VARCHAR(50) NOT NULL,
            difficulty_level FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create challenge_participants table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS challenge_participants (
            id INT AUTO_INCREMENT PRIMARY KEY,
            challenge_id INT NOT NULL,
            user_id INT NOT NULL,
            joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (challenge_id) REFERENCES collaborative_challenges(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)
        
        # Create badge table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS badges (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            badge_name VARCHAR(50) NOT NULL,
            badge_description TEXT NOT NULL,
            earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)
        
        # Insert sample data for challenges if none exist
        cursor.execute("SELECT COUNT(*) FROM collaborative_challenges")
        if cursor.fetchone()[0] == 0:
            sample_challenges = [
                ("Build a Simple Weather Station", "Create a device that measures temperature, humidity, and pressure using basic electronic components.", "Physics", 0.6),
                ("DNA Extraction Challenge", "Extract DNA from fruits and vegetables using household materials.", "Biology", 0.4),
                ("Sustainable City Design", "Design a model of a sustainable city focusing on renewable energy and waste management.", "Environmental Science", 0.7),
                ("Algebra in Real Life", "Find and document 10 examples of algebraic equations in everyday scenarios.", "Mathematics", 0.5),
                ("Machine Learning Image Classifier", "Build a simple ML model that can identify different types of animals.", "Computer Science", 0.8)
            ]
            cursor.executemany("""
            INSERT INTO collaborative_challenges (title, description, subject, difficulty_level)
            VALUES (%s, %s, %s, %s)
            """, sample_challenges)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Create sample content for Random Forest
        create_sample_content()
        
    except mysql.connector.Error as err:
        st.error(f"Database Initialization Error: {err}")

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# User authentication functions
def register_user(username, password, full_name, grade, learning_style):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            password_hash = hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, password_hash, full_name, grade, learning_style) VALUES (%s, %s, %s, %s, %s)",
                (username, password_hash, full_name, grade, learning_style)
            )
            conn.commit()
            
            # Get the user ID
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            user_id = cursor.fetchone()[0]
            
            # Initialize learning paths for the user
            subjects = ["Mathematics", "Physics", "Biology", "Chemistry", "Computer Science"]
            initial_modules = {
                "Mathematics": "Number Systems and Operations",
                "Physics": "Forces and Motion",
                "Biology": "Cell Structure and Function",
                "Chemistry": "Atoms and Elements",
                "Computer Science": "Introduction to Programming"
            }
            
            for subject in subjects:
                cursor.execute(
                    "INSERT INTO learning_paths (user_id, subject, current_module, progress, difficulty_level) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, subject, initial_modules[subject], 0.0, 0.5)
                )
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except mysql.connector.Error as err:
            st.error(f"Registration Error: {err}")
            conn.close()
            return False
    return False

def login_user(username, password):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            password_hash = hash_password(password)
            cursor.execute(
                "SELECT * FROM users WHERE username = %s AND password_hash = %s",
                (username, password_hash)
            )
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            return user
        except mysql.connector.Error as err:
            st.error(f"Login Error: {err}")
            conn.close()
            return None
    return None

# Function to create sample content for ML model training
def create_sample_content():
    # Create directories if they don't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Check if sample data already exists
    if os.path.exists('data/student_performance.csv'):
        return
    
    # Generate synthetic student performance data
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    grade_levels = np.random.randint(6, 13, n_samples)  # Grades 6-12
    subjects = np.random.choice(['Mathematics', 'Physics', 'Biology', 'Chemistry', 'Computer Science'], n_samples)
    prior_knowledge = np.random.uniform(0, 1, n_samples)
    time_spent = np.random.exponential(scale=2.5, size=n_samples)  # Hours spent on topic
    engagement_level = np.random.uniform(0, 1, n_samples)
    learning_style = np.random.choice(['Visual', 'Auditory', 'Reading/Writing', 'Kinesthetic'], n_samples)
    difficulty = np.random.uniform(0, 1, n_samples)
    
    # Subject encoding for easier calculation
    subject_encoding = {
        'Mathematics': 0,
        'Physics': 1,
        'Biology': 2,
        'Chemistry': 3,
        'Computer Science': 4
    }
    
    subject_nums = np.array([subject_encoding[s] for s in subjects])
    
    # Calculate proficiency level (target) based on features with some noise
    # Different weights for different subjects to simulate real-world conditions
    proficiency = (
        0.3 * prior_knowledge +
        0.2 * (time_spent / 5).clip(0, 1) +
        0.15 * engagement_level +
        0.1 * (grade_levels - 6) / 6 -
        0.25 * difficulty +
        0.05 * np.random.normal(0, 1, n_samples)
    )
    
    # Adjust by subject (some subjects might be harder)
    subject_difficulty = {
        0: 0,      # Mathematics - baseline
        1: -0.05,  # Physics - slightly harder
        2: 0.02,   # Biology - slightly easier
        3: -0.03,  # Chemistry - slightly harder
        4: -0.01   # Computer Science - slightly harder
    }
    
    for i, s in enumerate(subject_nums):
        proficiency[i] += subject_difficulty[s]
    
    # Clip to [0, 1] range
    proficiency = np.clip(proficiency, 0, 1)
    
    # Create test scores based on proficiency
    test_scores = np.clip(proficiency * 100 + np.random.normal(0, 5, n_samples), 0, 100)
    
    # Categorize proficiency into 5 levels (0-4) for classification tasks
    proficiency_level = np.digitize(proficiency, bins=[0.2, 0.4, 0.6, 0.8]) - 1
    
    # Topic mastery (specific areas within subjects)
    topic_mastery = {}
    topics = {
        'Mathematics': ['Algebra', 'Geometry', 'Calculus', 'Statistics', 'Trigonometry'],
        'Physics': ['Mechanics', 'Electricity', 'Magnetism', 'Optics', 'Thermodynamics'],
        'Biology': ['Cells', 'Genetics', 'Ecology', 'Evolution', 'Physiology'],
        'Chemistry': ['Elements', 'Compounds', 'Reactions', 'Organic', 'Physical'],
        'Computer Science': ['Programming', 'Data Structures', 'Algorithms', 'Databases', 'Web Development']
    }
    
    for subject in topics:
        for topic in topics[subject]:
            topic_mastery[f"{subject}_{topic}"] = np.zeros(n_samples)
    
    # Fill topic mastery with values related to overall proficiency but with variations
    for i in range(n_samples):
        subject = subjects[i]
        for topic in topics[subject]:
            # Base topic mastery on overall proficiency with random variation
            base = proficiency[i]
            variation = np.random.normal(0, 0.15)  # Standard deviation controls topic variation
            topic_mastery[f"{subject}_{topic}"][i] = np.clip(base + variation, 0, 1)
    
    # Convert to pandas DataFrame
    data = {
        'student_id': np.arange(1, n_samples + 1),
        'grade_level': grade_levels,
        'subject': subjects,
        'prior_knowledge': prior_knowledge,
        'time_spent': time_spent,
        'engagement_level': engagement_level,
        'learning_style': learning_style,
        'difficulty': difficulty,
        'test_score': test_scores,
        'proficiency': proficiency,
        'proficiency_level': proficiency_level
    }
    
    # Add topic mastery columns
    for key, values in topic_mastery.items():
        data[key] = values
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('data/student_performance.csv', index=False)
    
    # Train and save the Random Forest model
    train_and_save_rf_model(df)

# Train and save Random Forest model
def train_and_save_rf_model(df):
    if os.path.exists('data/rf_model.pkl'):
        return
    
    # Encode categorical features
    df_encoded = df.copy()
    df_encoded['learning_style'] = pd.Categorical(df_encoded['learning_style']).codes
    df_encoded['subject'] = pd.Categorical(df_encoded['subject']).codes
    
    # Features for training
    features = ['grade_level', 'subject', 'prior_knowledge', 'time_spent', 
                'engagement_level', 'learning_style', 'difficulty']
    
    # For feature importance analysis, include topic mastery
    topic_columns = [col for col in df_encoded.columns if '_' in col and col not in ['student_id', 'proficiency_level']]
    all_features = features + topic_columns
    
    # Train model to predict proficiency level
    X = df_encoded[all_features]
    y = df_encoded['proficiency_level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    feature_importance.to_csv('data/feature_importance.csv', index=False)
    
    # Save model
    with open('data/rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Load the trained Random Forest model
def load_rf_model():
    try:
        with open('data/rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model has been trained.")
        return None

# Function to predict proficiency level and learning gaps
def predict_proficiency_and_gaps(user_data):
    model = load_rf_model()
    if model is None:
        return None, None
    
    # Make prediction
    prediction = model.predict([user_data])[0]
    
    # Get feature importances to identify gaps
    try:
        feature_importance = pd.read_csv('data/feature_importance.csv')
        
        # Find topic mastery columns
        topic_cols = [col for col in feature_importance['Feature'] if '_' in col]
        
        # Get top 3 important topics
        important_topics = feature_importance[feature_importance['Feature'].isin(topic_cols)].head(10)
        
        # Find learning gaps (topics with low values in user data)
        gaps = []
        for i, row in important_topics.iterrows():
            topic = row['Feature']
            if topic in user_data.index and user_data[topic] < 0.6:  # Threshold for identifying gaps
                gaps.append((topic.replace('_', ' - '), user_data[topic]))
        
        # Sort gaps by mastery level (ascending)
        gaps.sort(key=lambda x: x[1])
        
        return prediction, gaps
    except Exception as e:
        st.error(f"Error analyzing learning gaps: {e}")
        return prediction, []

# Function to simulate a diagnostic test
def run_diagnostic_test(subject, grade, learning_style):
    # Load sample questions
    questions = get_sample_questions(subject, grade)
    
    if not questions:
        st.error("Failed to load questions for the diagnostic test.")
        return None, None
    
    st.subheader("📝 Diagnostic Test")
    st.write(f"Subject: **{subject}**")
    st.write("Please answer the following questions to assess your knowledge level.")
    
    # Initialize user responses
    responses = []
    correct_answers = 0
    total_time = 0
    
    # Display progress bar
    progress_bar = st.progress(0)
    
    for i, q in enumerate(questions):
        with st.container():
            st.markdown(f"**Question {i+1}:** {q['question']}")
            
            # Record start time
            start_time = time.time()
            
            # Handle different question types
            if q['type'] == 'multiple_choice':
                options = q['options']
                user_answer = st.radio("Select your answer:", options, key=f"q_{i}")
                is_correct = (user_answer == q['correct_answer'])
            else:  # Assume text input
                user_answer = st.text_input("Your answer:", key=f"q_{i}")
                is_correct = (user_answer.lower() == q['correct_answer'].lower())
            
            # Calculate time taken when next button is pressed
            submit_button = st.button("Submit Answer", key=f"submit_{i}")
            
            if submit_button:
                end_time = time.time()
                time_taken = end_time - start_time
                total_time += time_taken
                
                if is_correct:
                    correct_answers += 1
                    st.success("Correct!")
                else:
                    st.error(f"Incorrect. The correct answer is: {q['correct_answer']}")
                
                responses.append({
                    'question_id': i,
                    'topic': q['topic'],
                    'is_correct': is_correct,
                    'time_taken': time_taken,
                    'difficulty': q['difficulty']
                })
                
                # Update progress
                progress_bar.progress((i + 1) / len(questions))
                
                # Wait for a moment to show feedback
                time.sleep(1)
                
                if i < len(questions) - 1:
                    st.experimental_rerun()
                else:
                    # Test completed
                    st.balloons()
                    
                    # Process and return results
                    score = (correct_answers / len(questions)) * 100
                    
                    # Prepare user data for prediction
                    topic_mastery = {}
                    for resp in responses:
                        topic = resp['topic']
                        if topic not in topic_mastery:
                            topic_mastery[topic] = {'correct': 0, 'total': 0}
                        
                        topic_mastery[topic]['total'] += 1
                        if resp['is_correct']:
                            topic_mastery[topic]['correct'] += 1
                    
                    # Calculate mastery levels
                    for topic in topic_mastery:
                        topic_mastery[topic] = topic_mastery[topic]['correct'] / topic_mastery[topic]['total']
                    
                    # Average difficulty of the test
                    avg_difficulty = sum(q['difficulty'] for q in questions) / len(questions)
                    
                    # Create user data dictionary for prediction
                    user_data = {
                        'grade_level': grade,
                        'subject': subject_to_code(subject),
                        'prior_knowledge': score / 100,  # Use test score as prior knowledge
                        'time_spent': total_time / 3600,  # Convert to hours
                        'engagement_level': 0.8,  # Assume high engagement for now
                        'learning_style': learning_style_to_code(learning_style),
                        'difficulty': avg_difficulty
                    }
                    
                    # Add topic mastery
                    for topic, mastery in topic_mastery.items():
                        user_data[f"{subject}_{topic}"] = mastery
                    
                    return score, pd.Series(user_data)
    
    return None, None

# Helper functions for encoding categorical variables
def subject_to_code(subject):
    subjects = {'Mathematics': 0, 'Physics': 1, 'Biology': 2, 'Chemistry': 3, 'Computer Science': 4}
    return subjects.get(subject, 0)

def learning_style_to_code(style):
    styles = {'Visual': 0, 'Auditory': 1, 'Reading/Writing': 2, 'Kinesthetic': 3}
    return styles.get(style, 0)

# Function to get sample questions for the diagnostic test
def get_sample_questions(subject, grade):
    # Define questions for each subject and adjust based on grade level
    difficulty_factor = (grade - 6) / 6  # Normalize to 0-1 range
    
    base_questions = {
        'Mathematics': [
            {
                'question': "Solve for x: 2x + 5 = 13",
                'type': 'text',
                'correct_answer': "4",
                'topic': 'Algebra',
                'difficulty': 0.3 + (difficulty_factor * 0.2)
            },
            {
                'question': "What is the area of a circle with radius 5?",
                'type': 'multiple_choice',
                'options': ["25π", "10π", "25", "50π"],
                'correct_answer': "25π",
                'topic': 'Geometry',
                'difficulty': 0.4 + (difficulty_factor * 0.2)
            },
            {
                'question': "If you toss a fair coin 3 times, what is the probability of getting exactly 2 heads?",
                'type': 'multiple_choice',
                'options': ["1/8", "3/8", "1/2", "5/8"],
                'correct_answer': "3/8",
                'topic': 'Statistics',
                'difficulty': 0.6 + (difficulty_factor * 0.2)
            },
            {
                'question': "What is the value of sin(30°)?",
                'type': 'multiple_choice',
                'options': ["1/4", "1/3", "1/2", "√3/2"],
                'correct_answer': "1/2",
                'topic': 'Trigonometry',
                'difficulty': 0.5 + (difficulty_factor * 0.2)
            },
            {
                'question': "Simplify: (3x² + 2x - 1) - (x² - 2x + 3)",
                'type': 'multiple_choice',
                'options': ["2x² + 4x - 4", "4x² + 4", "2x² - 4", "4x - 4"],
                'correct_answer': "2x² + 4x - 4",
                'topic': 'Algebra',
                'difficulty': 0.7 + (difficulty_factor * 0.2)
            }
        ],
        'Physics': [
            {
                'question': "What is the formula for Newton's Second Law?",
                'type': 'multiple_choice',
                'options': ["F = ma", "E = mc²", "F = G(m₁m₂)/r²", "v = u + at"],
                'correct_answer': "F = ma",
                'topic': 'Mechanics',
                'difficulty': 0.3 + (difficulty_factor * 0.2)
            },
            {
                'question': "What is the unit of electric current?",
                'type': 'multiple_choice',
                'options': ["Volt", "Watt", "Ampere", "Ohm"],
                'correct_answer': "Ampere",
                'topic': 'Electricity',
                'difficulty': 0.4 + (difficulty_factor * 0.2)
            },
            {
                'question': "Light travels from air into glass. What happens to its speed?",
                'type': 'multiple_choice',
                'options': ["Increases", "Decreases", "Remains the same", "Becomes zero"],
                'correct_answer': "Decreases",
                'topic': 'Optics',
                'difficulty': 0.5 + (difficulty_factor * 0.2)
            },
            {
                'question': "What happens to gas pressure when volume decreases, assuming temperature remains constant?",
                'type': 'multiple_choice',
                'options': ["Increases", "Decreases", "Remains the same", "Becomes zero"],
                'correct_answer': "Increases",
                'topic': 'Thermodynamics',
                'difficulty': 0.6 + (difficulty_factor * 0.2)
            },
            {
                'question': "Calculate the force exerted by a 5kg object accelerating at 2m/s².",
                'type': 'text',
                'correct_answer': "10",
                'topic': 'Mechanics',
                'difficulty': 0.5 + (difficulty_factor * 0.2)
            }
        ],
        'Biology': [
            {
                'question': "What is the powerhouse of the cell?",
                'type': 'multiple_choice',
                'options': ["Nucleus", "Mitochondria", "Endoplasmic Reticulum", "Golgi Apparatus"],
                'correct_answer': "Mitochondria",
                'topic': 'Cells',
                'difficulty': 0.3 + (difficulty_factor * 0.2)
            },
            {
                'question': "What molecule carries genetic information?",
                'type': 'multiple_choice',
                'options': ["RNA", "DNA", "Protein", "Lipid"],
                'correct_answer': "DNA",
                'topic': 'Genetics',
                'difficulty': 0.4 + (difficulty_factor * 0.2)
            },
            {
                'question': "What process do plants use to make food?",
                'type': 'multiple_choice',
                'options': ["Respiration", "Photosynthesis", "Digestion", "Fermentation"],
                'correct_answer': "Photosynthesis",
                'topic': 'Physiology',
                'difficulty': 0.3 + (difficulty_factor * 0.2)
            },
            {
                'question': "What is the relationship between predator and prey called?",
                'type': 'multiple_choice',
                'options': ["Mutualism", "Parasitism", "Competition", "Predation"],
                'correct_answer': "Predation",
                'topic': 'Ecology',
                'difficulty': 0.5 + (difficulty_factor * 0.2)
            },
            {
                'question': "Humans and chimpanzees share approximately what percentage of their DNA?",
                'type': 'multiple_choice',
                'options': ["50%", "75%", "98%", "25%"],
                'correct_answer': "98%",
                'topic': 'Evolution',
                'difficulty': 0.6 + (difficulty_factor * 0.2)
            }
        ],
        'Chemistry': [
            {
                'question': "What is the chemical symbol for gold?",
                'type': 'text',
                'correct_answer': "Au",
                'topic': 'Elements',
                'difficulty': 0.3 + (difficulty_factor * 0.2)
            },
            {
                'question': "What is the pH of a neutral solution?",
                'type': 'text',
                'correct_answer': "7",
                'topic': 'Reactions',
                'difficulty': 0.4 + (difficulty_factor * 0.2)
            },
            {
                'question': "What type of bond forms when electrons are shared between atoms?",
                'type': 'multiple_choice',
                'options': ["Ionic bond", "Covalent bond", "Hydrogen bond", "Metallic bond"],
                'correct_answer': "Covalent bond",
                'topic': 'Compounds',
                'difficulty': 0.5 + (difficulty_factor * 0.2)
            },
            {
                'question': "What is the chemical formula for water?",
                'type': 'text',
                'correct_answer': "H2O",
                'topic': 'Compounds',
                'difficulty': 0.2 + (difficulty_factor * 0.2)
            },
            {

'question': "Balance this chemical equation: H₂ + O₂ → H₂O",
                'type': 'multiple_choice',
                'options': ["H₂ + O₂ → H₂O", "2H₂ + O₂ → 2H₂O", "H₂ + 2O₂ → H₂O₂", "2H₂ + O₂ → H₂O₂"],
                'correct_answer': "2H₂ + O₂ → 2H₂O",
                'topic': 'Reactions',
                'difficulty': 0.6 + (difficulty_factor * 0.2)
            }
        ],
        'Computer Science': [
            {
                'question': "What does CPU stand for?",
                'type': 'multiple_choice',
                'options': ["Central Processing Unit", "Computer Personal Unit", "Central Program Utility", "Core Processing Unit"],
                'correct_answer': "Central Processing Unit",
                'topic': 'Programming',
                'difficulty': 0.3 + (difficulty_factor * 0.2)
            },
            {
                'question': "Which data structure operates on a LIFO principle?",
                'type': 'multiple_choice',
                'options': ["Queue", "Stack", "Tree", "Graph"],
                'correct_answer': "Stack",
                'topic': 'Data Structures',
                'difficulty': 0.5 + (difficulty_factor * 0.2)
            },
            {
                'question': "What is the time complexity of binary search?",
                'type': 'multiple_choice',
                'options': ["O(n)", "O(n²)", "O(log n)", "O(n log n)"],
                'correct_answer': "O(log n)",
                'topic': 'Algorithms',
                'difficulty': 0.7 + (difficulty_factor * 0.2)
            },
            {
                'question': "What language is commonly used for web styling?",
                'type': 'multiple_choice',
                'options': ["HTML", "JavaScript", "CSS", "Python"],
                'correct_answer': "CSS",
                'topic': 'Web Development',
                'difficulty': 0.4 + (difficulty_factor * 0.2)
            },
            {
                'question': "What does SQL stand for?",
                'type': 'multiple_choice',
                'options': ["Structured Query Language", "Simple Question Language", "System Quality Language", "Sequential Query Logic"],
                'correct_answer': "Structured Query Language",
                'topic': 'Databases',
                'difficulty': 0.5 + (difficulty_factor * 0.2)
            }
        ]
    }
    
    # Return questions for the selected subject
    return base_questions.get(subject, [])

# Function to create personalized learning path
def create_learning_path(user_id, subject, proficiency_level, gaps):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            
            # Get subject modules based on proficiency level
            modules = get_subject_modules(subject, proficiency_level)
            
            # Prioritize modules that address learning gaps
            if gaps:
                for gap, score in gaps:
                    gap_topic = gap.split(' - ')[1]
                    # Add remedial modules for identified gaps
                    remedial_modules = get_remedial_modules(subject, gap_topic)
                    # Insert at the beginning of the learning path
                    modules = remedial_modules + modules
            
            # Update learning path in the database
            current_module = modules[0]['title'] if modules else "Introduction"
            cursor.execute(
                "UPDATE learning_paths SET current_module = %s, progress = %s, difficulty_level = %s, last_updated = NOW() WHERE user_id = %s AND subject = %s",
                (current_module, 0.0, 0.3 + (proficiency_level * 0.15), user_id, subject)
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return modules
        except mysql.connector.Error as err:
            st.error(f"Learning Path Error: {err}")
            conn.close()
            return []
    return []

# Get subject modules based on proficiency level
def get_subject_modules(subject, proficiency_level):
    # Define modules for each subject at different levels
    # Proficiency levels: 0=Beginner, 1=Elementary, 2=Intermediate, 3=Advanced, 4=Expert
    base_difficulty = 0.3 + (proficiency_level * 0.15)  # Scale difficulty based on proficiency
    
    module_library = {
        'Mathematics': [
            {
                'title': 'Number Systems and Operations',
                'description': 'Understanding different number systems and basic operations.',
                'difficulty': base_difficulty,
                'topics': ['Integers', 'Decimals', 'Fractions', 'Order of Operations']
            },
            {
                'title': 'Algebraic Expressions',
                'description': 'Working with variables and algebraic expressions.',
                'difficulty': base_difficulty + 0.1,
                'topics': ['Variables', 'Simplification', 'Factoring', 'Equations']
            },
            {
                'title': 'Geometry Fundamentals',
                'description': 'Basic geometric shapes and their properties.',
                'difficulty': base_difficulty + 0.2,
                'topics': ['Angles', 'Polygons', 'Circles', 'Area and Volume']
            },
            {
                'title': 'Data Analysis and Statistics',
                'description': 'Methods for analyzing and interpreting data.',
                'difficulty': base_difficulty + 0.3,
                'topics': ['Mean, Median, Mode', 'Probability', 'Graphs', 'Standard Deviation']
            },
            {
                'title': 'Advanced Functions',
                'description': 'Working with complex functions and their applications.',
                'difficulty': base_difficulty + 0.4,
                'topics': ['Polynomials', 'Exponential', 'Logarithmic', 'Trigonometric']
            }
        ],
        'Physics': [
            {
                'title': 'Forces and Motion',
                'description': 'Understanding Newton\'s laws and motion concepts.',
                'difficulty': base_difficulty,
                'topics': ['Newton\'s Laws', 'Kinematics', 'Momentum', 'Gravity']
            },
            {
                'title': 'Energy and Work',
                'description': 'Concepts of energy, work, and power.',
                'difficulty': base_difficulty + 0.1,
                'topics': ['Potential Energy', 'Kinetic Energy', 'Work', 'Conservation of Energy']
            },
            {
                'title': 'Electricity and Magnetism',
                'description': 'Fundamentals of electricity and magnetism.',
                'difficulty': base_difficulty + 0.2,
                'topics': ['Electric Fields', 'Circuits', 'Magnetic Fields', 'Electromagnetic Induction']
            },
            {
                'title': 'Waves and Optics',
                'description': 'Properties of waves and principles of optics.',
                'difficulty': base_difficulty + 0.3,
                'topics': ['Wave Properties', 'Sound', 'Light', 'Reflection and Refraction']
            },
            {
                'title': 'Modern Physics',
                'description': 'Introduction to quantum mechanics and relativity.',
                'difficulty': base_difficulty + 0.4,
                'topics': ['Quantum Mechanics', 'Special Relativity', 'Atomic Structure', 'Nuclear Physics']
            }
        ],
        'Biology': [
            {
                'title': 'Cell Structure and Function',
                'description': 'Understanding the basic units of life.',
                'difficulty': base_difficulty,
                'topics': ['Cell Organelles', 'Cell Membrane', 'Cell Division', 'Cell Transport']
            },
            {
                'title': 'Genetics and Heredity',
                'description': 'How traits are passed from parents to offspring.',
                'difficulty': base_difficulty + 0.1,
                'topics': ['DNA Structure', 'Genes', 'Inheritance Patterns', 'Genetic Disorders']
            },
            {
                'title': 'Human Body Systems',
                'description': 'Structure and function of major body systems.',
                'difficulty': base_difficulty + 0.2,
                'topics': ['Respiratory', 'Circulatory', 'Digestive', 'Nervous']
            },
            {
                'title': 'Ecology and Environment',
                'description': 'Interactions between organisms and their environment.',
                'difficulty': base_difficulty + 0.3,
                'topics': ['Ecosystems', 'Biodiversity', 'Food Webs', 'Environmental Issues']
            },
            {
                'title': 'Evolution and Adaptation',
                'description': 'How species change over time through natural selection.',
                'difficulty': base_difficulty + 0.4,
                'topics': ['Natural Selection', 'Adaptation', 'Speciation', 'Human Evolution']
            }
        ],
        'Chemistry': [
            {
                'title': 'Atoms and Elements',
                'description': 'The building blocks of matter.',
                'difficulty': base_difficulty,
                'topics': ['Atomic Structure', 'Periodic Table', 'Isotopes', 'Electron Configuration']
            },
            {
                'title': 'Chemical Bonds and Compounds',
                'description': 'How atoms join together to form compounds.',
                'difficulty': base_difficulty + 0.1,
                'topics': ['Ionic Bonds', 'Covalent Bonds', 'Molecular Geometry', 'Intermolecular Forces']
            },
            {
                'title': 'Chemical Reactions',
                'description': 'Understanding different types of chemical reactions.',
                'difficulty': base_difficulty + 0.2,
                'topics': ['Balancing Equations', 'Reaction Types', 'Equilibrium', 'Reaction Rates']
            },
            {
                'title': 'Acids, Bases, and Solutions',
                'description': 'Properties and behaviors of acids and bases.',
                'difficulty': base_difficulty + 0.3,
                'topics': ['pH Scale', 'Neutralization', 'Buffers', 'Concentration']
            },
            {
                'title': 'Organic Chemistry',
                'description': 'Chemistry of carbon-based compounds.',
                'difficulty': base_difficulty + 0.4,
                'topics': ['Hydrocarbons', 'Functional Groups', 'Polymers', 'Biochemistry']
            }
        ],
        'Computer Science': [
            {
                'title': 'Introduction to Programming',
                'description': 'Basic programming concepts and logic.',
                'difficulty': base_difficulty,
                'topics': ['Variables', 'Control Structures', 'Functions', 'Basic Algorithms']
            },
            {
                'title': 'Data Structures',
                'description': 'Ways to organize and store data.',
                'difficulty': base_difficulty + 0.1,
                'topics': ['Arrays', 'Lists', 'Stacks and Queues', 'Trees and Graphs']
            },
            {
                'title': 'Algorithms and Problem Solving',
                'description': 'Techniques for solving computational problems.',
                'difficulty': base_difficulty + 0.2,
                'topics': ['Searching', 'Sorting', 'Recursion', 'Algorithm Analysis']
            },
            {
                'title': 'Web Development',
                'description': 'Creating applications for the web.',
                'difficulty': base_difficulty + 0.3,
                'topics': ['HTML/CSS', 'JavaScript', 'APIs', 'Responsive Design']
            },
            {
                'title': 'Advanced Computing Concepts',
                'description': 'More complex topics in computer science.',
                'difficulty': base_difficulty + 0.4,
                'topics': ['Databases', 'Machine Learning', 'Cybersecurity', 'Computer Networks']
            }
        ]
    }
    
    # Return appropriate modules based on subject and adjust difficulty
    if subject in module_library:
        # Sort by difficulty to ensure proper progression
        modules = sorted(module_library[subject], key=lambda x: x['difficulty'])
        
        # For higher proficiency levels, skip some basic modules
        if proficiency_level > 2:
            return modules[proficiency_level-2:]
        
        return modules
    
    return []

# Get remedial modules for specific gaps
def get_remedial_modules(subject, topic):
    remedial_modules = {
        'Mathematics': {
            'Algebra': {
                'title': 'Algebra Foundations',
                'description': 'Strengthening algebraic skills and understanding.',
                'difficulty': 0.4,
                'topics': ['Variables', 'Expressions', 'Equations', 'Problem Solving']
            },
            'Geometry': {
                'title': 'Geometry Essentials',
                'description': 'Reinforcing spatial reasoning and geometric concepts.',
                'difficulty': 0.4,
                'topics': ['Shapes', 'Angles', 'Measurements', 'Coordinate Geometry']
            },
            'Calculus': {
                'title': 'Pre-Calculus Review',
                'description': 'Building foundations needed for calculus.',
                'difficulty': 0.6,
                'topics': ['Functions', 'Limits', 'Rates of Change', 'Graphical Analysis']
            },
            'Statistics': {
                'title': 'Statistics Fundamentals',
                'description': 'Strengthening data analysis skills.',
                'difficulty': 0.5,
                'topics': ['Data Collection', 'Descriptive Statistics', 'Probability', 'Statistical Inference']
            },
            'Trigonometry': {
                'title': 'Trigonometry Foundations',
                'description': 'Reinforcing understanding of trigonometric concepts.',
                'difficulty': 0.5,
                'topics': ['Angles', 'Trigonometric Functions', 'Identities', 'Applications']
            }
        },
        'Physics': {
            'Mechanics': {
                'title': 'Mechanics Fundamentals',
                'description': 'Strengthening understanding of forces and motion.',
                'difficulty': 0.4,
                'topics': ['Forces', 'Motion', 'Energy', 'Momentum']
            },
            'Electricity': {
                'title': 'Electricity Basics',
                'description': 'Reinforcing understanding of electrical concepts.',
                'difficulty': 0.5,
                'topics': ['Charge', 'Current', 'Voltage', 'Circuits']
            },
            'Magnetism': {
                'title': 'Magnetism Essentials',
                'description': 'Building foundations in magnetic phenomena.',
                'difficulty': 0.5,
                'topics': ['Magnetic Fields', 'Electromagnetic Induction', 'Magnetic Materials', 'Applications']
            },
            'Optics': {
                'title': 'Optics Foundations',
                'description': 'Strengthening understanding of light and its properties.',
                'difficulty': 0.5,
                'topics': ['Reflection', 'Refraction', 'Lenses', 'Wave Properties of Light']
            },
            'Thermodynamics': {
                'title': 'Thermodynamics Basics',
                'description': 'Reinforcing concepts of heat and energy.',
                'difficulty': 0.5,
                'topics': ['Temperature', 'Heat Transfer', 'Laws of Thermodynamics', 'Thermal Properties']
            }
        },
        'Biology': {
            'Cells': {
                'title': 'Cell Biology Fundamentals',
                'description': 'Strengthening understanding of cellular structures and functions.',
                'difficulty': 0.4,
                'topics': ['Cell Structure', 'Cell Processes', 'Cell Division', 'Cell Specialization']
            },
            'Genetics': {
                'title': 'Genetics Basics',
                'description': 'Building foundations in genetic principles.',
                'difficulty': 0.5,
                'topics': ['DNA Structure', 'Inheritance', 'Genetic Variation', 'Genetic Disorders']
            },
            'Ecology': {
                'title': 'Ecology Essentials',
                'description': 'Reinforcing understanding of ecological relationships.',
                'difficulty': 0.4,
                'topics': ['Ecosystems', 'Population Dynamics', 'Environmental Factors', 'Human Impact']
            },
            'Evolution': {
                'title': 'Evolution Principles',
                'description': 'Strengthening understanding of evolutionary processes.',
                'difficulty': 0.5,
                'topics': ['Natural Selection', 'Adaptation', 'Evidence for Evolution', 'Speciation']
            },
            'Physiology': {
                'title': 'Physiology Foundations',
                'description': 'Building foundations in body system functions.',
                'difficulty': 0.5,
                'topics': ['Homeostasis', 'Organ Systems', 'Cellular Processes', 'System Integration']
            }
        },
        'Chemistry': {
            'Elements': {
                'title': 'Elements and Atomic Structure',
                'description': 'Reinforcing understanding of atomic components and properties.',
                'difficulty': 0.4,
                'topics': ['Atomic Models', 'Electron Configuration', 'Periodic Trends', 'Isotopes']
            },
            'Compounds': {
                'title': 'Chemical Compounds Basics',
                'description': 'Strengthening understanding of chemical bonding and compounds.',
                'difficulty': 0.5,
                'topics': ['Ionic Compounds', 'Covalent Compounds', 'Molecular Geometry', 'Naming Compounds']
            },
            'Reactions': {
                'title': 'Chemical Reactions Fundamentals',
                'description': 'Building foundations in reaction types and balancing.',
                'difficulty': 0.5,
                'topics': ['Reaction Types', 'Balancing Equations', 'Stoichiometry', 'Reaction Rates']
            },
            'Organic': {
                'title': 'Organic Chemistry Essentials',
                'description': 'Reinforcing understanding of carbon-based compounds.',
                'difficulty': 0.6,
                'topics': ['Hydrocarbons', 'Functional Groups', 'Organic Reactions', 'Biochemistry']
            },
            'Physical': {
                'title': 'Physical Chemistry Basics',
                'description': 'Strengthening understanding of energy and matter interactions.',
                'difficulty': 0.6,
                'topics': ['Thermodynamics', 'Kinetics', 'Equilibrium', 'States of Matter']
            }
        },
        'Computer Science': {
            'Programming': {
                'title': 'Programming Fundamentals',
                'description': 'Reinforcing basic programming concepts and practices.',
                'difficulty': 0.4,
                'topics': ['Syntax', 'Variables', 'Control Structures', 'Functions']
            },
            'Data Structures': {
                'title': 'Data Structures Basics',
                'description': 'Building foundations in organizing and storing data.',
                'difficulty': 0.5,
                'topics': ['Arrays', 'Lists', 'Stacks', 'Queues']
            },
            'Algorithms': {
                'title': 'Algorithms Essentials',
                'description': 'Strengthening problem-solving and algorithm design skills.',
                'difficulty': 0.5,
                'topics': ['Complexity Analysis', 'Searching', 'Sorting', 'Recursion']
            },
            'Databases': {
                'title': 'Database Fundamentals',
                'description': 'Reinforcing understanding of data storage and management.',
                'difficulty': 0.5,
                'topics': ['Data Models', 'SQL', 'Normalization', 'Database Design']
            },
            'Web Development': {
                'title': 'Web Development Basics',
                'description': 'Building foundations in web technologies.',
                'difficulty': 0.5,
                'topics': ['HTML', 'CSS', 'JavaScript', 'Web Architecture']
            }
        }
    }
    
    if subject in remedial_modules and topic in remedial_modules[subject]:
        return [remedial_modules[subject][topic]]
    
    return []

# Function to record assessment results
def record_assessment(user_id, subject, score, difficulty):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO assessments (user_id, subject, score, difficulty, completed_at) VALUES (%s, %s, %s, %s, NOW())",
                (user_id, subject, score, difficulty)
            )
            
            # Check if user deserves a badge for this assessment
            if score >= 90:
                badge_name = f"{subject} Expert"
                badge_description = f"Achieved excellence in {subject} with a score of {score}%"
                cursor.execute(
                    "INSERT INTO badges (user_id, badge_name, badge_description) VALUES (%s, %s, %s)",
                    (user_id, badge_name, badge_description)
                )
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except mysql.connector.Error as err:
            st.error(f"Assessment Recording Error: {err}")
            conn.close()
            return False
    return False

# Function to get user progress and metrics
def get_user_progress(user_id):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Get learning paths
            cursor.execute(
                "SELECT subject, current_module, progress, difficulty_level FROM learning_paths WHERE user_id = %s",
                (user_id,)
            )
            learning_paths = cursor.fetchall()
            
            # Get assessment history
            cursor.execute(
                "SELECT subject, score, difficulty, completed_at FROM assessments WHERE user_id = %s ORDER BY completed_at DESC",
                (user_id,)
            )
            assessments = cursor.fetchall()
            
            # Get badges
            cursor.execute(
                "SELECT badge_name, badge_description, earned_at FROM badges WHERE user_id = %s ORDER BY earned_at DESC",
                (user_id,)
            )
            badges = cursor.fetchall()
            
            # Get participation in challenges
            cursor.execute("""
                SELECT c.title, c.subject, c.difficulty_level, cp.completed, cp.joined_at
                FROM challenge_participants cp
                JOIN collaborative_challenges c ON cp.challenge_id = c.id
                WHERE cp.user_id = %s
                ORDER BY cp.joined_at DESC
            """, (user_id,))
            challenges = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {
                'learning_paths': learning_paths,
                'assessments': assessments,
                'badges': badges,
                'challenges': challenges
            }
        except mysql.connector.Error as err:
            st.error(f"Progress Retrieval Error: {err}")
            conn.close()
            return None
    return None

# Function to join a collaborative challenge
def join_challenge(user_id, challenge_id):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            
            # Check if user is already participating
            cursor.execute(
                "SELECT id FROM challenge_participants WHERE challenge_id = %s AND user_id = %s",
                (challenge_id, user_id)
            )
            if cursor.fetchone():
                cursor.close()
                conn.close()
                return "Already joined this challenge"
            
            # Join the challenge
            cursor.execute(
                "INSERT INTO challenge_participants (challenge_id, user_id) VALUES (%s, %s)",
                (challenge_id, user_id)
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            return "Successfully joined challenge"
        except mysql.connector.Error as err:
            st.error(f"Challenge Joining Error: {err}")
            conn.close()
            return f"Error: {err}"
    return "Database connection error"

# Function to get collaborative challenges
def get_challenges():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT c.id, c.title, c.description, c.subject, c.difficulty_level,
                       COUNT(cp.id) as participant_count
                FROM collaborative_challenges c
                LEFT JOIN challenge_participants cp ON c.id = cp.challenge_id
                GROUP BY c.id
                ORDER BY c.created_at DESC
            """)
            challenges = cursor.fetchall()
            cursor.close()
            conn.close()
            return challenges
        except mysql.connector.Error as err:
            st.error(f"Challenge Retrieval Error: {err}")
            conn.close()
            return []
    return []

# Chatbot functionality (placeholder for Gemini integration)
def stem_chatbot(query, api_key=None):
    # This is a placeholder. In a real implementation, you would use the Gemini API
    if not api_key:
        return "Please provide your Gemini API key in the settings to activate the AI tutor."
    
    try:
        # Simulate API call
        # In a real implementation:
        # import google.generativeai as genai
        # genai.configure(api_key=api_key)
        # model = genai.GenerativeModel('gemini-pro')
        # response = model.generate_content(query)
        # return response.text
        
        # For hackathon demo, simulate response
        time.sleep(1.5)  # Simulate API call delay
        
        # Simple responses for demo
        if "concept" in query.lower() or "explain" in query.lower():
            return f"I'd be happy to explain this concept! Let's break down {query.replace('explain', '').replace('concept', '').strip()}. In STEM education, understanding fundamental principles is key. This concept involves examining the relationships between variables and how they interact within a system. Would you like me to provide some examples or practice problems to help reinforce your understanding?"
        
        elif "problem" in query.lower() or "solve" in query.lower() or "help" in query.lower():
            return "To solve this problem, let's approach it step by step. First, identify what you're trying to find and what information you have. Then, determine which formulas or principles apply to this situation. Let's work through it systematically, and remember to check your units throughout the calculation. Would you like me to show you a similar example first?"
        
        elif "test" in query.lower() or "exam" in query.lower() or "prepare" in query.lower():
            return "Preparing for a test requires a strategic approach! I recommend creating a study schedule, reviewing your notes and textbook materials, solving practice problems, and perhaps forming a study group with classmates. Focus on understanding concepts rather than memorizing, and don't forget to take breaks to keep your mind fresh. What specific subject is your test covering?"
        
        else:
            return "That's an interesting question about STEM! Remember that building a strong foundation in science, technology, engineering, and mathematics involves both theoretical understanding and practical application. Would you like me to help you explore this topic in more depth, suggest some resources, or perhaps work through some practice problems together?"
    
    except Exception as e:
        return f"Error with chatbot: {str(e)}"

# Main application structure
def main():
    # Initialize database on first run
    initialize_database()
    
    # Check if user is logged in
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    
    # Sidebar navigation (only if logged in)
    if st.session_state.user_id:
        with st.sidebar:
            st.image("https://via.placeholder.com/150x150.png?text=STEM+NAV", width=150)
            st.title("STEM Navigator")
            
            # Navigation
            st.subheader("Navigation")
            if st.button("📊 Dashboard", use_container_width=True):
                st.session_state.page = 'dashboard'
            if st.button("📝 Diagnostic Tests", use_container_width=True):
                st.session_state.page = 'diagnostic'
            if st.button("🧭 Learning Paths", use_container_width=True):
                st.session_state.page = 'learning_paths'
            if st.button("👥 Collaborative Challenges", use_container_width=True):
                st.session_state.page = 'challenges'
            if st.button("🤖 AI Tutor", use_container_width=True):
                st.session_state.page = 'chatbot'
            if st.button("⚙️ Settings", use_container_width=True):
                st.session_state.page = 'settings'
            
            # Logout
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.user_id = None
                st.session_state.page = 'login'
                st.experimental_rerun()
    
    # Main content area
    if st.session_state.page == 'login':
        render_login_page()
    elif st.session_state.page == 'register':
        render_register_page()
    elif st.session_state.page == 'dashboard':
        render_dashboard()
    elif st.session_state.page == 'diagnostic':
        render_diagnostic_page()
    elif st.session_state.page == 'learning_paths':
        render_learning_paths()
    elif st.session_state.page == 'challenges':
        render_challenges()
    elif st.session_state.page == 'chatbot':
        render_chatbot()
    elif st.session_state.page == 'settings':
        render_settings()

# Render login page
def render_login_page():
    st.title("STEM Navigator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h2>AI-Powered STEM Education Platform</h2>
            <p>Welcome to STEM Navigator, the intelligent platform that personalizes your learning experience in Science, Technology, Engineering, and Mathematics.</p>
            <ul>
                <li>🧠 <span class="highlight">Smart Learning Gap Detection</span> using Random Forest ML</li>
                <li>📊 <span class="highlight">Personalized Learning Paths</span> that adapt to your progress</li>
                <li>👥 <span class="highlight">Collaborative Challenges</span> to learn with peers</li>
                <li>🤖 <span class="highlight">AI Tutor</span> powered by advanced language models</li>
                <li>📱 <span class="highlight">Progress Tracking</span> with interactive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", use_container_width=True):
            user = login_user(username, password)
            if user:
                st.session_state.user_id = user['id']
                st.session_state.user_name = user['full_name']
                st.session_state.user_grade = user['grade']
                st.session_state.user_learning_style = user['learning_style']
                

                st.session_state.page = 'dashboard'
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
        
        if st.button("Create New Account", use_container_width=True):
            st.session_state.page = 'register'
            st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Render register page
def render_register_page():
    st.title("Create New Account")
    
    with st.form("register_form"):
        username = st.text_input("Username (must be unique)")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        full_name = st.text_input("Full Name")
        grade = st.selectbox("Grade Level", range(6, 13), index=2)  # Grades 6-12
        learning_style = st.selectbox(
            "Preferred Learning Style",
            ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"]
        )
        
        if st.form_submit_button("Register"):
            if password != confirm_password:
                st.error("Passwords do not match!")
            elif not all([username, password, full_name]):
                st.error("Please fill in all required fields")
            else:
                if register_user(username, password, full_name, grade, learning_style):
                    st.success("Registration successful! Please login.")
                    st.session_state.page = 'login'
                    st.experimental_rerun()
                else:
                    st.error("Registration failed. Username may already exist.")
    
    if st.button("Back to Login"):
        st.session_state.page = 'login'
        st.experimental_rerun()

# Render dashboard
def render_dashboard():
    st.title(f"Welcome back, {st.session_state.user_name}!")
    
    # Get user progress data
    progress_data = get_user_progress(st.session_state.user_id)
    
    if not progress_data:
        st.warning("Could not load your progress data. Please try again later.")
        return
    
    # Top metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">6</div>
            <div class="metric-label">STEM Subjects</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        completed_assessments = len(progress_data['assessments'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{completed_assessments}</div>
            <div class="metric-label">Completed Assessments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        earned_badges = len(progress_data['badges'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{earned_badges}</div>
            <div class="metric-label">Earned Badges</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress charts
    st.subheader("📈 Your Learning Progress")
    
    # Subject progress chart
    subjects = [lp['subject'] for lp in progress_data['learning_paths']]
    progress_values = [lp['progress'] * 100 for lp in progress_data['learning_paths']]
    
    fig = px.bar(
        x=subjects,
        y=progress_values,
        labels={'x': 'Subject', 'y': 'Progress (%)'},
        color=subjects,
        title="Subject Progress"
    )
    fig.update_layout(plot_bgcolor='#1E1E1E', paper_bgcolor='#1E1E1E', font_color='white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Assessment history chart
    if progress_data['assessments']:
        assessment_df = pd.DataFrame(progress_data['assessments'])
        assessment_df['completed_at'] = pd.to_datetime(assessment_df['completed_at'])
        assessment_df = assessment_df.sort_values('completed_at')
        
        fig = px.line(
            assessment_df,
            x='completed_at',
            y='score',
            color='subject',
            markers=True,
            labels={'completed_at': 'Date', 'score': 'Score (%)'},
            title="Assessment Scores Over Time"
        )
        fig.update_layout(plot_bgcolor='#1E1E1E', paper_bgcolor='#1E1E1E', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent badges
    if progress_data['badges']:
        st.subheader("🏆 Recently Earned Badges")
        cols = st.columns(3)
        for i, badge in enumerate(progress_data['badges'][:3]):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="card">
                    <h4>{badge['badge_name']}</h4>
                    <p>{badge['badge_description']}</p>
                    <small>Earned on {badge['earned_at'].strftime('%b %d, %Y')}</small>
                </div>
                """, unsafe_allow_html=True)

# Render diagnostic page
def render_diagnostic_page():
    st.title("📝 Diagnostic Tests")
    st.markdown("""
    <div class="card">
        <p>Take diagnostic tests to assess your current knowledge and identify learning gaps. 
        Our AI system will analyze your results and create a personalized learning path.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Subject selection
    subject = st.selectbox(
        "Select a STEM Subject",
        ["Mathematics", "Physics", "Biology", "Chemistry", "Computer Science"]
    )
    
    if st.button("Start Diagnostic Test", use_container_width=True):
        st.session_state.diagnostic_subject = subject
        st.session_state.diagnostic_started = True
        st.session_state.diagnostic_score = None
        st.session_state.diagnostic_user_data = None
        st.experimental_rerun()
    
    # Handle diagnostic test if started
    if st.session_state.get('diagnostic_started', False):
        subject = st.session_state.diagnostic_subject
        grade = st.session_state.user_grade
        learning_style = st.session_state.user_learning_style
        
        if st.session_state.get('diagnostic_score') is None:
            score, user_data = run_diagnostic_test(subject, grade, learning_style)
            if score is not None:
                st.session_state.diagnostic_score = score
                st.session_state.diagnostic_user_data = user_data
                st.experimental_rerun()
        else:
            # Show results and create learning path
            score = st.session_state.diagnostic_score
            user_data = st.session_state.diagnostic_user_data
            
            st.success(f"Diagnostic test completed! Your score: {score:.1f}%")
            
            # Predict proficiency level and learning gaps
            proficiency_level, gaps = predict_proficiency_and_gaps(user_data)
            
            # Display results
            st.subheader("📊 Results Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="card">
                    <h4>Proficiency Level</h4>
                """, unsafe_allow_html=True)
                
                levels = ["Beginner", "Elementary", "Intermediate", "Advanced", "Expert"]
                st.metric("Your Level", levels[proficiency_level])
                
                # Progress bar for proficiency
                progress_value = (proficiency_level + 1) / 5
                st.progress(progress_value)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>Recommended Actions</h4>
                """, unsafe_allow_html=True)
                
                if proficiency_level < 2:
                    st.write("Focus on building foundational knowledge with our beginner modules.")
                elif proficiency_level < 4:
                    st.write("Strengthen your understanding with intermediate content and practice problems.")
                else:
                    st.write("Challenge yourself with advanced topics and real-world applications.")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Show learning gaps if any
            if gaps:
                st.subheader("🔍 Identified Learning Gaps")
                st.write("These are areas where you might need additional focus:")
                
                for gap, score in gaps[:3]:  # Show top 3 gaps
                    st.markdown(f"""
                    <div class="card">
                        <h4>{gap}</h4>
                        <div class="progress-container">
                            <small>Mastery Level: {score*100:.1f}%</small>
                            <progress value="{score}" max="1"></progress>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Create learning path button
            if st.button("Create Personalized Learning Path", use_container_width=True):
                modules = create_learning_path(
                    st.session_state.user_id,
                    subject,
                    proficiency_level,
                    gaps
                )
                
                if modules:
                    st.session_state.learning_path_modules = modules
                    st.session_state.learning_path_subject = subject
                    st.success("Learning path created successfully!")
                    st.session_state.page = 'learning_paths'
                    st.experimental_rerun()
                else:
                    st.error("Failed to create learning path. Please try again.")

# Render learning paths
def render_learning_paths():
    st.title("🧭 Your Learning Paths")
    
    # Get current learning paths from database
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT subject, current_module, progress, difficulty_level FROM learning_paths WHERE user_id = %s",
                (st.session_state.user_id,)
            )
            learning_paths = cursor.fetchall()
            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            st.error(f"Database Error: {err}")
            learning_paths = []
    else:
        learning_paths = []
    
    # Show current learning paths
    if learning_paths:
        st.subheader("Your Active Learning Paths")
        
        for path in learning_paths:
            with st.expander(f"{path['subject']} - {path['current_module']}"):
                st.markdown(f"""
                <div class="card">
                    <div class="progress-container">
                        <small>Progress: {path['progress']*100:.1f}%</small>
                        <progress value="{path['progress']}" max="1"></progress>
                    </div>
                    <p>Current Module: <strong>{path['current_module']}</strong></p>
                    <p>Difficulty Level: {path['difficulty_level']*100:.1f}/100</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show module content if available
                if st.session_state.get('learning_path_modules') and st.session_state.get('learning_path_subject') == path['subject']:
                    st.subheader("Recommended Modules")
                    
                    for module in st.session_state.learning_path_modules:
                        st.markdown(f"""
                        <div class="card">
                            <h4>{module['title']}</h4>
                            <p>{module['description']}</p>
                            <small>Topics: {', '.join(module['topics'])}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Simulate module completion
                        if st.button(f"Complete {module['title']}", key=f"complete_{module['title']}"):
                            # Update progress in database
                            conn = get_db_connection()
                            if conn:
                                try:
                                    cursor = conn.cursor()
                                    cursor.execute(
                                        "UPDATE learning_paths SET progress = LEAST(progress + 0.2, 1.0), last_updated = NOW() WHERE user_id = %s AND subject = %s",
                                        (st.session_state.user_id, path['subject'])
                                    )
                                    conn.commit()
                                    cursor.close()
                                    conn.close()
                                    st.success("Progress updated! Great work!")
                                    st.experimental_rerun()
                                except mysql.connector.Error as err:
                                    st.error(f"Database Error: {err}")
    else:
        st.warning("You don't have any active learning paths yet. Take a diagnostic test to get started!")

# Render challenges page
def render_challenges():
    st.title("👥 Collaborative Challenges")
    st.markdown("""
    <div class="card">
        <p>Join collaborative challenges to work with peers on real-world STEM problems. 
        These challenges help you apply your knowledge in practical scenarios while learning teamwork.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get available challenges
    challenges = get_challenges()
    
    if challenges:
        st.subheader("Available Challenges")
        
        for challenge in challenges:
            st.markdown(f"""
            <div class="card">
                <h3>{challenge['title']}</h3>
                <p>{challenge['description']}</p>
                <div style="display: flex; justify-content: space-between;">
                    <span>Subject: <strong>{challenge['subject']}</strong></span>
                    <span>Difficulty: <strong>{challenge['difficulty_level']*100:.0f}/100</strong></span>
                    <span>Participants: <strong>{challenge['participant_count']}</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Join challenge button
            if st.button(f"Join {challenge['title']}", key=f"join_{challenge['id']}"):
                result = join_challenge(st.session_state.user_id, challenge['id'])
                st.toast(result)
                st.experimental_rerun()
    
    # Show user's current challenges
    progress_data = get_user_progress(st.session_state.user_id)
    if progress_data and progress_data['challenges']:
        st.subheader("Your Challenges")
        
        for challenge in progress_data['challenges']:
            status = "✅ Completed" if challenge['completed'] else "⏳ In Progress"
            st.markdown(f"""
            <div class="card">
                <h4>{challenge['title']}</h4>
                <p>Subject: {challenge['subject']} | Difficulty: {challenge['difficulty_level']*100:.0f}/100</p>
                <p>Status: {status}</p>
                <small>Joined on {challenge['joined_at'].strftime('%b %d, %Y')}</small>
            </div>
            """, unsafe_allow_html=True)

# Render chatbot page
def render_chatbot():
    st.title("🤖 STEM AI Tutor")
    st.markdown("""
    <div class="card">
        <p>Ask our AI tutor any questions about STEM subjects. The tutor can explain concepts, 
        help solve problems, and provide personalized learning recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message">
                <strong>AI Tutor:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    query = st.text_input("Ask your STEM question:", key="chat_input")
    
    if st.button("Send", key="send_message"):
        if query:
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": query})
            
            # Get AI response (simulated for demo)
            with st.spinner("AI Tutor is thinking..."):
                response = stem_chatbot(query)
                
                # Add AI response to chat history
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                
                # Rerun to show new messages
                st.experimental_rerun()

# Render settings page
def render_settings():
    st.title("⚙️ Settings")
    
    with st.form("settings_form"):
        st.subheader("Account Information")
        
        # Get current user info
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(
                    "SELECT full_name, grade, learning_style FROM users WHERE id = %s",
                    (st.session_state.user_id,)
                )
                user_info = cursor.fetchone()
                cursor.close()
                conn.close()
            except mysql.connector.Error as err:
                st.error(f"Database Error: {err}")
                user_info = None
        else:
            user_info = None
        
        if user_info:
            new_name = st.text_input("Full Name", value=user_info['full_name'])
            new_grade = st.selectbox(
                "Grade Level", 
                range(6, 13), 
                index=user_info['grade']-6  # Convert grade to index
            )
            new_style = st.selectbox(
                "Learning Style",
                ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"],
                index=["Visual", "Auditory", "Reading/Writing", "Kinesthetic"].index(user_info['learning_style'])
            )
            
            if st.form_submit_button("Update Profile"):
                conn = get_db_connection()
                if conn:
                    try:
                        cursor = conn.cursor()
                        cursor.execute(
                            "UPDATE users SET full_name = %s, grade = %s, learning_style = %s WHERE id = %s",
                            (new_name, new_grade, new_style, st.session_state.user_id)
                        )
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        # Update session state
                        st.session_state.user_name = new_name
                        st.session_state.user_grade = new_grade
                        st.session_state.user_learning_style = new_style
                        
                        st.success("Profile updated successfully!")
                        st.experimental_rerun()
                    except mysql.connector.Error as err:
                        st.error(f"Database Error: {err}")
        
        st.subheader("AI Tutor Settings")
        api_key = st.text_input("Gemini API Key (optional)", type="password")
        
        if st.form_submit_button("Save Settings"):
            st.session_state.gemini_api_key = api_key
            st.success("Settings saved!")

# Run the app
if __name__ == "__main__":
    main()