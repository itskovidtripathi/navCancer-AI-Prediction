import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import os
from datetime import datetime
import config
from database import Database
import re

# Initialize database
db = Database()

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Detection System",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stApp {
        background-color: #f5f5f5;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .metric-box {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .user-form {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)
    
    model.load_state_dict(torch.load('models/lung_cancer_detector.pth'))
    model = model.to(config.DEVICE)
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess the image for model input"""
    transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])
    
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def predict_image(model, input_tensor):
    """Make prediction for a single image"""
    with torch.no_grad():
        output = model(input_tensor.to(config.DEVICE))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    return predicted_class, confidence, probabilities

def plot_probabilities(probabilities, class_names):
    """Create probability plot"""
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(class_names))
    
    # Plot horizontal bars
    bars = ax.barh(y_pos, probabilities.cpu().numpy())
    
    # Customize colors based on probabilities
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.RdYlGn(probabilities[i].item()))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Probabilities')
    
    # Add percentage labels on bars
    for i, v in enumerate(probabilities):
        ax.text(v.item(), i, f'{v.item():.1%}', va='center')
    
    plt.tight_layout()
    return fig

def generate_detailed_report(predicted_class, confidence, probabilities, class_names):
    """Generate a detailed medical report"""
    diagnosis = class_names[predicted_class]
    medical_info = config.MEDICAL_INFO[diagnosis]
    
    report = f"""
LUNG CANCER DETECTION REPORT
{'=' * 50}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DIAGNOSIS
--------
Primary Diagnosis: {diagnosis}
Confidence Level: {confidence:.1%}

DETAILED ANALYSIS
---------------
Description: {medical_info['description']}

Probability Distribution:
{'-' * 30}"""
    
    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        report += f"\n{name}: {prob.item():.1%}"

    if diagnosis != 'Normal':
        report += f"""

CHARACTERISTICS
-------------
The following characteristics are typical for {diagnosis}:
"""
        for char in medical_info['characteristics']:
            report += f"- {char}\n"

    report += f"""

RISK FACTORS
-----------
Key risk factors to consider:
"""
    for risk in medical_info['risk_factors']:
        report += f"- {risk}\n"

    report += f"""

RECOMMENDATIONS
-------------
Based on the analysis, the following actions are recommended:
"""
    for rec in medical_info['recommendations']:
        report += f"- {rec}\n"

    report += f"""

FOLLOW-UP
--------
{medical_info['follow_up']}

IMPORTANT NOTES
-------------
1. This analysis is based on AI interpretation and should be confirmed by a qualified healthcare professional.
2. The confidence level indicates the AI model's certainty but should not be used as the sole diagnostic criterion.
3. Additional imaging studies or tests may be required for definitive diagnosis.
4. Early detection and treatment are crucial for better outcomes.

DISCLAIMER
---------
This report is generated by an AI system and is intended to assist healthcare professionals. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions regarding a medical condition.

Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')}
"""
    return report

def validate_email(email):
    """Validate email format"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Validate phone number format"""
    pattern = r'^\+?1?\d{9,15}$'
    return re.match(pattern, phone) is not None

def save_prediction(image, prediction_info):
    """Save the prediction results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "predictions"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save image
    image_path = os.path.join(save_dir, f"prediction_{timestamp}.jpg")
    image.save(image_path)
    
    # Save prediction info
    info_path = os.path.join(save_dir, f"prediction_{timestamp}.txt")
    with open(info_path, "w") as f:
        f.write(prediction_info)
    
    return image_path, info_path

def main():
    # Header
    st.title("🫁 Lung Cancer Detection System")
    st.markdown("---")
    
    # Initialize session state for user registration
    if 'user_registered' not in st.session_state:
        st.session_state.user_registered = False
        st.session_state.user_id = None
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This system uses deep learning to detect lung cancer from chest X-ray images.
        
        **Supported Types:**
        - Normal
        - Squamous Cell Carcinoma
        - Adenocarcinoma
        
        **Model Details:**
        - Architecture: ResNet18
        - Input Size: 224x224
        - Accuracy: 97.03%
        """)
        
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Fill in your information
        2. Upload a chest X-ray image
        3. Wait for the analysis
        4. Review the detailed results
        5. Download the report if needed
        """)
    
    # User registration form
    if not st.session_state.user_registered:
        st.markdown("### User Information")
        with st.form("user_form"):
            name = st.text_input("Full Name*")
            email = st.text_input("Email Address*")
            phone = st.text_input("Phone Number*")
            address = st.text_area("Address*")
            
            if st.form_submit_button("Submit"):
                if not all([name, email, phone, address]):
                    st.error("All fields are required!")
                elif not validate_email(email):
                    st.error("Please enter a valid email address!")
                elif not validate_phone(phone):
                    st.error("Please enter a valid phone number!")
                else:
                    user_id = db.add_user(name, email, phone, address)
                    if user_id:
                        st.session_state.user_registered = True
                        st.session_state.user_id = user_id
                        st.success("Information submitted successfully!")
                        st.rerun()
    
    # Main content - only show after user registration
    if st.session_state.user_registered:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Upload X-ray Image")
            uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
                
                # Make prediction
                model = load_model()
                input_tensor = preprocess_image(image)
                predicted_class, confidence, probabilities = predict_image(model, input_tensor)
                
                # Get class names
                class_names = list(config.CLASS_NAMES.values())
                
                with col2:
                    st.markdown("### Analysis Results")
                    
                    # Display prediction
                    prediction_box = f"""
                    <div class='prediction-box'>
                        <h3>Diagnosis</h3>
                        <p style='font-size: 24px; color: {'red' if predicted_class != 0 else 'green'};'>
                            {class_names[predicted_class]}
                        </p>
                        <p style='font-size: 18px;'>Confidence: {confidence:.1%}</p>
                    </div>
                    """
                    st.markdown(prediction_box, unsafe_allow_html=True)
                    
                    # Display probability plot
                    st.markdown("### Probability Distribution")
                    prob_fig = plot_probabilities(probabilities, class_names)
                    st.pyplot(prob_fig)
                    
                    # Generate detailed report
                    detailed_report = generate_detailed_report(
                        predicted_class, 
                        confidence, 
                        probabilities, 
                        class_names
                    )
                    
                    # Save results
                    image_path, info_path = save_prediction(image, detailed_report)
                    
                    # Save prediction to database
                    db.add_prediction(
                        st.session_state.user_id,
                        image_path,
                        class_names[predicted_class],
                        confidence,
                        info_path
                    )
                    
                    # Display key findings
                    st.markdown("### Key Findings")
                    with st.expander("View Detailed Analysis", expanded=True):
                        diagnosis = class_names[predicted_class]
                        medical_info = config.MEDICAL_INFO[diagnosis]
                        
                        st.markdown(f"**Description:**")
                        st.write(medical_info['description'])
                        
                        if diagnosis != 'Normal':
                            st.markdown(f"**Characteristics:**")
                            for char in medical_info['characteristics']:
                                st.write(f"- {char}")
                        
                        st.markdown(f"**Recommendations:**")
                        for rec in medical_info['recommendations']:
                            st.write(f"- {rec}")
                    
                    # Download buttons
                    st.markdown("### Download Results")
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        with open(info_path, 'r') as f:
                            st.download_button(
                                label="📄 Download Detailed Report",
                                data=f.read(),
                                file_name=f"lung_cancer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                    with col_btn2:
                        with open(image_path, 'rb') as f:
                            st.download_button(
                                label="🖼️ Download Image",
                                data=f.read(),
                                file_name=f"xray_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                mime="image/jpeg"
                            )

if __name__ == "__main__":
    main() 