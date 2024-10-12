import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from torchvision import models
import torch.nn as nn
import cv2
import numpy as np
import gdown
import os
import urllib.request

# Set the device (either GPU if available, or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load a custom ResNet model
def get_model(num_classes):
    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

# Load the trained model from Google Drive
@st.cache(allow_output_mutation=True)
def load_trained_model(num_classes):
    # Google Drive file ID
    file_id = "12qBYyOVYjoNK1WOOyrOt2GEGDyj9yK4N"
    url = f"https://drive.google.com/uc?id={file_id}"
    model_path = "eye_diagram_classifier_v1.pth"

    # Download the model file
    gdown.download(url, model_path, quiet=False)

    # Load the model
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Function to classify an eye diagram image
def classify_eye_diagram(image, model, classes):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        return predicted_class, probabilities

# Check Google Drive link accessibility
def check_drive_accessibility(file_url):
    try:
        response = urllib.request.urlopen(file_url)
        return response.status == 200
    except:
        return False

# GPT API request function (updated for all classification results)
def get_gpt_explanation(classification_result, user_api_key=None):
    # Use the user's API key if provided, otherwise use the default from Streamlit secrets
    if user_api_key is None and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = user_api_key

    if not api_key:
        return "Error: No API key found. Please provide your OpenAI API key."
    
    prompt = f"""
    In high-speed signal Eye Diagram analysis, a '{classification_result}' problem has been detected.
    Please provide the following information:
    1. What are the main causes of {classification_result} in high-speed signals?
    2. What areas should be checked when {classification_result} is detected?
    3. What are the general countermeasures for {classification_result} issues in high-speed signals?
    Please provide a detailed explanation suitable for display on a web interface.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=data
        )
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error {response.status_code}: Could not retrieve data from GPT API."
    except Exception as e:
        return f"Error: {str(e)} - Could not contact GPT API."

# Display functions
def display_in_yellow_box(text):
    st.markdown(f"""
        <div style="background-color:#FFFF99;padding:10px;border-radius:5px">
        {text}
        </div>
        """, unsafe_allow_html=True)

def display_in_pink_box(text):
    st.markdown(f"""
        <div style="background-color:#FFCCCC;padding:10px;border-radius:5px">
        {text}
        </div>
        """, unsafe_allow_html=True)

# Image preprocessing function
def preprocess_eye_diagram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image, x, y, w, h
    else:
        return image, 0, 0, image.shape[1], image.shape[0]

# Mask evaluation function
def mask_evaluation(image, mask_width_ns, mask_height_mv, start_time_ns, end_time_ns, voltage_range_mv):
    preprocessed_image, x, y, w, h = preprocess_eye_diagram(image)

    image_height, image_width = preprocessed_image.shape[:2]
    total_time_ns = end_time_ns - start_time_ns
    time_per_pixel = total_time_ns / image_width
    voltage_per_pixel = voltage_range_mv / image_height

    mask_center_x_ns = (start_time_ns + end_time_ns) / 2
    mask_center_y_mv = voltage_range_mv / 2

    mask_center_x_px = int((mask_center_x_ns - start_time_ns) / time_per_pixel)
    mask_center_y_px = int((voltage_range_mv - mask_center_y_mv) / voltage_per_pixel)

    mask_width_px = int(mask_width_ns / time_per_pixel)
    mask_height_px = int(mask_height_mv / voltage_per_pixel)

    points = np.array([
        [mask_center_x_px, mask_center_y_px - mask_height_px // 2],
        [mask_center_x_px - mask_width_px // 2, mask_center_y_px],
        [mask_center_x_px, mask_center_y_px + mask_height_px // 2],
        [mask_center_x_px + mask_width_px // 2, mask_center_y_px]
    ])

    mask_layer = np.zeros((image_height, image_width, 4), dtype=np.uint8)
    cv2.fillPoly(mask_layer, [points], (0, 0, 255, 76))  # Semi-transparent red
    cv2.polylines(mask_layer, [points], isClosed=True, color=(255, 255, 255, 255), thickness=3)
    cv2.polylines(mask_layer, [points], isClosed=True, color=(0, 0, 0, 255), thickness=1)

    image_with_mask = preprocessed_image.copy()
    alpha_mask = mask_layer[:, :, 3] / 255.0
    for c in range(3):
        image_with_mask[:, :, c] = (1 - alpha_mask) * preprocessed_image[:, :, c] + alpha_mask * mask_layer[:, :, c]

    gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2GRAY)
    _, thresh_signal = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    mask_area = cv2.fillPoly(np.zeros_like(thresh_signal), [points], 255)
    intrusion_area = cv2.bitwise_and(thresh_signal, mask_area)
    intrusion = np.any(intrusion_area == 255)

    return image_with_mask, "NG" if intrusion else "OK", (x, y, w, h)

# Streamlit Interface
st.title('Eye Diagram Classifier and Evaluator v5')

# URL to check Google Drive access
drive_url = "https://drive.google.com/file/d/1HTvXXWXsrXceqb4w2ZDiy4B22p3rdpIj/view?usp=drive_link"
drive_accessible = check_drive_accessibility(drive_url)

# Check if user has access to the drive link
if drive_accessible:
    st.info("hb*** 님 API key입력을 생략합니다.")
    user_api_key = None
else:
    st.info("GPT-4 API key입력이 필요합니다.")
    user_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Number of classes in your model
num_classes = 3
class_names = ['crosstalk', 'Loss and ISI', 'reflection']

# Mask parameters
st.sidebar.title("Mask Parameters")
mask_width_ns = st.sidebar.number_input("Enter mask width (ns)", min_value=0.0, value=1.0)
mask_height_mv = st.sidebar.number_input("Enter mask height (mV)", min_value=0.0, value=500.0)
start_time_ns = st.sidebar.number_input("Enter start time (ns)", min_value=0.0, value=0.0)
end_time_ns = st.sidebar.number_input("Enter end time (ns)", min_value=0.0, value=2.0)
voltage_range_mv = st.sidebar.number_input("Enter voltage range (mV)", min_value=0.0, value=1000.0)

# File uploader for the eye diagram image
uploaded_image = st.file_uploader("Upload an Eye Diagram Image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Load original image
    original_image = Image.open(uploaded_image).convert('RGB')
    st.image(original_image, caption='Uploaded Eye Diagram', use_column_width=True)

    # Convert image to numpy array
    image_np = np.array(original_image)

    # Perform mask evaluation
    image_with_mask, evaluation_result, (x, y, w, h) = mask_evaluation(image_np, mask_width_ns, mask_height_mv, start_time_ns, end_time_ns, voltage_range_mv)

    # Display the image with mask
    st.image(image_with_mask, caption='Preprocessed Eye Diagram with Mask', use_column_width=True)

    # Step 1: Display Mask Evaluation Result
    st.markdown("<h2>Step 1. Mask Evaluation Result:</h2>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: {'red' if evaluation_result == 'NG' else 'green'};'>**{evaluation_result}**</h2>", unsafe_allow_html=True)

    # If OK, no further action needed
    if evaluation_result == "OK":
        st.write("The eye diagram is acceptable. No further action needed.")
    else:
        # Load the trained model
        model = load_trained_model(num_classes)

        # Prepare original image for classification
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_image = test_transform(original_image).unsqueeze(0).to(device)

        # Classify the uploaded image
        predicted_class, probabilities = classify_eye_diagram(input_image, model, class_names)

        # Step 2: Display the classification result
        st.markdown("<h2>Step 2. Classification Result by CNN:</h2>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: blue;'>{class_names[predicted_class]}</h2>", unsafe_allow_html=True)

        # Display probabilities for each class
        for i, prob in enumerate(probabilities):
            display_in_pink_box(f"{class_names[i]}: {prob:.4f}")

        # Step 3: GPT Analysis
        st.markdown("<h2>Step 3. GPT Analysis:</h2>", unsafe_allow_html=True)
        st.markdown(f"[Click here for detailed GPT analysis of {class_names[predicted_class]}](https://chatgpt.com/g/g-9NESyIPPB-eye-pattern-analyzer)")

        # Fetch GPT analysis
        with st.spinner('Fetching detailed analysis from GPT...'):
            gpt_response = get_gpt_explanation(class_names[predicted_class], user_api_key=user_api_key)

            # Display GPT analysis
            st.markdown("<h3>Detailed Analysis:</h3>", unsafe_allow_html=True)

            # Split GPT response into paragraphs and apply formatting
            paragraphs = gpt_response.split('
')
            formatted_response = ''.join([
                f"<div style='background-color:#FFFF99;padding:10px;border-radius:5px;margin-bottom:10px;'>"
                f"<pre style='background-color:#F0F0F0;padding:10px;border-radius:5px;'>{paragraph.strip()}</pre>"
                f"</div>" for paragraph in paragraphs if paragraph.strip()])

            st.markdown(formatted_response, unsafe_allow_html=True)
