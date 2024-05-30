import streamlit as st
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json
import base64
import io

# Set the title and description
st.title("🔍 YOLOv8n Object Detection")
st.markdown("""
    Upload an image and let the YOLOv8n model detect objects in it.
    This model can identify a variety of objects and draw bounding boxes around them.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    vps_model_client = model.ModelClient()
    model_id = "mdl-xd03onbvnj3u2"
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    input_data = img_str
    if st.button('🔍 Detect Objects'):
        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=img_str)
            # Decode the base64 image from the response
            print(api_response)
            output_base64 = api_response["image"]
            output_image_data = base64.b64decode(output_base64)
            result_image = Image.open(io.BytesIO(output_image_data))
            
            st.image(result_image, caption='Detected Objects', use_column_width=True)
        except UnauthorizedException:
            st.error("Unauthorized exception")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f5f5;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle, .stMarkdown, .stButton, .stImage {
            text-align: center;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stImage > img {
            border: 2px solid #4CAF50;
            border-radius: 8px;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)
