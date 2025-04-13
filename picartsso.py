import streamlit as st
from PIL import Image
import numpy as np
import os

from merger import merger, initialize_model

@st.cache_resource(show_spinner="Mounting Model... Generate button will appear once Done!")  # Ensures model loads only once
def load_model():
    # Dummy call to trigger model loading (e.g., warm-up)
    initialize_model()

def main():
    st.set_page_config(
        page_title="Picartso",
        page_icon="🎨",
        layout="wide"
    )

    st.markdown("""
        <style>
        /* Main background */
        .stApp {
            background-color: rgba(20, 20, 20, 0.9);
            background-image: url('https://media.sketchfab.com/models/f0c293932c3343e5a840e5367486d802/thumbnails/b166862dde4e427eb433e85449fde3bf/349af1877ccc43e1bd30f3ed040b7dd7.jpeg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-blend-mode: overlay;
            font-family: 'Segoe UI', sans-serif;
        }
        .stApp header {
            background-color: #2D1E40;
            color: white;
        }
        
        /* Adjust the hamburger menu and other header elements */
        .stApp header .eYKPeo {
            color: white !important;
        }
        
        /* Optional: Adjust other header elements to ensure proper contrast */
        .stApp header button {
            color: white;
        }
        
        /* For Streamlit ≥ 1.22.0 where some class names might have changed */
        .stApp [data-testid="stHeader"] {
            background-color: #262730;
        }
        /* Titles */
        .linked-title { /* Add a class to the linked title for styling */
            color: #e600ff !important;
            text-shadow: 0 0 5px #e600ff, 0 0 15px #7a00cc;
            transition: all 0.3s ease-in-out; /* Add transition for smooth effect */
            text-decoration: none; /* Remove default link underline */
        }
        /* Title Hover Glow */
        .linked-title:hover {
            transform: scale(1.03); /* Slightly scale up on hover */
            text-shadow: 0 0 10px #e600ff, 0 0 25px #7a00cc; /* Enhance text shadow on hover */
        }
        /* Subheadings */

        h2, h3, .stSubheader {
            color: #36f9f6 !important; /* Vibrant purple inspired by Arcane */
            text-shadow: 0 0 3px #36f9f6, 0 0 10px #6320a9;
            transition: all 0.3s ease; /* Smooth transition for hover effect */
        }

        /* Hover effects for subheadings */
        h2:hover, h3:hover, .stSubheader:hover {
            color: #7afffd !important; /* Lighter purple on hover */
            text-shadow: 0 0 8px #7afffd, 0 0 15px #8a24eb, 0 0 25px #6320a9;
            transform: scale(1.03); /* Slightly scale up on hover */
        }
        /* Body text */
        p, label, .stMarkdown, div {
            color: #d3b2ff !important;
        }
        /* Buttons */
        .stButton>button {
            background-color: transparent;
            color: #00f0ff;
            border: 2px solid #00f0ff;
            font-size: 16px;
            padding: 10px 15px;
            border-radius: 8px;
            transition: all 0.3s ease-in-out;
            text-shadow: 0 0 3px #00f0ff;
            box-shadow: 0 0 10px #00f0ff40;
        }
        /* Button Hover */
        .stButton>button:hover {
            background-color: #00ffff20;
            transform: scale(1.05);
            box-shadow: 0 0 20px #00ffff;
            border-color: #00ffff;
            color: #00ffff;
        }
        /* Upload blocks */
        .uploadBlock {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 0 15px #7a00cc;
            backdrop-filter: blur(3px);
        }
        </style>
    """, unsafe_allow_html=True)


    github_repo_url = "https://github.com/Ahmad-Waseem/picartsso"
    st.markdown(f"""
    <a href="{github_repo_url}" target="_blank" style="text-decoration: none;">
        <h1 class="linked-title">🖼️ Picartsso</h1>
    </a>
    """, unsafe_allow_html=True)
    st.subheader("Upload Shirt Image & Art image to transfer Styles!", anchor=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='uploadBlock'>", unsafe_allow_html=True)
        st.subheader("Shirt Image", help="add SHIRT image TO BE STYLED", anchor=False)
        image1 = st.file_uploader("Choose Shirt image", type=["png", "jpg", "jpeg"], key="img1")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='uploadBlock'>", unsafe_allow_html=True)
        st.subheader("Art Image", help="add the ART image as STYLE REFERENCE", anchor=False)
        image2 = st.file_uploader("Choose an Art image", type=["png", "jpg", "jpeg"], key="img2")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Trigger model download on page load
    load_model()
    if st.button("✨ Generate Magic ✨"):
        result = None;
        if image1 and image2:
            img1 = Image.open(image1).convert('RGB').resize((256, 256), Image.LANCZOS)
            img2 = Image.open(image2).convert('RGB').resize((256, 256), Image.LANCZOS)
            with st.spinner("Just there...!"):
                result = generate_placeholder_image(img1, img2)

            if result is not None:
                st.markdown("### Generated Result")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(img1, caption="Shirt Image")

                with col2:
                    st.image(img2, caption="Art Image")

                with col3:
                    st.image(result, caption="Generated Image")
        else:
            st.warning("Please upload both images to proceed.")

def generate_placeholder_image(img1, img2):
    try:
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        styled_image = merger(img1_array, img2_array)
        return styled_image
    except Exception as e:
        st.error(f"Error in image generation: {e}")
        return None

if __name__ == "__main__":
    main()