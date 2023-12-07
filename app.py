import os
import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

# Set Streamlit page configuration, including title and icon
st.set_page_config(
    page_title='Neural style transfer',
    page_icon='🖼️',
    layout='wide'
)

# Load custom CSS style for the app
@st.cache_data
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Load the pre-trained style transfer model from TensorFlow Hub
hub_handle_path = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'

@st.cache_resource
def load_modules(hub_path):
    return hub.load(hub_path)

hub_module = load_modules(hub_handle_path)

# Define the function to perform neural style transfer
def stylize_image(content_image, style_image):
    # Convert images to NumPy arrays
    content_image = np.array(content_image)
    style_image = np.array(style_image)

    # Convert the images to float32 in the range [0, 1]
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

    style_image = tf.image.resize(style_image, (259, 259))

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))

    stylized_image = outputs[0].numpy()

    return stylized_image

# Set up the Streamlit app interface
st.markdown("""
    <div style="text-align:left;">
        <h1 style="color:#000000;font-style: sans-serif;font-size: 3em;margin-bottom: 30px;">
            🎨🖌️ Artistic Neural Style Transfer
        </h1>
    </div>
""", unsafe_allow_html=True)

# Create a sidebar with image selection options
with st.sidebar:
    st.info("""  🎈:white[Upload your image alongside famous works of art to apply 
    their unique styles to your own creations!] ✨️ """)

    # Content image options
    content_image_options = [
        "content_1.jpg",
        "content_2.jpg",
        "Upload Image"
    ]
    content_image_option = st.selectbox(":white[Choose Content Image:]", content_image_options)

    # Style image options
    style_image_options = [
        "style_1.jpg",
        "style_2.jpg",
        "style_3.jpg",
        "style_4.jpg",
        "style_5.jpg",
        "style_6.jpg",
        "style_7.jpg",
        "style_8.jpg",
        "style_9.jpg",
        "style_10.jpg",
        "Upload Image"
    ]
    style_image_option = st.selectbox(":white[Choose Style Image:]", style_image_options)

    # Upload file
    content_image_upload = None
    style_image_upload = None

    if content_image_option == "Upload Image":
        content_image_upload = st.file_uploader(label=":blue[**Upload Custom Content Image**]:", type=['jpg', 'jpeg'])

    if style_image_option == "Upload Image":
        style_image_upload = st.file_uploader(label=":blue[**Upload Custom Style Image**]:", type=['jpg', 'jpeg'])

    create_style = st.button(":white[Submit]", type="primary", use_container_width=True)

# Create columns to display the images
col1, col2 = st.columns(2)

content_image = None
style_image = None
h = 0
w = 0

# Get the content image
if content_image_option != "Upload Image":
    content_image_path = os.path.join("contents", content_image_option)
    content_image = Image.open(content_image_path)
    h = content_image.height
    w = content_image.width

# Get the style image
if style_image_option != "Upload Image":
    style_image_path = os.path.join("styles", style_image_option)
    style_image = Image.open(style_image_path)

# Check if content_image_upload is not None and has valid dimensions
if content_image_upload:
    content_image = Image.open(content_image_upload)
    h = content_image.height
    w = content_image.width

# Check if style_image_upload is not None and has valid dimensions
if style_image_upload:
    style_image = Image.open(style_image_upload)
    if h > 0 and w > 0:
        style_image = style_image.resize((w, h))

# Check for valid dimensions before attempting resizing
if h > 0 and w > 0:
    if content_image is not None:
        content_image = content_image.resize((w, h))

    if style_image is not None:
        style_image = style_image.resize((w, h))


# Display the content and style images
if content_image is not None:
    with col1:
        st.image(content_image, caption='Content Image', use_column_width=True)

if style_image is not None:
    with col2:
        st.image(style_image, caption='Style Image', use_column_width=True)


# Perform neural style transfer and display the result
if create_style and content_image is not None and style_image is not None:

    st.markdown("""
        <div style="text-align:center;">
            <h1 style="color:#000000;
                font-style: sans-serif;
                font-size: 1.5em;
                margin-top:30px;
                margin-bottom: 30px;">
                🎈 Stylized Image 🎉🎉
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Styling process
    with st.spinner('⚙️:rainbow[Stylizing Image ...]'):
        try:
            output = stylize_image(content_image, style_image)

            # Convert the NumPy array to a PIL image
            output_pil = Image.fromarray((output * 255).astype('uint8')[0])

            # Determine image dimensions for display
            # display_width = 800
            # display_height = int(output_pil.height * (display_width / output_pil.width))

            # Resize the image for display
            #img_display = output_pil.resize((display_width, display_height), Image.ANTIALIAS)
            # display_width = 3350
            # display_height = 2550 7200

            display_width = 9000
            display_height = 7800
            img_display = output_pil.resize((display_width, display_height), Image.LANCZOS)
            # Display the stylized image
            st.image(img_display, caption='Stylized Image', use_column_width=True)

            # Save the stylized image as bytes for download
            output_bytes = io.BytesIO()

            # output_pil.save(output_bytes, format='jpeg')
            img_display.save(output_bytes, format='jpeg', dpi=(600, 600))

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Add a download button for the stylized image
    if output_bytes:
        st.download_button(':white[**Download Stylized Image**]', data=output_bytes, type='primary',
                           file_name='stylized_image.jpeg',
                           mime="image/jpeg",
                           key="download-button",
                           use_container_width=False)
