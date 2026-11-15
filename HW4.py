import streamlit as st
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(layout="wide", page_title="MY DIP")
st.write("## Homework 2 전제익")
st.write(":dog:")
st.sidebar.write("## my sidebar")

def process_image(upload):
    image = ImageOps.exif_transpose(Image.open(upload))

    col1.write("Input")
    col1.image(image)

    output = image.convert("L")
    col2.write("Output 1: grayscale")
    col2.image(output)

    output = ImageOps.equalize(output)
    col3.write("Output 2: histeq")
    col3.image(output)

    output = np.max(np.array(output)) - np.array(output)
    col4.write("Output 3: your own processing")
    col4.image(output)

col1, col2, col3, col4 = st.columns(4)

my_upload = st.sidebar.file_uploader(
    "Upload", type=["png", "jpg", "jpeg"]
)

if my_upload is not None:
    process_image(my_upload)
else:
    st.info("왼쪽 Upload에서 사진을 업로드해 주세요.")
