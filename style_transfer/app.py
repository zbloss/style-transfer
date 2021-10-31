import streamlit as st
from datetime import datetime
from PIL import Image
from train_model import train_model

st.write("# Scripted Artistry")

main = st.container()
col1, col2 = main.columns(2)

col1.write(
    f"""## Your Image

Upload the image you want to apply a filter onto below!

"""
)
user_image = col1.file_uploader("Upload image", type=["png", "jpeg", "jpg"])

col2.write(
    f"""## Style Image

Upload the image you want us to extract the stylings from and apply onto your Image!

"""
)
style_image = col2.file_uploader("Upload image")

if user_image is not None and style_image is not None:
    with st.spinner(text="Style Transfer in progress..."):
        user_image = Image.open(user_image)
        style_image = Image.open(style_image)
        styled_image = train_model(
            user_image = user_image,
            style_image = style_image
        )
    st.image(styled_image)
