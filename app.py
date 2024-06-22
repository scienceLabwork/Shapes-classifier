import streamlit as st
from streamlit_drawable_canvas import st_canvas
from deep_model import image_to_numpy, make_pred
import cv2
import pandas as pd

st.set_page_config(
    page_title="Shape Recognition",
    page_icon=":pencil:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

col1, col2 = st.columns(2)

with col1:
    st.title('Drawable Canvas')
    st.write('Use the canvas drawing tool to draw a Square, Circle, and Triangle.')
    canvas_result = st_canvas(
        fill_color="rgb(255, 165, 0)",  # Fixed fill color with some opacity
        stroke_width=10,
        stroke_color="rgb(0, 0, 0)",
        background_color="#fff",
        height=540,
        width=500,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img_data = cv2.imencode('.png', canvas_result.image_data)[1].tobytes()
        st.download_button(
            label="Download canvas image",
            data=img_data,
            file_name='canvas_image.png',
            mime='image/png'
        )

with col2:
    st.title('Prediction')
    st.write('Click the button below to predict the shape.')
    button = st.button("Predict", key="predict", type="primary", use_container_width=True)

    if button:
        with st.spinner("Predicting..."):
            if canvas_result.image_data is not None:  
                arr = image_to_numpy(canvas_result.image_data)
                output,pred = make_pred(arr)
                st.success(f"Prediction: {output}")
                dictx = {"Apple": pred[0][0], "Pineapple": pred[0][1], "Banana": pred[0][2]}
                df = pd.DataFrame(dictx, index=[0], columns=["Apple", "Pineapple", "Banana"])
                st.bar_chart(df, use_container_width=True)
            else:
                st.error("Please draw a fruit first.")
    else:
        st.warning("Please draw a fruit first.")
    
    st.write('---')
    st.write('**Note:**')
    st.write('1. Draw a shape on the canvas.')
    st.write('2. Click the "Predict" button to predict the shape.')
    st.write('3. The model can predict Square, Circle, and Triangle.')
    st.write('4. This model is developed by [Rudra Shah](https://www.linkedin.com/in/rudra-shah-b044781b4/).')