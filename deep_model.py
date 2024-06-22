import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("shapes.h5")

def image_to_numpy(arr):
    image = Image.fromarray(arr)
    image = image.convert("RGB")
    image = image.resize((240, 270))
    image = np.array(image)
    image = image.reshape(-1, 270, 240, 3)
    return image

def make_pred(arr):
    labels = ["Quadrilateral","Circle","Triangle"]
    emojis = ["🟥","🔵","🔺"]
    pred = model.predict(arr)
    return str(emojis[pred.argmax()] + " " + labels[pred.argmax()] + " " + str(round(pred.max() * 100, 2)) + "%"), pred