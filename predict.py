import argparse
import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image
from workspace_utils import process_image
import numpy as np

parser = argparse.ArgumentParser(description='Predict flower type')

parser.add_argument("path", action="store")
parser.add_argument("model", action="store")
parser.add_argument("--top_k", action="store", type=int, default=3)
parser.add_argument("--category_names", action="store")
results = parser.parse_args()

class_names = None
try:
    with open(f"{results.category_names}", "r") as f:
        class_names = json.load(f)
        condition = True
except:
    condition = False

    
model = tf.keras.models.load_model(results.model,
                                          custom_objects={'KerasLayer':hub.KerasLayer})

image = Image.open(results.path)
image = np.asarray(image)
image = process_image(image)
image = np.expand_dims(image, axis=0)

predictions = model.predict(image)

classes = predictions[0].argsort()[-results.top_k:][::-1]
top_preds = predictions[:, classes][0]

classes = classes + 1

if condition:
    preds_class_names = [class_names[str(x)] for x in classes]
    print(top_preds, preds_class_names)
else:
    print(top_preds, classes)



