import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_files")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "runs")

def load_image(path, target_size=331):
    img = Image.open(path).convert('RGB')
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img = np.array(img)
    img = tf.keras.applications.nasnet.preprocess_input(img)
    return tf.convert_to_tensor(img)

base_model = tf.keras.applications.NASNetLarge(
    include_top=False,
    weights='imagenet',
    input_shape=(331, 331, 3)
)

target_layers = [layer.output for layer in base_model.layers[-30:] if 'normal' in layer.name or 'relu' in layer.name]
dream_model = tf.keras.Model(inputs=base_model.input, outputs=target_layers)

def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    return tf.reduce_sum([tf.reduce_mean(act) for act in layer_activations])

@tf.function
def deepdream_step(model, image, step_size):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = calc_loss(image, model)

    gradients = tape.gradient(loss, image)
    gradients /= (tf.math.reduce_std(gradients) + 1e-8)
    image = image + gradients * step_size
    image = tf.clip_by_value(image, -1, 1)
    return loss, image

def deprocess(image):
    image = 255 * (image + 1.0) / 2.0
    return tf.cast(image, tf.uint8)

def run_deep_dream_multiscale(model, image, steps=800, step_size=0.025, octaves=[1.0, 1.3, 1.6, 1.9]):
    base_shape = tf.shape(image)[:-1]
    image = tf.identity(image)

    run_name = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_name, exist_ok=True)

    for octave, scale in enumerate(octaves):
        new_size = tf.cast(tf.cast(base_shape, tf.float32) * scale, tf.int32)
        image = tf.image.resize(image, new_size)

        for step in range(steps):
            loss, image = deepdream_step(model, image, step_size)

            print(f"Octave {octave+1}/{len(octaves)} | Step {step+1}/{steps} | Loss: {loss.numpy():.4f}")

            if step % 150 == 0 or step == steps - 1:
                img_show = np.array(deprocess(image))
                filename = os.path.join(run_name, f"dream_octave{octave+1}_step{step}.png")
                plt.imsave(filename, img_show)

    return deprocess(image), run_name

if __name__ == "__main__":
    input_path = os.path.join(DATA_DIR, "sample.jpg")
    dream_image = load_image(input_path)

    final_image, folder = run_deep_dream_multiscale(
        model=dream_model,
        image=dream_image,
        steps=1000,
        step_size=0.03,
        octaves=[1.0, 1.4, 1.8, 2.2]
    )

    Image.fromarray(final_image.numpy()).save(os.path.join(folder, "dream_final.jpg"))
    print(f"DeepDream complete. Final image saved in folder: {folder}")
