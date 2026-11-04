# DeepDream — Multi-Scale Feature Amplification with NASNetLarge

An implementation of DeepDream: the system takes a single input image and progressively rewrites it so that the features a pre-trained **NASNetLarge** network already detects in it become stronger and more pronounced. Faint suggestions of eyes, feathers, arches and scales get amplified into explicit, repeated patterns across the whole canvas.

The method is the inverse of ordinary training. The network is frozen and never updated; the pixels of the image are the only thing that changes. More unusually, the objective is **maximised rather than minimised** — there is no target to converge on, only the network's own activations to drive upward. What appears in the output is the feature vocabulary the network learned from ImageNet, made visible.

Built with Python, TensorFlow and Keras.

## Functionality

Running `main.py` loads the input from `samples/`, runs the multi-scale dream loop, and writes progress frames and the final image to a timestamped folder under `outputs/runs/`.

| Step | What happens |
|---|---|
| 1 | The input image is loaded, resized to **331 × 331** with Lanczos resampling — NASNetLarge's native input size — and preprocessed by `nasnet.preprocess_input` into the `[-1, 1]` range |
| 2 | **NASNetLarge** is loaded with ImageNet weights and `include_top=False`, discarding the classifier head and leaving the convolutional feature extractor |
| 3 | Target layers are chosen automatically: of the network's last 30 layers, those whose names contain `normal` or `relu`. The model is rebuilt to emit all of them at once |
| 4 | **Loss** = the sum of the mean activation of every target layer. This value is *maximised*: a higher mean activation means the image contains more of whatever those layers respond to |
| 5 | `tf.GradientTape` takes the gradient of that loss with respect to the image. The gradient is divided by its own standard deviation, so the step size behaves consistently no matter how large the raw activations are |
| 6 | The image is updated by `image + gradients * step_size` — gradient **ascent**, not descent — then clipped back to `[-1, 1]` |
| 7 | **Octave loop:** the whole process repeats at four increasing scales (`1.0, 1.4, 1.8, 2.2`). Each octave resizes the current image upward and dreams again, so patterns form at both fine and coarse scales instead of one texture size only |
| 8 | Every 150 steps, and on the final step of each octave, a progress frame is written to the run folder. The finished image is saved as `dream_final.jpg` |

**Why gradient ascent changes the character of the output.** In style transfer the loss measures distance from a target and is minimised. DeepDream has no target. The quantity being raised is the network's own response, so the optimisation drives the image toward whatever the selected filters were trained to detect. The recurring eye-like and animal-like motifs in the output are those learned features made explicit, not arbitrary noise.

`video.py` is a separate utility that stitches a numbered sequence of progress frames into an AVI with OpenCV at 10 fps.

## Results produced so far

Each pair below is the first and last saved frame of a single run. The left image is the input essentially unmodified; the right image is the same canvas after the full sequence of ascent steps.

### 1 — Ballpoint sketch

| Input | After amplification |
|---|---|
| ![Sketch input](samples/sketch_source.png) | ![Sketch output](samples/sketch_dream.png) |

Full frame sequence: `outputs/progress_frames/run_2/`

### 2 — Painted portrait

| Input | After amplification |
|---|---|
| ![Portrait input](samples/portrait_source.png) | ![Portrait output](samples/portrait_dream.png) |

Full frame sequence: `outputs/runs/1/`

### What the two pairs show together

Running the same procedure on two very different inputs makes the central property of the method visible: **amplification is driven by what is already in the image, so the result depends on the input's existing structure rather than being applied evenly.**

- **The sketch is consumed; the portrait survives.** The ballpoint drawing is dense line work — high-frequency edges across almost the entire surface. Every region offers the ascent something to build on, so pattern growth is close to uniform and the original subject is largely overwritten. The painting has broad smooth passages in the face alongside heavily textured background and clothing; growth concentrates in the textured areas while the facial structure stays legible.
- **New motifs are not in either input.** Eye-like spots, scale and feather textures and curved ridges appear in both outputs. Neither source contains them — they are what the selected NASNetLarge layers respond to most strongly.
- **Saturation rises with step count.** The loss has no upper bound and the only constraint is the clip to `[-1, 1]`, so colour intensity keeps climbing the longer the loop runs. This follows from the objective rather than from a tuning choice.

## Testing

**Prerequisites:** Python 3.12, and roughly 350 MB of free space for the NASNetLarge ImageNet weights, which Keras downloads and caches automatically on the first run.

**Steps:**

1. Create and activate a virtual environment, then `pip install -r requirements.txt`.
2. Place the input image in `samples/`. The images shown above are included, so the project runs unchanged.
3. Run `python main.py`. Octave, step and loss are printed continuously; progress frames appear in a new timestamped folder under `outputs/runs/` as the run proceeds.
4. To build a video from a numbered frame sequence, run `python video.py`. It reads `outputs/progress_frames/run_1/` and writes `outputs/videos/deepdreamvideo.avi`.

> **Run time.** Every step is a full forward and backward pass through NASNetLarge, and the cost of each octave grows with its scale factor. A complete four-octave run at the default 1000 steps per octave takes hours on CPU. Lower `steps` or drop the larger octaves for a quick test.

To change the balance of the result, adjust the arguments to `run_deep_dream_multiscale`: `steps` controls how far the amplification is pushed, `step_size` how aggressively each update moves, and `octaves` the range of pattern scales that form.

## Repository layout

```
main.py                     Image loading, loss, ascent step, octave loop and output
video.py                    Stitches a numbered frame sequence into an AVI
requirements.txt            Pinned dependencies
samples/
  sketch_source.png         Input and output of the ballpoint sketch run
  sketch_dream.png
  portrait_source.png       Input and output of the painted portrait run
  portrait_dream.png
outputs/
  runs/
    1/                      Multi-scale run — octave 1 complete, octave 2 partial
    2/                      Multi-scale run — octave 1 complete
  progress_frames/
    run_1/                  Numbered frame sequence from an earlier single-scale run
    run_2/                  Numbered frame sequence from an earlier single-scale run
    source_image.jpg        Input used for those earlier runs
  videos/
    deepdreamvideo.avi      Assembled by video.py from progress_frames/run_1
```

## Tech stack

Python 3.12 · TensorFlow 2.16 · Keras 3.1 · NASNetLarge (ImageNet weights) · NumPy · Matplotlib · Pillow · OpenCV

## Scope and limitations

* **No run has completed all four octaves.** `outputs/runs/1` stops during the second octave and `outputs/runs/2` after the first, so no `dream_final.jpg` exists yet. The cost per octave rises with its scale factor, and a full run was not practical on the hardware used.
* **The current script forces a square 331 × 331 input.** `load_image` resizes every image to NASNetLarge's native input size regardless of its shape, so non-square inputs are distorted and output resolution is set by the octave multipliers rather than chosen. The higher-resolution results shown above predate this constraint and were produced by the earlier single-scale path; reproducing them at that size requires relaxing the fixed resize.
* **`video.py` only matches one frame set.** It expects sequentially numbered files at a hard-coded 910 × 605, which fits `outputs/progress_frames/run_1` alone. The other sequences use different filenames and resolutions, so stitching those into a video needs the script's naming and size assumptions changed.
