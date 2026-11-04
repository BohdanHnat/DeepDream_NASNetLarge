import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "outputs", "progress_frames", "run_1")
VIDEO_DIR = os.path.join(BASE_DIR, "outputs", "videos")

os.makedirs(VIDEO_DIR, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(os.path.join(VIDEO_DIR, 'deepdreamvideo.avi'), fourcc, 10.0, (910, 605))

i = 0
while True:
    if os.path.isfile(os.path.join(FRAMES_DIR, f'{i}.png')):
        pass
    else:
        dream_length = i
        break
    i += 1

print(dream_length)

for i in range(dream_length):
    img_path = os.path.join(FRAMES_DIR, f'{i}.png')

    print(img_path)

    frame = cv2.imread(img_path)
    out.write(frame)

out.release()
