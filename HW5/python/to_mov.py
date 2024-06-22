import cv2
import os

folder_path = '../results/lk/landing/' # input path
output_name = '../results/landing_lk.mp4' # output path

images = [img for img in os.listdir(folder_path) if img.endswith(".jpg")]
images = sorted(images)
frame = cv2.imread(os.path.join(folder_path, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(folder_path, image)))

cv2.destroyAllWindows()
video.release()
