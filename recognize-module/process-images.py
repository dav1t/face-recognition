import os
import face_recognition
from PIL import Image

PRE_PROCESSED_IMAGES_DIR = "../Data/pre-processed-images"
PROCESSED_IMAGES_DIR = "../Data/processed-images/"


def processImages():
    for img_path in os.listdir(PRE_PROCESSED_IMAGES_DIR):
        full_path = os.path.join(PRE_PROCESSED_IMAGES_DIR, img_path)

        image = face_recognition.load_image_file(full_path)
        face_bounding_boxes = face_recognition.face_locations(image)

        for i in range(len(face_bounding_boxes)):
            top, right, bottom, left = face_bounding_boxes[i]
            faceImage = image[top:bottom, left:right]

            final = Image.fromarray(faceImage)
            final.save(PROCESSED_IMAGES_DIR + "img%s.png" % (str(i)), "PNG")


processImages()