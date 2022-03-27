
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img


def predict(model_path='ferNet', full_output=False, crop_face=False, img_path=None, img=None):
    """
    Receives an image containing a face and returns the predominant emotion assosiated with the face.
    :param model_path: Path to the trained model file.
    :param full_output: If false, will return only one string with the predicted emotion, if true will return the prediction vector as a list.
    :param crop_face: If true we use the MTCNN algorithm to detect and crop the face in the image, otherwise assume that the picture is already cropped.
    :param img_path: Path to the image to be predicted, if None, img parameter can't also be None.
    :param img: Image file to be predicted, if None, img_path can't also be None.
    :return: By default returns a string with the predicted emotion, if full_output is True, returns a list of the prediction values for each emotion.
    """
    img_size = 48
    if model_path is not None and model_path[-3:] != '.h5':
        model_path += '.h5'
    fernet = load_model(model_path)

    if img is None:
        img = load_img(img_path, target_size=(
            img_size, img_size), color_mode="grayscale")
        img = np.array(img)
    else:
        img = np.array(img)

    label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
                  3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    if crop_face:
        detector = MTCNN()
        result = detector.detect_faces(img)
        if result != []:
            for person in result:
                bounding_box = person['box']
                ROI = img[bounding_box[1]:bounding_box[1] + bounding_box[3],
                          bounding_box[0]:bounding_box[0]+bounding_box[2]]
                ROI = cv2.resize(ROI, (img_size, img_size))
                ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

                # makes image shape (1,48,48)
                ROI = np.expand_dims(ROI, axis=0)
                ROI = ROI.reshape(1, img_size, img_size, 1)
                ROI = ROI * 1/255

                result = fernet.predict(ROI)
    else:
        img = np.expand_dims(img, axis=0)
        img = img.reshape(1, img_size, img_size, 1)

        result = fernet.predict(img)

    result = list(result[0])
    for i in range(len(result)):
        result[i] = f'{result[i]:.2f}'

    if full_output:
        return result
    else:
        return label_dict[np.argmax(result)]


if __name__ == "__main__":
    # If you run this file, it will simply call the function with the default parameters.
    # You can also import it as a function and call the train function yourself.
    print(predict(img_path="database/test/surprise/PrivateTest_54842414.jpg"))
