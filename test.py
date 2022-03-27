import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from keras.models import load_model


def test(model_path='ferNet', output_video_path="emotions.avi"):
    """
    Opens your camera, displays a video with the predicted results of the emotions displayed, and saves it as an AVI file.
    :param model_path: Path to the trained model file.
    :param output_video_path : Path to the output video file.
    :return: Nothing.
    """

    img_size = 48
    if model_path is not None and model_path[-3:] != '.h5':
        model_path += '.h5'

    if output_video_path is not None and output_video_path[-4:] != '.avi':
        output_video_path += '.avi'

    fernet = load_model(model_path)
    detector = MTCNN()

    cap = cv2.VideoCapture(0)

    label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
                  3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 4.0, (640, 480))

    while True:
        # Capture frame-by-frame
        __, frame = cap.read()

        # Use MTCNN to detect faces
        result = detector.detect_faces(frame)
        if result != []:
            for person in result:
                bounding_box = person['box']
                keypoints = person['keypoints']
                ROI = frame[bounding_box[1]:bounding_box[1] + bounding_box[3],
                            bounding_box[0]:bounding_box[0]+bounding_box[2]]
                ROI = cv2.resize(ROI, (img_size, img_size))
                ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

                # makes image shape (1,48,48)
                ROI = np.expand_dims(ROI, axis=0)
                ROI = ROI.reshape(1, img_size, img_size, 1)
                ROI = ROI * 1/255

                result = fernet.predict(ROI)
                result = list(result[0])
                result = label_dict[np.argmax(result)]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    frame, result, (bounding_box[0], bounding_box[1] - 15), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.rectangle(frame,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0]+bounding_box[2],
                                  bounding_box[1] + bounding_box[3]),
                              (0, 155, 255),
                              2)

                cv2.circle(frame, (keypoints['left_eye']), 2, (0, 155, 255), 1)
                cv2.circle(
                    frame, (keypoints['right_eye']), 2, (0, 155, 255), 1)
                cv2.circle(frame, (keypoints['nose']), 2, (0, 155, 255), 1)
                cv2.circle(
                    frame, (keypoints['mouth_left']), 2, (0, 155, 255), 1)
                cv2.circle(
                    frame, (keypoints['mouth_right']), 2, (0, 155, 255), 1)

        # display resulting frame, close webcam when the Q key is pressed on the keyboard
        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # when everything's done, release capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # If you run this file, it will simply call the function with the default parameters.
    # You can also import it as a function and call the train function yourself.
    test()
