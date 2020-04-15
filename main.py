from imutils.video import VideoStream
import imutils
import numpy as np
import time
import cv2

# help function
def detect_predict_age(frame, face_model, age_model, minConf=0.5):
    AGE_BINS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    results = []

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_model.setInput(blob)
    detections = face_model.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > minConf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x_start, y_start, x_end, y_end = box.astype('int')

            face = frame[y_start:y_end, x_start:x_end]

            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), 
                    (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            age_model.setInput(face_blob)
            predictions = age_model.forward()
            index = predictions[0].argmax()
            age = AGE_BINS[index]
            age_conf = predictions[0][index]

            d = {
                "loc": (x_start, y_start, x_end, y_end),
                "age": (age, age_conf)
            }
            results.append(d)

    return results


print('[INFO] Loading face model...')
faceNet = cv2.dnn.readNet('face_model/deploy.prototxt', 'face_model/res10_300x300_ssd_iter_140000.caffemodel')

print('[INFO] Loading age model...')
ageNet = cv2.dnn.readNet('age_model/age_deploy.prototxt', 'age_model/age_net.caffemodel')

print('[INFO] Loading camera...')
vs = VideoStream(src=0).start()
time.sleep(2)

run = True
while run:
    frame = vs.read()
    # frame = imutils.resize(frame, width=400)
    results = detect_predict_age(frame, faceNet, ageNet, minConf=0.3)

    for r in results:
        text = f'{r["age"][0]}: {round(r["age"][1] * 100, 2)}'
        x_start, y_start, x_end, y_end = r["loc"]
        y = y_start - 10 if y_start - 10 > 10 else y_start + 10
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        cv2.putText(frame, text, (y_start, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        run = False
        break
cv2.destroyAllWindows()
vs.stop()
