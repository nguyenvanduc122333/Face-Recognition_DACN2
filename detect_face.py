import os

import pymysql
from imutils import paths
from keras.utils.image_utils import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

import xlwrite


def getProfile(id):
    conn = pymysql.connect(host="test", user="root", password="123456", database="students")
    query = "SELECT * FROM data WHERE roll_no= " + str(id)
    cur = conn.cursor()
    cur.execute(query)

    # biến lưu dữ liệu lấy từ db về
    profile = None
    for row in cur:
        profile = row

    conn.close()
    return profile


# Thiết lập detect_face.py [-v file video]
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# Nạp model dò tìm khuôn mặt
# face detector cascade (tích hợp trong OpenCV)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Nạp model lenet (dò tìm mặt có cười/không cười)
model = load_model("lenet.hdf5")

# Nếu không sử dụng video thì mở Webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)  # Mở Webcam
# Trường hợp khác mở file video
else:
    camera = cv2.VideoCapture(args["video"])  # Sử dụng trường hợp đọc file video

while True:
    # Lấy khung hình hiện tại
    (grabbed, frame) = camera.read()
    # Nếu chúng ta đang xem một video và chúng ta không lấy được khung hình,
    # thì kết thúc video
    if args.get("video") and not grabbed:
        break

    # Thay đổi kích thước frame
    frame = imutils.resize(frame, width=600)
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # sao chép frame để vẽ trên nó sau này
    frame_clone = frame.copy()

    # Dò mặt người trong frame, lấy frame để vẽ trên nó sau này
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # Lặp qua hộp xung quanh khuông mặt
    for (fX, fY, fW, fH) in faces:
        # rích xuất ROI của khuôn mặt từ ảnh xám,thay đổi kích thước thành 28x28 pixel
        # chuẩn bị ROI để phân loại mặt cười/không cười bằng CNN sau này
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Dự đoán "smiling" và "not smiling" để gán nhãn cho nó
        class_path = 'datasets'
        class_names = []
        folders = [f for f in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, f))]
        for folder in folders:
            class_names.append(folder)  # Nạp nhãn vào danh sách labels
        predict = model.predict(roi)
        confidence = np.max(predict)
        predicted_label_name = class_names[np.argmax(predict)]

        if confidence > 0.7:
            profile = getProfile(predicted_label_name)
            confidence = "  {0}%".format(round(confidence*100))
            if (profile != None):
                # if((str(id)) not in dict):
                filename = xlwrite.output('attendance', 'sheet', profile[0], profile[1], profile[2], profile[3],
                                          profile[4], 'yes')

                cv2.putText(frame_clone, "Name: " + str(profile[1]) + str(confidence), (fX, fY + fH + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame_clone, "Gender: " + str(profile[3]), (fX, fY + fH + 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.putText(frame_clone, "Address: " + str(profile[6]), (fX, fY + fH + 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Unknown", (fX + 10, fY + fH + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        # Hiển thị nhãn ("Smiling" hoặc "Not Smiling") và hộp hình vuông trên frame

    # Hiển thị mặt người và nhãn "smiling" hoặc "not smiling"
    cv2.imshow("Demo thu nghiem Face", frame_clone)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Xóa camera và đóng ứng dụng
camera.release()
cv2.destroyAllWindows()
