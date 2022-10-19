import time
import cv2

def majorityElement(nums):
    for i in nums[len(nums) // 2:]:
        if nums.count(i) > len(nums) // 2:
            return i

def format_image(image):
    # image如果为彩色图：image.shape[0][1][2](水平、垂直像素、通道数)
    if len(image.shape) > 2 and image.shape[2] == 3:
        # 将图片变为灰度图
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
        # 调整scaleFactor参数的大小，可以增加识别的灵敏度，推荐1.1
        # CASC_PATH = 你的haarcascade_frontalface_alt2.xml文件地址
        CASC_PATH = './include/FindFace.xml'  #
        cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
        faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    # 如果图片中没有检测到人脸，则返回None
    if not len(faces) > 0:
        return None, None, 0
    # max_are_face包含了人脸的坐标，大小
    max_are_face = faces[0]
    # 在所有人脸中选一张最大的脸
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face

    # 这两步可有可无
    face_coor = max_are_face
    image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
    if len(faces) > 0:
        Count = 1

    else:Count = 0

    return image, face_coor, Count


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    fps = 0.0
    RawNum = 0
    lab_num = 0
    while 1:
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 进行检测
        (p_image, face_coor, Count) = format_image(frame)
        if RawNum != Count:
            RawNum = Count
            if Count == 1:
                lab_num += 1
        else:
            RawNum = Count
        if face_coor is not None:
            # 获取人脸的坐标,并用矩形框出
            [x, y, w, h] = face_coor
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fps = (fps + (1. / (time.time() - t1))) / 2
        cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Count= %.2f" % lab_num, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("video", frame)
        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break
    capture.release()
    cv2.destroyAllWindows()

