import cv2, math, av, sys, torch, imutils
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# face and eye detectors
path = cv2.__file__
path = path.split('cv2.')[0]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load(path+'data/haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")
eye_detector.load(path+'data/haarcascade_eye.xml')

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def align(img):
    eyes = eye_detector.detectMultiScale(img)
    index = 0
    for (eye_x, eye_y, eye_w, eye_h) in eyes:
        if index == 0:
            eye_1 = (eye_x, eye_y, eye_w, eye_h)
        elif index == 1:
            eye_2 = (eye_x, eye_y, eye_w, eye_h)
        index = index + 1
    if eye_1[0] < eye_2[0]:
        left_eye = eye_1
        right_eye = eye_2
    else:
        left_eye = eye_2
        right_eye = eye_1
    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0];
    left_eye_y = left_eye_center[1]
    right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
    right_eye_x = right_eye_center[0];
    right_eye_y = right_eye_center[1]
    if left_eye_y < right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock
    a = euclidean_distance(left_eye_center, point_3rd)
    b = euclidean_distance(right_eye_center, left_eye_center)
    c = euclidean_distance(right_eye_center, point_3rd)
    cos_a = (b * b + c * c - a * a) / (2 * b * c)
    angle = np.arccos(cos_a)
    angle = (angle * 180) / math.pi
    if direction == -1:
        angle = 90 - angle
    return direction, angle


def EmotionRecognitionVideo(video_name='test_video.mp4', model_name='CNN5'):

    container = av.open('Video/' + video_name)
    net = torch.load('Models/' + model_name + '.p', map_location=torch.device('cpu'))
    original_frames = []
    # save frames
    count = 0
    for frame in container.decode(video=0):
        img = frame.to_image()
        original_frames.append([count, np.array(img)])
        count += 1

    emo_dict = {1:'Surprise', 2:'Fear', 3:'Disgust', 4:'Happiness', 5:'Sadness', 6:'Anger', 7:'Neutral'}
    face_list = []

    for iframe in original_frames:
        i, frame = iframe[0], iframe[1]
        frame = np.array(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # align each face and apply our trained FER model
        if faces is not None:
            for face in faces:
                with torch.no_grad():
                    face_img = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
                    img_raw = frame
                    try: # align the pic if the both eyes are detected
                        direction, angle = align(face_img)
                        img = Image.fromarray(img_raw)
                        img = np.array(img.rotate(direction * angle))
                    except: # just resize the pic for 100*100 without aligning
                        (x, y, w, h) = face[0], face[1], face[2], face[3]
                        img = imutils.resize(frame[y:y + h, x:x + w], width=100)
                    img = transforms.ToTensor()(img.copy())
                    train = [[img, 0]]
                    trainLoader = torch.utils.data.DataLoader(train, shuffle=True)
                    for data in trainLoader:
                        image, label = data
                        img = image
                        try:
                            outputs = net(img)
                        except:
                            continue
                    _, predicted = torch.max(outputs.data, 1)
                    emo = emo_dict[predicted.data.tolist()[0]]
                    face_list.append([i, face, emo])
    # put rectangles and emotions as text for faces
    for face in face_list:
        i,face,emo = face[0],face[1],face[2]
        original_frames[i][1] = cv2.rectangle(original_frames[i][1], (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), (255, 0, 0), 2)
        cv2.putText(original_frames[i][1], emo, (face[0], face[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    dataset = []
    # convert images to RGB
    for data in original_frames:
        img = data[1]
        img_rgb = img[:, :, ::-1]
        dataset.append(img_rgb)
    # making a video by putting all the frames together
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('Video/' + model_name + '.mp4', fourcc, 30, (640, 360), True)
    for i in range(len(dataset)):
        out.write(dataset[i])
    out.release()

if __name__ == '__main__':
    EmotionRecognitionVideo(video_name='test_video.mp4')