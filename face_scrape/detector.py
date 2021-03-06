import cv2
import numpy as np
import math
import os
from math import sin, cos

max_size = 720

def detect(img):
    base = os.path.dirname(os.path.abspath(__file__))
    f_file = os.path.normpath(os.path.join(base, 'haarcascade_frontalface_alt2.xml'))
    e_file = os.path.normpath(os.path.join(base, 'haarcascade_eye.xml'))
    cascade_f = cv2.CascadeClassifier(f_file)
    cascade_e = cv2.CascadeClassifier(e_file)
    # resize if learch image
    rows, cols, _ = img.shape
    if max(rows, cols) > max_size:
        l = max(rows, cols)
        img = cv2.resize(img, (cols * max_size // l, rows * max_size // l))
    rows, cols, _ = img.shape
    # create gray image for rotate
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hypot = int(math.ceil(math.hypot(rows, cols)))
    frame = np.zeros((hypot, hypot, 3), np.uint8)
    frame[(hypot - rows) * 0.5:(hypot + rows) * 0.5, (hypot - cols) * 0.5:(hypot + cols) * 0.5, :] = img

    def translate(coord, deg):
        x, y = coord
        rad = math.radians(deg)
        return {
            'x': (  cos(rad) * x + sin(rad) * y - hypot * 0.5 * cos(rad) - hypot * 0.5 * sin(rad) + hypot * 0.5 - (hypot - cols) * 0.5) / float(cols) * 100.0,
            'y': (- sin(rad) * x + cos(rad) * y + hypot * 0.5 * sin(rad) - hypot * 0.5 * cos(rad) + hypot * 0.5 - (hypot - rows) * 0.5) / float(rows) * 100.0,
        }

    def rotate(img, deg):
        hypot = img.shape[0]
        M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), deg, 1.0)
        rotated = cv2.warpAffine(img, M, (hypot, hypot))
        return rotated

    # rotate and detect faces
    results = []
    for deg in range(-48, 49, 6):
        rotated = rotate(frame, deg)
        faces = cascade_f.detectMultiScale(rotated, 1.08, 2)
        for face in faces:
            x, y, w, h = face
            # eyes in face?
            y_offset = int(h * 0.1)
            roi = rotated[y + y_offset: y + h, x: x + w]
            eyes = cascade_e.detectMultiScale(roi, 1.05)
            validator = lambda e: (e[0] > w / 2 or e[0] + e[2] < w / 2) and e[1] + e[3] < h / 2
            eyes = [eye for eye in eyes if validator(eye)]
            # eyes = filter(lambda e: (e[0] > w / 2 or e[0] + e[2] < w / 2) and e[1] + e[3] < h / 2, eyes)
            if len(eyes) == 2 and abs(eyes[0][0] - eyes[1][0]) > w / 4:
                score = math.atan2(abs(eyes[1][1] - eyes[0][1]), abs(eyes[1][0] - eyes[0][0]))
                if eyes[0][1] == eyes[1][1]:
                    score = 0.0
                results.append({
                    'center': translate([x + w * 0.5, y + h * 0.5], -deg),
                    'w': float(w) / float(cols) * 100.0,
                    'h': float(h) / float(rows) * 100.0,
                    'eyes': [translate([x + e[0] + e[2] * 0.5, y + y_offset + e[1] + e[3] * 0.5], -deg) for e in eyes],
                    'score': score,
                    'coordinates': [x, y, w, h],
                    'deg': deg,
                })

    # unify duplicate faces
    faces = []
    for result in results:
        x, y = result['center']['x'], result['center']['y']
        exists = False
        for i in range(len(faces)):
            face = faces[i]
            if (face['center']['x'] - face['w'] * 0.5 < x < face['center']['x'] + face['w'] * 0.5 and
                face['center']['y'] - face['h'] * 0.5 < y < face['center']['y'] + face['h'] * 0.5):
                exists = True
                if result['score'] < face['score']:
                    faces[i] = result
                    break
        if not exists:
            faces.append(result)
    for face in faces:
        del face['score']

    if len(faces) == 0:
        raise Exception("There is no face.")
    elif len(faces) > 1:
        raise Exception("There is too many faces.")
    else:
        face = faces[0]
        x, y, w, h = face['coordinates']
        rotated = rotate(frame, face['deg'])
        return rotated[y:y+h, x:x+w]

