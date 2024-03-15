import cv2
import os
import numpy as np

eingenFace = cv2.face.EigenFaceRecognizer.create(threshold=8500)
fisherFace = cv2.face.FisherFaceRecognizer.create(2, 2000)
lbph = cv2.face.LBPHFaceRecognizer.create(threshold=60)


def get_image_by_id():
    paths = [os.path.join('pictures', p) for p in os.listdir('pictures')]

    faces = []
    ids = []

    for imagePath in paths:
        imgFace = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)
        ID = int(os.path.split(imagePath)[-1].split('.')[1])

        ids.append(ID)
        faces.append(imgFace)

    return np.array(ids), faces


ids, faces = get_image_by_id()

eingenFace.train(faces, ids)
eingenFace.write('classifierFiles/classifierEigen.yml')

fisherFace.train(faces, ids)
fisherFace.write('classifierFiles/classifierFisher.yml')

lbph.train(faces, ids)
lbph.write('classifierFiles/classifierLBPH.yml')
