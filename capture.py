import cv2
import numpy as np

classifier = cv2.CascadeClassifier("xmlFiles/haarcascade_frontalface_default.xml")
classifierEyes = cv2.CascadeClassifier("xmlFiles/haarcascade_eye.xml")

webcam = cv2.VideoCapture(0)

sample = 1  # ammount of detected faces
sampleSize = 25  # ammount of pictures per face
lighting = 110  # luminosity treshold for capture

ID = input("Type ID: ")
width, height = 220, 220

while True:
    conected, image = webcam.read()
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # matrix of positions where a face is detected
    detectedFaces = classifier.detectMultiScale(grayImage, scaleFactor=1.5, minSize=(150, 150))

    for (x, y, w, h) in detectedFaces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        region = image[y:y + h, x:x + w]
        grayRegionEye = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        detectedEyes = classifierEyes.detectMultiScale(grayRegionEye)

        # by putting the capture block inside the eye detection loop we are able to capture only when eyes are detected
        for (xEye, yEye, wEye, hEye) in detectedEyes:
            cv2.rectangle(region, (xEye, yEye), (xEye + wEye, yEye + hEye), (0, 255, 0), 2)

            # Capture and save image
            if (cv2.waitKey(1) & 0xFF == ord('q')) and np.average(grayImage) > lighting:
                imgFace = cv2.resize(grayImage[y:y + h, x:x + w], (width, height))
                cv2.imwrite("pictures/person." + ID + '.' + str(sample) + ".jpg", imgFace)
                print("saved picture " + str(sample))
                sample += 1

    cv2.imshow("Face", image)
    cv2.waitKey(1)

    if sample > sampleSize:
        break

webcam.release()
cv2.destroyAllWindows()
