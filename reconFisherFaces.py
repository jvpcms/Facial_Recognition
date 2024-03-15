import cv2

names = ["Belchior", "Paim", "Saboya", "Unknown"]

faceDetector = cv2.CascadeClassifier("xmlFiles/haarcascade_frontalface_default.xml")
recon = cv2.face.FisherFaceRecognizer.create()
recon.read("classifierFiles/classifierFisher.yml")

width, height = 220, 220
font = cv2.FONT_HERSHEY_PLAIN

webcam = cv2.VideoCapture(0)

while True:
    conected, image = webcam.read()

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detectedFaces = faceDetector.detectMultiScale(grayImage, scaleFactor=1.5, minSize=(30, 30))

    for x, y, w, h in detectedFaces:
        imgFace = cv2.resize(grayImage[y:y + h, x:x + w], (width, height))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        ID, trust = recon.predict(imgFace)
        cv2.putText(image, names[ID], (x, y + (h + 30)), font, 2, (0, 0, 255))
        cv2.putText(image, str(trust), (x, y + (h + 50)), font, 2, (0, 0, 255))

    cv2.imshow("Face", image)

    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
