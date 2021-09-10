import mediapipe as mp
import time
import cv2 as cv


class FaceMeshDetector():

    def __init__(self,staticMode=False,maxFaces=2, minDetectionsCon=0.5, minTrackCon = 0.5):
        self.staticMode = staticMode
        self.maxFaces= maxFaces
        self.minDetectionCon = minDetectionsCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.minDetectionCon,self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self,img, draw=True):
        self.imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:

            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                    self.drawSpec,self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    # cv.putText(img, str(id), (x,y), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                    # print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces






def main():
    cap = cv.VideoCapture("Videos/9.mp4") # capture video file
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()

        # generally image dimensions are height, width but open cv expects it to be in (w, h) therefore next line will give wrong dimension
        # img = cv.resize(img,(540,1024),fx=0.5,fy=0.5,interpolation=cv.INTER_AREA) # This line will give wrong value of (h, w) instead (w, h)

        img = cv.resize(img,((img.shape[1]//4),(img.shape[0]//4)),fx=0.5,fy=0.5,interpolation= cv.INTER_AREA) # This line and following performs same action
        # img = cv.resize(img,(1024, 540),fx=0.5,fy=0.5,interpolation=cv.INTER_AREA)


        img, faces = detector.findFaceMesh(img, True)
        if len(faces)!=0:
            print(faces[0])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(img,f'FPS: {int(fps)}', (20,70), cv.FONT_HERSHEY_PLAIN, 3, (0,255,0),3)

        cv.imshow('Image', img)
        if cv.waitKey(1) & 0xFF==ord('q'):
            break

if __name__ == "__main__":
    main()
