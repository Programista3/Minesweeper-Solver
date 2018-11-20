import numpy as np
import cv2
from scipy.special import expit
from PIL import ImageGrab

class Image:
    def takeScreenShot(self):
        self.image = np.array(ImageGrab.grab())
        self.img2 = self.image.copy()

    def toHSV(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def extractBlack(self):
        lowerBlack = np.array([0,0,0])
        upperBlack = np.array([180,255,60])
        self.image = cv2.inRange(self.image, lowerBlack, upperBlack)

    def detectEdges(self):
        self.image = cv2.Canny(self.image, 100, 200)
    
    def dilate(self):
        self.image = cv2.dilate(self.image, np.ones((3,3), np.uint8), iterations=1)

    def erode(self):
        self.image = cv2.erode(self.image, np.ones((3,3), np.uint8), iterations = 1)

    def findLines(self):
        lines = cv2.HoughLines(self.image, 1, np.pi/180, 300)
        image = self.img2.copy()
        for line in lines:
            rho, theta = line[0][0], line[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1500*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1500*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
        print("Znaleziono linii:", len(lines))
        cv2.imshow('test2', image)
        cv2.waitKey(0)
        return lines

    def removeNegativeLines(self):
        pass

    def removeSimilarLines(self):
        pass

    def sortLines(self):
        pass

    def classifyFields(self):
        pass

if __name__ == '__main__':
    img = Image()
    img.takeScreenShot()
    img.toHSV()
    img.extractBlack()
    img.detectEdges()
    img.dilate()
    img.erode()
    img.findLines()
    cv2.imshow('test', img.image)
    cv2.waitKey(0)