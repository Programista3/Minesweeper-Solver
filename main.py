import numpy as np
import cv2
from scipy.special import expit
from PIL import ImageGrab

import time

class Screen:
    def takeScreenShot(self):
        return np.array(ImageGrab.grab())

class Image:
    def __init__(self, image):
        self.image = image

    def toHSV(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        return self

    def extractBlack(self):
        lowerBlack = np.array([0,0,0])
        upperBlack = np.array([180,255,60])
        self.image = cv2.inRange(self.image, lowerBlack, upperBlack)
        return self

    def detectEdges(self):
        self.image = cv2.Canny(self.image, 100, 200)
        return self
    
    def dilate(self):
        self.image = cv2.dilate(self.image, np.ones((3,3), np.uint8), iterations=1)
        return self

    def erode(self):
        self.image = cv2.erode(self.image, np.ones((3,3), np.uint8), iterations = 1)
        return self

    def findLines(self):
        return cv2.HoughLines(self.image, 1, np.pi/180, 300)

    def classifyFields(self):
        pass

class Lines:
    def __init__(self, lines):
        self.lines = lines

    def removeNegativeLines(self):
        toRemove = []

        for i in range(len(self.lines)):
            rho, theta = self.lines[i][0]
            if(rho < 0):
                toRemove.append(i)
        self.lines = np.delete(self.lines, toRemove, axis=0)
        return self

    def removeSimilarLines(self):
        rho_treshold = 30
        theta_threshold = 0.1
        similarLines = []

        for i in range(len(self.lines)-1):
            if(i in similarLines):
                continue
            for j in range(i+1, len(self.lines)):
                if(j in similarLines):
                    continue
                rho_i, theta_i = self.lines[i][0]
                rho_j, theta_j = self.lines[j][0]
                if(abs(abs(rho_i)-abs(rho_j)) < rho_treshold and abs(theta_i-theta_j) < theta_threshold):
                    similarLines.append(j)
        self.lines = np.delete(self.lines, similarLines, axis=0)
        return self

    def sortLines(self, imageShape):
        horizontalLines = []
        verticalLines = []
        for line in self.lines:
            rho, theta = line[0]
            if(theta < 1):
                verticalLines.append(rho)
            elif(rho > 50 and rho < imageShape[0]-50):
                horizontalLines.append(rho)
        horizontalLines.sort()
        verticalLines.sort()
        verticalLines.pop(0)
        verticalLines.pop()
        return list(map(int, horizontalLines)), list(map(int, verticalLines))

class MinesweeperSolver:
    def preprocessing(self):
        screen = Screen()
        screenshot = screen.takeScreenShot()
        image = Image(screenshot)
        image.toHSV().extractBlack().detectEdges().dilate().erode()
        lines = Lines(image.findLines())
        lines.removeNegativeLines().removeSimilarLines()
        horizontalLines, verticalLines = lines.sortLines(screenshot.shape)
        return horizontalLines, verticalLines

    def classifyFields(self, image, horizontalLines, verticalLines):
        pass

if __name__ == '__main__':
    minesweeper = MinesweeperSolver()
    minesweeper.preprocessing()