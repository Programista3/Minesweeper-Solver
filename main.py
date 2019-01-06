import numpy as np
import cv2
from scipy.special import expit
from PIL import ImageGrab
import win32api
import win32con

import time
import os

class Screen:
    def takeScreenShot(self):
        return np.array(ImageGrab.grab())

class Image:
    def __init__(self, image):
        self.image = image

    def get(self):
        return self.image

    def toGray(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self

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
        return cv2.HoughLines(self.image, 1, np.pi/180, 280)

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

    def checkCorrectness(self, horizontalLines, verticalLines):
        differences = []
        for i in range(len(horizontalLines)-1):
            differences.append(horizontalLines[i+1]-horizontalLines[i])
        average = sum(differences)/len(differences)
        for i in range(len(horizontalLines)-1):
            if(horizontalLines[i+1]-horizontalLines[i] > 1.5*average):
                horizontalLines.append(int(horizontalLines[i]+(horizontalLines[i+1]-horizontalLines[i])/2))
                print("Dodano h")
        for i in range(len(verticalLines)-1):
            if(verticalLines[i+1]-verticalLines[i] > 1.5*average):
                verticalLines.append(int(verticalLines[i]+(verticalLines[i+1]-verticalLines[i])/2))
                print("Dodano w")
        horizontalLines.sort()
        verticalLines.sort()
        return horizontalLines, verticalLines

class Classifier:
    def __init__(self, filename):
        self.loadModel(filename)
    
    def loadModel(self, filename):
        data = np.load(filename)
        self.n_features = data['header'][0]
        self.n_hidden = data['header'][1]
        self.n_output = data['header'][2]
        self.w1 = data['w1']
        self.w2 = data['w2']

    def predict(self, image):
        inputWithBiasUnit, sumForHiddenLayer, activationForHiddenLayer, sumForOutputLayer, activationForOutputLayer = self.feedForward(image)
        return np.argmax(sumForOutputLayer, axis=0)

    def imageTo1D(self, image):
        return np.reshape(image, image.shape[0]*image.shape[1])

    def feedForward(self, x):
        inputWithBiasUnit = self.addBiasUnitColumn(x)
        sumForHiddenLayer = self.w1.dot(inputWithBiasUnit.T)
        activationForHiddenLayer = self.sigmoid(sumForHiddenLayer)
        activationForHiddenLayer = self.addBiasUnitRow(activationForHiddenLayer)
        sumForOutputLayer = self.w2.dot(activationForHiddenLayer)
        activationForOutputLayer = self.sigmoid(sumForOutputLayer)
        return inputWithBiasUnit, sumForHiddenLayer, activationForHiddenLayer, sumForOutputLayer, activationForOutputLayer

    def sigmoid(self, z):
        return expit(z)

    def addBiasUnitColumn(self, x):
        xWithBiasUnit = np.ones((x.shape[0], x.shape[1]+1))
        xWithBiasUnit[:, 1:] = x
        return xWithBiasUnit

    def addBiasUnitRow(self, x):
        xWithBiasUnit = np.ones((x.shape[0]+1, x.shape[1]))
        xWithBiasUnit[1:, :] = x
        return xWithBiasUnit

class Mouse:
    def clickLeft(self, x, y):
        win32api.SetCursorPos((x,y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

    def clickRight(self, x, y):
        win32api.SetCursorPos((x,y))
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,x,y,0,0)

    def moveTo(self, x, y):
        win32api.SetCursorPos((x,y))

class Board:
    def __init__(self, board):
        self.board = board
        self.height = board.shape[0]
        self.width = board.shape[1]
    
    def findMoves(self):
        toMark = set()
        toDiscover = set()
        for y in range(self.height):
            for x in range(self.width):
                if(self.board[y][x] > 0 and self.board[y][x] < 7):
                    marked, undiscovered = self.findMarkedAndUndiscovered(x+1,y+1)
                    if(len(marked) == self.board[y][x] and len(undiscovered) > 0):
                        #print("Odkrywanie",x,y,"\n",self.board)
                        print("Odkrywanie",x,y)
                        toDiscover.update(undiscovered)
                    elif(len(undiscovered) == self.board[y][x] and len(marked) == 0):
                        #print("Oznaczanie 1",x,y,"\n",self.board)
                        print("Oznaczanie 1",x,y)
                        toMark.update(undiscovered)
                    elif(len(marked) + len(undiscovered) == self.board[y][x] and len(undiscovered) > 0):
                        #print("Oznaczanie 2",x,y,"\n",self.board)
                        print("Oznaczanie 2",x,y)
                        toMark.update(undiscovered)
        return toMark, toDiscover

    def findMarkedAndUndiscovered(self, x, y):
        board = np.pad(self.board, ((1,1),(1,1)), 'constant', constant_values=0)
        indexes = [(y-1,x-1), (y-1,x), (y-1,x+1), (y,x-1), (y,x+1), (y+1,x-1), (y+1,x), (y+1,x+1)]
        marked, undiscovered = set(), set()
        for i in indexes:
            if(board[i] == 7):
                undiscovered.add((i[1]-1,i[0]-1))
            elif(board[i] == 8):
                marked.add((i[0]-1,i[1]-1))
        return marked, undiscovered

class MinesweeperSolver:
    def nextIndexInDir(self, dir):
        for root, dirs, files in os.walk(dir):
            files = [filename.split('.')[0] for filename in files]
            indexes = list(map(int, files))
            return 0 if not indexes else max(indexes)+1

    def undiscoveredExists(self, board):
        for i in range(len(board)):
            for j in range(len(board[0])):
                if(board[i][j] == 7):
                    return True
        return False

    def solve(self):
        mouse = Mouse()
        mouse.clickLeft(100, 100)
        pause = 0.05
        checkPos = False
        pos = {'x': 0, 'y': 0}
        while(True):
            #start = time.time()
            screen = Screen()
            screenshot = screen.takeScreenShot()
            #cv2.imshow('test', screenshot)
            #cv2.waitKey(0)
            screenshotGray = Image(screenshot)
            screenshotGray.toGray()
            if(self.isOver(screenshotGray.get())):
                break
            horizontalLines, verticalLines = minesweeper.preprocessing(screenshot)
            #image = screenshotGray.get().copy()
            """for line in horizontalLines:
                cv2.line(image,(0,line),(1000, line),(255,255,0),2)
            for line in verticalLines:
                cv2.line(image,(line, 0),(line, 800),(255,255,0),2)"""
            #cv2.imwrite('sc/'+str(self.nextIndexInDir('sc'))+'.png', image)
            print("Linie", len(horizontalLines), len(verticalLines))
            board = Board(minesweeper.classifyFields(screenshotGray.get(), horizontalLines, verticalLines))
            if(checkPos):
                print("a", x, y)
                print("x", board.board[y][x])
                checkPos = False
                cv2.imwrite('screen.png', screenshotGray.get())
                if(board.board[y][x] == 7 or board.board[y][x] == 9):
                    print("Bomba")
                    break
                #continue
            toMark, toDiscover = board.findMoves()
            #end = time.time()
            #print(end-start)
            if(len(toMark) == 0 and len(toDiscover) == 0):
                if(self.undiscoveredExists(board.board)):
                    while(True):
                        y = np.random.choice(board.board.shape[0], 1)[0]
                        x = np.random.choice(board.board.shape[1], 1)[0]
                        if(board.board[y,x] == 7):
                            print("Wylosowano:", x, y)
                            fieldX, fieldY, fieldWidth, fieldHeight, = self.getFieldPositionAndSize(x, y, horizontalLines, verticalLines)
                            mouse.clickLeft(round(fieldX+fieldWidth/2), round(fieldY+fieldHeight/2))
                            checkPos = True
                            pos[x] = x
                            pos[y] = y
                            time.sleep(pause)
                            break
                else:
                    print("Cała plansza rozwiązana")
                    break
            else:
                for field in toDiscover:
                    x, y, w, h, = self.getFieldPositionAndSize(field[0], field[1], horizontalLines, verticalLines)
                    mouse.clickLeft(round(x+w/2), round(y+h/2))
                    time.sleep(pause)
                for field in toMark:
                    x, y, w, h, = self.getFieldPositionAndSize(field[0], field[1], horizontalLines, verticalLines)
                    mouse.clickRight(round(x+w/2), round(y+h/2))
                    time.sleep(pause)
            mouse.moveTo(10,10)
        print("Koniec")

    def isOver(self, screenshot):
        template = cv2.imread('end.png', 0)
        templateWin = cv2.imread('win.png', 0)
        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        if(maxVal > 0.9):
            return True
        else:
            res = cv2.matchTemplate(screenshot, templateWin, cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
            if(maxVal > 0.9):
                return True
            else:
                return False

    def preprocessing(self, screenshot):
        image = Image(screenshot)
        image.toHSV().extractBlack().detectEdges().dilate().erode()
        lines = Lines(image.findLines())
        lines.removeNegativeLines().removeSimilarLines()
        horizontalLines, verticalLines = lines.sortLines(screenshot.shape)
        horizontalLines, verticalLines = lines.checkCorrectness(horizontalLines, verticalLines)
        return horizontalLines, verticalLines

    def classifyFields(self, image, horizontalLines, verticalLines):
        classifier = Classifier('model.npz')
        samples = []
        for i in range(0, len(horizontalLines)-1):
            for j in range(0, len(verticalLines)-1):
                field = image[horizontalLines[i]:horizontalLines[i+1], verticalLines[j]:verticalLines[j+1]]
                field = cv2.resize(field, (16,16))
                samples.append(classifier.imageTo1D(field))
        board = classifier.predict(np.asarray(samples)).reshape(len(horizontalLines)-1, len(verticalLines)-1)
        return board

    def getFieldPositionAndSize(self, x, y, horizontalLines, verticalLines):
        return verticalLines[x], horizontalLines[y], verticalLines[x+1]-verticalLines[x], horizontalLines[y+1]-horizontalLines[y]

if __name__ == '__main__':
    minesweeper = MinesweeperSolver()
    minesweeper.solve()