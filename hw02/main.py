import cv2
import os
import numpy as np

scriptDir = os.path.dirname(__file__)
sourceDir = "pic"
outputDir = "output"


def main():
    outputDirPath = os.path.join(scriptDir, outputDir)
    if not os.path.exists(outputDirPath):
        os.makedirs(outputDirPath)
    filenames = ["1.jpg", "2.jpg", "3.jpg"]
    for filename in filenames:
        sourcePath = os.path.join(scriptDir, sourceDir, filename)
        outputPath = os.path.join(scriptDir, outputDir, f"gray_{filename}")
        # gray
        img = cv2.imread(sourcePath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(outputPath, img_gray)

        # filter
        outputPath = os.path.join(scriptDir, outputDir, f"filter_{filename}")
        img_filter = cv2.medianBlur(img_gray, 5)
        cv2.imwrite(outputPath, img_filter)

        # edge
        outputPath = os.path.join(scriptDir, outputDir, f"edge_{filename}")
        img_edge = cv2.Canny(img_filter, 100, 200)
        cv2.imwrite(outputPath, img_edge)

        # (bin)

        # morphology
        outputPath = os.path.join(
            scriptDir, outputDir, f"morphology_{filename}")
        kernel = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((3, 3), np.uint8)
        img_morphology = cv2.dilate(img_edge, kernel)
        img_morphology = cv2.erode(img_morphology, kernel)
        cv2.imwrite(outputPath, img_morphology)

        # line
        outputPath = os.path.join(
            scriptDir, outputDir, f"line_{filename}")
        lines = cv2.HoughLinesP(
            img_morphology, 1, np.pi/180, 100, minLineLength=75, maxLineGap=7)
        img_line = np.copy(img)
        RED = (0, 0, 255)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_line, (x1, y1), (x2, y2), RED, 3, cv2.LINE_AA)
        cv2.imwrite(outputPath, img_line)


if __name__ == "__main__":
    main()
