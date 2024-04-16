import cv2
import os
import numpy as np

scriptDir = os.path.dirname(__file__)
sourceDir = "hw1_picture"
outputDir = "output"


def main():
    outputDirPath = os.path.join(scriptDir, outputDir)
    if not os.path.exists(outputDirPath):
        os.makedirs(outputDirPath)
    pic1()
    pic2()
    pic3()
    labelPeople()


def labelPeople():
    # label
    sourcePath = os.path.join(scriptDir, sourceDir, "people.jpg")
    outputPath = os.path.join(scriptDir, outputDir, "label_people.jpg")
    img = cv2.imread(sourcePath)
    height, width, channel = img.shape
    redColor = (0, 0, 255)
    topLeft = (width*1//4, height*1//4)
    bottomRight = (width*3//4, height*3//4)
    textTopLeft = (topLeft[0], topLeft[1]-10)
    cv2.rectangle(img, topLeft, bottomRight, redColor, 2, cv2.LINE_AA)
    cv2.putText(img, "Maybe this is face", textTopLeft,
                cv2.FONT_HERSHEY_SIMPLEX, 1, redColor, 1, cv2.LINE_AA)
    cv2.imwrite(outputPath, img)


def pic1():
    filename = "pic1.jpg"
    # gray
    sourcePath = os.path.join(scriptDir, sourceDir, filename)
    img = cv2.imread(sourcePath)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    outputPath = os.path.join(scriptDir, outputDir, f"gray_{filename}")
    cv2.imwrite(outputPath, img_gray)

    # filter
    img_filter = cv2.medianBlur(img_gray, 5)
    outputPath = os.path.join(scriptDir, outputDir, f"filter_{filename}")
    cv2.imwrite(outputPath, img_filter)

    # binarization
    ret, img_binarization = cv2.threshold(
        img_filter, 110, 255, cv2.THRESH_BINARY)
    outputPath = os.path.join(
        scriptDir, outputDir, f"binarization_{filename}")
    cv2.imwrite(outputPath, img_binarization)

    # morphology
    kernel = np.ones((5, 5), np.uint8)
    img_morphology = cv2.morphologyEx(
        img_binarization, cv2.MORPH_CLOSE, kernel)
    outputPath = os.path.join(
        scriptDir, outputDir, f"morphology_{filename}")
    cv2.imwrite(outputPath, img_morphology)


def pic2():
    filename = "pic2.jpg"
    # gray
    sourcePath = os.path.join(scriptDir, sourceDir, filename)
    img = cv2.imread(sourcePath)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    outputPath = os.path.join(scriptDir, outputDir, f"gray_{filename}")
    cv2.imwrite(outputPath, img_gray)

    # filter
    img_filter = cv2.bilateralFilter(img_gray, 50, 50, 25)
    outputPath = os.path.join(scriptDir, outputDir, f"filter_{filename}")
    cv2.imwrite(outputPath, img_filter)

    # binarization
    ret, img_binarization = cv2.threshold(
        img_filter, 150, 255, cv2.THRESH_BINARY)
    outputPath = os.path.join(
        scriptDir, outputDir, f"binarization_{filename}")
    cv2.imwrite(outputPath, img_binarization)

    # morphology
    kernel = np.ones((5, 5), np.uint8)
    img_morphology = cv2.morphologyEx(img_binarization, cv2.MORPH_OPEN, kernel)
    outputPath = os.path.join(
        scriptDir, outputDir, f"morphology_{filename}")
    cv2.imwrite(outputPath, img_morphology)


def pic3():
    filename = "pic3.jpg"
    # gray
    sourcePath = os.path.join(scriptDir, sourceDir, filename)
    img = cv2.imread(sourcePath)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    outputPath = os.path.join(scriptDir, outputDir, f"gray_{filename}")
    cv2.imwrite(outputPath, img_gray)

    # filter
    img_filter = cv2.blur(img_gray, (9, 9))
    outputPath = os.path.join(scriptDir, outputDir, f"filter_{filename}")
    cv2.imwrite(outputPath, img_filter)

    # binarization
    ret, img_binarization = cv2.threshold(
        img_filter, 125, 255, cv2.THRESH_BINARY)
    outputPath = os.path.join(
        scriptDir, outputDir, f"binarization_{filename}")
    cv2.imwrite(outputPath, img_binarization)

    # morphology
    kernel = np.ones((5, 5), np.uint8)
    img_morphology = cv2.morphologyEx(
        img_binarization, cv2.MORPH_CLOSE, kernel)
    outputPath = os.path.join(
        scriptDir, outputDir, f"morphology_{filename}")
    cv2.imwrite(outputPath, img_morphology)


if __name__ == "__main__":
    main()
