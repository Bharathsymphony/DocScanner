import cv2
import numpy as np
import utlis
import tkinter as tk
from PIL import ImageTk, Image
import time
import sys
from tkinter import messagebox

########################################################################
webCamFeed = True
pathImage = "testImg.jpg"
# cap = cv2.VideoCapture(1)
# cap.set(10,160)
heightImg = 640
widthImg  = 480
########################################################################

utlis.initializeTrackbars()
count=0

while True:

    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    thres=utlis.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    ## FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    _,contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS


    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utlis.biggestContour(contours) # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest=utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        #REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)

        # Image Array for Display
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])

    else:
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # LABELS FOR DISPLAY
    lables = [["Original","Gray","Threshold","Contours"],
              ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]

    stackedImage = utlis.stackImages(imageArray,0.75,lables)
    cv2.imshow("Result",stackedImage)
    dImg=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2RGBA)
    
    # BUTTONS TO SAVE IMAGE BY ADDING FILTER
    root=tk.Tk()
    saveImg=imgWarpColored
    org_image = ImageTk.PhotoImage(Image.open("OriginalImage.jpg"))
    gray_image=ImageTk.PhotoImage(Image.open("grayImage.jpg"))
    sat_image=ImageTk.PhotoImage(Image.open("saturated.jpg"))
    
    def orgImg():
        saveImg=imgWarpColored
        newImg=utlis.saveImage(stackedImage,saveImg,count)
    
    def grayImg():
        saveImg= imgWarpGray
        newImg=utlis.saveImage(stackedImage,saveImg,count)
    
    def darkImg():
        saveImg= dImg
        newImg=utlis.saveImage(stackedImage,saveImg,count)
        
    def ifExit():
        root.destroy()
                
    o=tk.Button(root,text="Original",command=orgImg)
    g=tk.Button(root,text="Gray",command=grayImg)
    d=tk.Button(root,text="More Saturated",command=darkImg)
    e=tk.Button(root,text="Exit",command=ifExit)
         
    neworg_img=tk.Label(image=org_image)
    newgray_img=tk.Label(image=gray_image)
    newsat_img=tk.Label(image=org_image)
    
    text=tk.Label(text="SELECT FILTER TO BE SAVED")
    
    neworg_img.grid(row=1,column=1)
    newgray_img.grid(row=1,column=2)
    newsat_img.grid(row=1,column=3)

    text.grid(row=0,column=2)
    o.grid(row=2,column=1)
    g.grid(row=2,column=2)
    d.grid(row=2,column=3)
    e.grid(row=0,column=4)

    root.mainloop()

    # EXIT BUTTON
    nroot=tk.Tk()
    msgbox=tk.messagebox.askquestion('EXIT APP','Do You want to Exit ?',icon='warning')
    
    if msgbox=='yes':
        nroot.destroy()
        break
    else:
        nroot.destroy()

    count+=1



    
    