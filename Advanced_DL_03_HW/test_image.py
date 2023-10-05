import cv2

image = cv2.imread("./football/Match_1951_1_0_subclip/images/frame2.jpg")
height,width,_ = image.shape #BGR

with open("./football/Match_1951_1_0_subclip/labels/frame_02.txt", 'r') as fo:
    annotation = fo.readlines()
annotation = [anno.rstrip().split(" ") for anno in annotation]
for anno in annotation:
    category, x, y, w, h = anno   #number
    xcent = float(x)*width
    ycent = float(y)*height
    w = float(w)*width
    h = float(h)*height
    # As image display the image must be in the int form
    xtl= int(xcent - w/2)
    ytl= int(ycent - h/2)
    xbr= int(xcent + w/2)
    ybr= int(ycent + h/2)
    cv2.rectangle(image, (xtl,  ytl), (xbr, ybr), (0, 255, 0), 2)

cv2.imwrite("sample.jpg",image)