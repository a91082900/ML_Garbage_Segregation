import numpy as np
import cv2

BELT_Y_THRESHOLD = 390
OUTPUT_SIZE = 320

def garbage_extract_no_preprocess(img, showCompare = False):
    """
        Extracts the garbage from image
        img: A 4:3 image
        Returns the object's image with size 320x320, black borders, scale to fit

        if showCompare is True, returns the image with annotations without clipping
    """

    # Flip if needed
    if img.shape[0] < img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 

    # Resize
    img = cv2.resize(img, (390, 520), interpolation=cv2.INTER_AREA)

    output = img.copy()

    # Chroma Keying (blue)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maskKey = cv2.inRange(hsv, (90,100,130),(110,255,255))
    maskKey = cv2.bitwise_not(maskKey)
    output = cv2.bitwise_and(output, output, mask = maskKey)

    # cv2.imshow('Output', hsv)
    # cv2.waitKey(0)
    
    # Find upper borders
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bw[:BELT_Y_THRESHOLD - 10, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fliter small dots
    fil = list(filter(lambda x: cv2.contourArea(x) > 300, contours))

    cv2.drawContours(output, fil, -1, 255, 3)
    # cv2.imshow('Output', output)
    # cv2.waitKey(0)

    x,y,w,h = cv2.boundingRect(np.vstack(fil))

    # Delete left & right
    output[:, :x, :] = 0
    output[:, x+w: , :] = 0

    # Object border
    rect = [x, y, x+w, 500]

    # grabCut
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    output = output*mask2[:,:,np.newaxis]

    if showCompare:
        output = cv2.rectangle(output, rect[:2], rect[2:], (0,255,0), 3)
        return output

    # Clip
    # output = output[rect[1]:rect[3], rect[0]:rect[2], :]
    print(rect[1], rect[3], rect[0], rect[2])
    h = rect[3] - rect[1]
    w = rect[2] - rect[0]
    if h > w:
        rect[2] = min(rect[2] + (h-w)//2, output.shape[0]-1)
        rect[0] = max(rect[0] - (h-w)//2, 0)
    else:
        rect[3] = min(rect[3] + (w-h)//2, output.shape[1]-1)
        rect[1] = max(rect[1] - (w-h)//2, 0)
    print(rect[1], rect[3], rect[0], rect[2])
    output = img[rect[1]:rect[3], rect[0]:rect[2], :]
    
    # Resize
    h, w, c = output.shape
    scale = OUTPUT_SIZE/w if w > h else OUTPUT_SIZE/h
    output = cv2.resize(output, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    delta_w = OUTPUT_SIZE - output.shape[1]
    delta_h = OUTPUT_SIZE - output.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    output = cv2.copyMakeBorder(output, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return output

def garbage_extract(img, showCompare = False):
    """
        Extracts the garbage from image
        img: A 4:3 image
        Returns the object's image with size 320x320, black borders, scale to fit

        if showCompare is True, returns the image with annotations without clipping
    """

    # Flip if needed
    if img.shape[0] < img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 

    # Resize
    img = cv2.resize(img, (390, 520), interpolation=cv2.INTER_AREA)

    output = img.copy()

    # Chroma Keying (blue)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maskKey = cv2.inRange(hsv, (90,100,130),(110,255,255))
    maskKey = cv2.bitwise_not(maskKey)
    output = cv2.bitwise_and(output, output, mask = maskKey)

    # cv2.imshow('Output', hsv)
    # cv2.waitKey(0)
    
    # Find upper borders
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bw[:BELT_Y_THRESHOLD - 10, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fliter small dots
    fil = list(filter(lambda x: cv2.contourArea(x) > 300, contours))

    # cv2.drawContours(output, fil, -1, 255, 3)
    # cv2.imshow('Output', output)
    # cv2.waitKey(0)

    x,y,w,h = cv2.boundingRect(np.vstack(fil))

    # Delete left & right
    output[:, :x, :] = 0
    output[:, x+w: , :] = 0

    # Object border
    rect = (x, y, x+w, 500)

    # grabCut
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    output = output*mask2[:,:,np.newaxis]

    if showCompare:
        output = cv2.rectangle(output, rect[:2], rect[2:], (0,255,0), 3)
        return output

    # Clip
    output = output[rect[1]:rect[3], rect[0]:rect[2], :]
    
    # Resize
    h, w, c = output.shape
    scale = OUTPUT_SIZE/w if w > h else OUTPUT_SIZE/h
    output = cv2.resize(output, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    delta_w = OUTPUT_SIZE - output.shape[1]
    delta_h = OUTPUT_SIZE - output.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    output = cv2.copyMakeBorder(output, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return output
if __name__ == "__main__":
    img = cv2.imread('Camera/paper/paper_leafyellow_14.jpg')

    # compare = garbage_extract(img, True)
    output = garbage_extract(img)

    # Flip if needed
    if img.shape[0] < img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 

    # Resize
    img = cv2.resize(img, (390, 520), interpolation=cv2.INTER_AREA)
    
    # cv2.imshow('Compare', np.hstack([compare, img]))
    cv2.imshow('Output', output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
