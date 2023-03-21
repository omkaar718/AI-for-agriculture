import numpy as np
import cv2

def four_point_transform(image, pts, maxHeight, maxWidth):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = pts
	'''

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order

	'''
	'''
	# grid
	maxWidth = 1060
	maxHeight = 444
	'''
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return M, warped

'''

image = cv2.imread('isometric-view-2.jpg')
#image = cv2.imread('grid.jpg')
#pts = np.array([(1768, 587), (2843, 1019), (1985, 2035), (639, 1316)], dtype = "float32")

#poster
pts = np.array([(1763, 1098), (3375, 1476), (2284, 4086), (98, 2829)], dtype = "float32")

# grid
# pts = np.array([(347, 686), (2725, 49), (2626, 1730), (45, 1619)], dtype = "float32")

transformation_matrix, warped = four_point_transform(image, pts)
#cv2.imwrite('warped.png', warped)
#cv2.imwrite('OG.png', image)
'''
'''
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
'''

def generate_graphics(points, text, maxHeight, maxWidth):
    border = 150
    img = np.full((maxHeight, maxWidth,3), 255, np.uint8)
    img = cv2.rectangle(img, (0, 0), (maxWidth, maxHeight), (0, 0, 0), 2)
    img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, None, value = (255, 255, 255))


    for (x_coord, y_coord), annotation in zip(points, text):
        x_coord += border
        y_coord += border
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(img, (x_coord, y_coord), radius= 5,  color=(0, 0, 255), thickness=cv2.FILLED)
        for i, t in enumerate(annotation):
                cv2.putText(img, t, (x_coord, y_coord + (i+1)*17), font, 0.6,(255,0,0),1,cv2.LINE_AA)
		
    cv2.imshow("generated", img)
    cv2.waitKey(0)
    cv2.imwrite('graphics_with_activity.jpg', img)
    return img


#generate_graphics([(100, 100), (200, 200)], 500, 500)
