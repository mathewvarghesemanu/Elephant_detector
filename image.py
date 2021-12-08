import cv2
vc=cv2.VideoCapture(0)
while(1):
	ret,frame=vc.read()
	cv2.imshow("im",frame)
	if cv2.waitKey(1)==ord('q'):
		break
vc.release()