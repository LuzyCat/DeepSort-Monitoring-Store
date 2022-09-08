import cv2

def test():
    cap = cv2.VideoCapture('http://root:Realketiketi34@192.168.1.201/mjpg/1/video.mjpg')  # video capture object
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size
    cap.set(cv2.CAP_PROP_FPS, 30)  # set fps
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

if __name__ == '__main__':
    test()