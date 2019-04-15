import numpy as np
import cv2

def getTarget(p,s,b,sn,bn):
    spd = p - s
    bpd = p - b
    try:
        a = sn + spd/(spd - bpd) * (bn - sn)
    except:
        print(p,s,b,sn,bn)
        print(spd,bpd,bn,sn)
        a = -1.-1
    return a

def getPoint(X,Y,matchPoint):
    matchPoint.sort(key=lambda x:(x[0][0]-X)**2+(x[0][1]-Y)**2)
    for i in range(len(matchPoint)):
        if matchPoint[0][0][0] != matchPoint[i][0][0] and matchPoint[0][1][0] != matchPoint[i][1][0] and \
            matchPoint[0][0][1] != matchPoint[i][0][1] and matchPoint[0][1][1] != matchPoint[i][1][1]:
            xp = getTarget(X,matchPoint[0][0][0],matchPoint[i][0][0],matchPoint[0][1][0],matchPoint[i][1][0])
            yp = getTarget(Y,matchPoint[0][0][1],matchPoint[i][0][1],matchPoint[0][1][1],matchPoint[i][1][1])
            break
    return (int(xp),int(yp))

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        result = getPoint(x,y,matchPoint)
        img2out = img2[:]
        cv2.circle(img2out,result,10,(0, 0, 255), -1)

img1 = cv2.imread('tate.PNG')
img2 = cv2.imread('yoko.PNG')

detector = cv2.AKAZE_create()
#kpは特徴的な点の位置 destは特徴を現すベクトル
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)
#特徴点の比較機
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
#割合試験を適用
good = []
match_param = 0.6
matchPoint = []
H,W,v = img1.shape

for m,n in matches:
    if m.distance < match_param*n.distance:
        p1 = tuple(map(int, kp1[m.queryIdx].pt))
        p2 = tuple(map(int, kp2[m.trainIdx].pt))
        matchPoint.append([p1,p2])
        good.append([m])

if __name__ == "__main__":
    img2 = cv2.imread('yoko.PNG')
    img1 = cv2.imread('tate.PNG')
    img2out = img2[:]
    cv2.imshow("img1",img1)
    cv2.setMouseCallback('img1', onMouse)
    count = 0
    while True:
        cv2.imshow("img1",img1)
        cv2.imshow("target",img2out)
        #if count == 30:
            #img2 = cv2.imread('yoko.PNG')
            #count = 0
        #else:
            #count += 1
        img2out = img2[:]
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()