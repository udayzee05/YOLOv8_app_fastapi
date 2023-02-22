from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import asyncio
from aiofiles import os
import uvicorn
from ultralytics import YOLO
import cvzone
from sort import *
import math

app = FastAPI()

async def video_feed(request: Request):
    model = YOLO('../weights/yolov8s.pt')
    # Tracking
    tracker = Sort(max_age=20,min_hits=3, iou_threshold=0.3)
    totalCountUp=[]
    totalCountDown = []
    limitsUp = [103,161,296,161]
    limitsDown =[527,489,735,489]
    classNames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
    "teddy bear", "hair drier", "toothbrush"]
    mask1 = cv2.imread('mask.png')

    video_capture = cv2.VideoCapture("../videos/people.mp4")
    while True:
        try:
            ret, img = video_capture.read()
            if not ret:
                break
            imgRegion = cv2.bitwise_and(img,mask1)
            imgGraphics = cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
            img = cvzone.overlayPNG(img,imgGraphics,(730,260))
            results = model(imgRegion, stream = True)
            detections = np.empty((0,5))
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # bounding box
                    x1, y1, x2 ,y2  = box.xyxy[0]
                    x1, y1, x2 ,y2 = int(x1), int(y1), int(x2) ,int(y2)
                    w, h = x2-x1 , y2-y1
                    # confidence threshold
                    conf = math.ceil((box.conf[0]*100))/100
                    # class name
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]
                    if currentClass == 'person' and conf > 0.3:
            
                        currentArray = np.array([x1,y1,x2,y2,conf])
                        detections = np.vstack([detections,currentArray])
            resultTracker = tracker.update(detections)
            cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,0,255),5)
            cv2.line(img,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,0,255),5)

            for result in resultTracker:
                x1,y1,x2,y2,id = result
                x1, y1, x2 ,y2 = int(x1), int(y1), int(x2) ,int(y2)
                w, h = x2-x1 , y2-y1
                cvzone.cornerRect(img,(x1,y1,w,h),l = 10,rt = 2,colorR = (255,0,0))
                cvzone.putTextRect(img,f"{int(id)}",(max(0,x1),max(40,y1)), scale = 2, thickness=3,offset = 10)
                cx,cy = x1 + w//2, y1 + h//2
                cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
                if limitsUp[0]<cx<limitsUp[2] and limitsUp[1] -20 <cy < limitsUp[1]+20:
                    if totalCountUp.count(int(id))==0:
                        totalCountUp.append(int(id))
                        cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,255,0),5)

                if limitsDown[0]<cx<limitsDown[2] and limitsDown[1] -20 <cy < limitsDown[1]+20:
                    if totalCountDown.count(int(id))==0:
                        totalCountDown.append(int(id))
                        cv2.line(img,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,255,0),5)

            # cvzone.putTextRect(img,f"Count : {len(totalCounts)}",(50,50)) 
            cv2.putText(img,str(len(totalCountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)  
            cv2.putText(img,str(len(totalCountDown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)  

            _, buffer = cv2.imencode('.jpg', img)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(0.05)
        except:
            pass
    video_capture.release()
    await os.remove(".jpg")

@app.get("/video_feed")
async def video_feed_route(request: Request):
    return StreamingResponse(video_feed(request), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/video_feed", response_class=HTMLResponse)
async def index():
    with open("index.html") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
