import cv2
import numpy as np

#load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
#determine only the (output) layer names that we need from YOLO
layer_names = net.getLayerNames()
output_layers=[layer_names[i-1] for i in net.getUnconnectedOutLayers()]
#initialize a list of colors to represent each possible class label
colors = np.random.uniform(0,255,size=(len(classes),3))
#set the font type
font = cv2.FONT_HERSHEY_PLAIN

#Loading the video
cap = cv2.VideoCapture("cars_-_1900 (Original).mp4") #0 for the webcam

while True:
    _,frame = cap.read()

    height, width, channels = frame.shape

    #construct a blob from the input image and then perform a forward
    #pass of the YOLO object detector, giving us our bounding boxes and
    #associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    class_ids = []
    confidences = []
    boxes = []
    # loop over each of the layer outputs
    for out in outs:
    # loop over each of the detections
        for detection in out:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                #box = detection[0:4] * np.array([W, H, W, H])
			    #(centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(center_x-w/2)
                y = int(center_y-h/2)

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # ensure at least one detection exists
    for i in range(len(boxes)):
    # loop over the indexes we are keeping
        if i in indexes:
            # extract the bounding box coordinates
            x,y,w,h = boxes[i]
            confid = round(confidences[i],2)
            #x, y) = (boxes[i][0], boxes[i][1])
		    #(w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(confid) , (x, y + 30), font, 2, color, 2)
    # Display the resulting frame
    # show the image
    cv2.imshow('img', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()