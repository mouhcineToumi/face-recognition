import numpy as np
import cv2
import os


class FaceDetector:
    def __init__(self, confid = 0.5, prototxt="deploy.prototxt.txt", model="res10_300x300_ssd_iter_140000.caffemodel"):
        self.confid = 0.5
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)


    def get_face2(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()
        m = 0;
        # loop over the detections
        for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.confid:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                #cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
                #cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                
                #only get the most accurate one
                if( confidence > m):
                    face = image[startY:endY, startX:endX]
                    m = confidence
        
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


    def get_face(self, image_path):
        #load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()
        m = 0;
        # loop over the detections
        for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.confid:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                #cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
                #cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                
                #only get the most accurate one
                if( confidence > m):
                    face = image[startY:endY, startX:endX]
                    m = confidence
        
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


