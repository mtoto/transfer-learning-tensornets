#!/usr/bin/env python3
import numpy as np
import argparse
import cv2

def test_videocam(cascade_file, pb_file):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cascade_file)
    net = cv2.dnn.readNetFromTensorflow(pb_file)
    emotions = {0: "Angry",1: "Disgust",2: "Fear",3: "Happy",4: "Sad",5: "Suprise",6: "Neutral"}

    while(True):
        # Capture frame-by-frame
        _, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            detected_face = gray[int(y):int(y+h), int(x):int(x+w)]# crop detected face
            detected_face = cv2.resize(detected_face, (48, 48)) # resize to 48*48 first
            detected_face = cv2.resize(detected_face, (224, 224)) # resize to 224*224 
            grayd = cv2.cvtColor(detected_face, cv2.COLOR_GRAY2RBG)
            im = np.expand_dims(grayd, axis=0)

            net.setInput(im.transpose(0, 3, 1, 2), name = "IteratorGetNext")
            dnn_out = net.forward("densenet169/probs")
            
            max_index = np.argmax(dnn_out[0])
            emotion = emotions[max_index]
            
            cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--cascade-file',
      help='Path to cascade file.',
      type=str,
      required=True
  )
  parser.add_argument(
      '--pb-file',
      help='Path to frozen model.',
      type=str,
      required=True
  )
  args = parser.parse_args()
  test_videocam(**args.__dict__)