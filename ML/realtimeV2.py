from keras.models import load_model
from imutils.video import VideoStream
from collections import deque
import numpy as np
import argparse
import time
import pickle
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
ap.add_argument("-i", "--input", required=False,
	help="path to our input video")
ap.add_argument("-o", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args('-m model -l label-bin -o output'.split()))

print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

print("[INFO] starting video stream...")
vs = cv2.VideoCapture('demo.mp4')
writer = None
time.sleep(2.0)
#-------------------
(W, H) = (None, None)
prelabel = ''
prelabel = ''
ok = 'Normal'
fi_label = []
framecount = 0
while True:
	flag,frame = vs.read()

	if W is None or H is None:
		(H, W) = frame.shape[:2]
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype("float32")
	frame -= mean

	preds = model.predict(np.expand_dims(frame, axis=0))[0]

	prediction = preds.argmax(axis=0)
	Q.append(preds)

	results = np.array(Q).mean(axis=0)
	print('Results = ', results)
	maxprob = np.max(results)
	print('Maximun Probability = ', maxprob)
	i = np.argmax(results)
	label = lb[i]
	rest = 1 - maxprob
    
	diff = (maxprob) - (rest)
	print('Difference of prob ', diff)
	th = 100
	if diff > .80:
		th = diff
      
        
        
        
	if (preds[prediction]) < th:
		text = "Alert : {} - {:.2f}%".format((ok), 100 - (maxprob * 100))
		cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
	else:
		fi_label = np.append(fi_label, label)
		text = "Alert : {} - {:.2f}%".format((label), maxprob * 100)
		cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5) 

		prelabel = label

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	writer.write(output)

	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break
print("[INFO] cleaning up...")
writer.release()
vs.stream.release()
