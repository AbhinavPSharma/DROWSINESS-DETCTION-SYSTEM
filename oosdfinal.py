from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import os
import pyrebase
import sys
import urllib.request
from playsound import playsound

class Sound:
	def __init__(self):
		self.file="C:/Users/abhin/OOSD/Buzze.mp3"
	def play(self):
		playsound(self.file)

class Model:
	def __init__(self,m):
		self.modelP=m
	def eye_aspect_ratio(self,eye):
		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])
		C = dist.euclidean(eye[0], eye[3])
		ear = (A + B) / (2.0 * C)
		return ear
	def predictINIT(self):
		print("[INFO] loading facial landmark predictor...")
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(self.modelP)
		(self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
		

class predictor:
	def predict(self,detector,predictor,lStart,lEnd,rStart,rEnd,EYE_AR_THRESH,EYE_AR_CONSEC_FRAMES,url):
		self.COUNTER = 0
		self.ALARM_ON = False
		modelPr=Model(1)
		print("[INFO] starting video stream thread...")
		while True:
			frame = urllib.request.urlopen(url)
			frame = np.array(bytearray(frame.read()),dtype=np.uint8)
			frame=cv2.imdecode(frame,-1)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			rects = detector(gray, 0)

			for rect in rects:
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = modelPr.eye_aspect_ratio(leftEye)
				rightEAR = modelPr.eye_aspect_ratio(rightEye)
				ear = (leftEAR + rightEAR) / 2.0
				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
				if ear < EYE_AR_THRESH:
					self.COUNTER += 1
					if self.COUNTER >= EYE_AR_CONSEC_FRAMES:
						if not self.ALARM_ON:
							self.ALARM_ON = True
						cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				else:
					self.COUNTER = 0
					self.ALARM_ON = False
				cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				if(self.ALARM_ON==True):
					s= Sound()
					s.play()
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
	

class Entry:
	def __init__(self):
		self.url="http://"
		self.EYE_AR_THRESH = 0.3
		self.EYE_AR_CONSEC_FRAMES = 20
		self.modelP=None
	def inp(self):
		print("Enter the camera Server Url")
		self.url=self.url+str(input())+"/shot.jpg"
		self.modelP="shape_predictor_face_landmarks.dat"

class Main():
	def __init__(self):
		in1 = Entry()
		in1.inp()
		m1 = Model(in1.modelP)
		m1.predictINIT()
		p =predictor()
		p.predict(m1.detector,m1.predictor,m1.lStart,m1.lEnd,m1.rStart,m1.rEnd,in1.EYE_AR_THRESH,in1.EYE_AR_CONSEC_FRAMES,in1.url)
		cv2.destroyAllWindows()

m = Main()