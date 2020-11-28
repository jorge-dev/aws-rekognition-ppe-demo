import sys
import pickle
import datetime
from pprint import pprint
import cv2
import boto3
import time
from multiprocessing import Pool
from random import randrange
import pytz
import traceback

# get the Client
# session = boto3.Session()
session = boto3.Session()
rekog_client = session.client("rekognition", region_name='us-east-1')

# Camera Parameters
camera_index = 0
width = 1280
height = 720
rand1 = randrange(256)
rand2 = randrange(256)
rand3 = randrange(256)
# Image details global data as currently only one face is supported
x1 = None
x2 = None
y1 = None
y2 = None
ppe = None
confidence = None
person_box_coord = []

''' 
Function to send frame to Aws recognition
frame: OpenCv/Numpy image
frame_count: variable 
enable_rekog: Variable for disabling recognition petition
'''


def send_frame(frame, frame_count, enable_rekog=False):
    try:
        _, buff = cv2.imencode(".jpg", frame)
        img_bytes = bytearray(buff)
        if enable_rekog:
            response = rekog_client.detect_protective_equipment(
                Image={
                    'Bytes': img_bytes
                },
                SummarizationAttributes={
                    'MinConfidence': 0.85,
                    'RequiredEquipmentTypes': [
                        'FACE_COVER', 'HAND_COVER', 'HEAD_COVER',
                    ]
                })
    except Exception as e:
        print("An error was found while trying to get api response :", e)
    return response


'''
Function to process the response from AWS Rekognition 
response: AWS rekognition response
'''


def is_equipment_empty(bodypart):
    if 'EquipmentDetections' in bodypart and len(bodypart['EquipmentDetections']) >= 1:
        eq = bodypart['EquipmentDetections']
    else:
        eq = ['No result']

    return eq


def extract_info_person(person):
    body_parts = person['BodyParts'] if 'BodyParts' in person else []
    return body_parts


def get_data(response):
    # Global data as currently only one face is supported
    global x1
    global x2
    global y1
    global y2
    global ppe
    global confidence
    global person_box_coord

    # Get the faces
    ppe_response = response
    person_box = []
    person_body_parts = []

    # Check if
    if 'Persons' in ppe_response and len(ppe_response['Persons']) >= 1:
        people = ppe_response['Persons']

        for person in people:
            person_box.append(person['BoundingBox'])
            person_body_parts.append(person['BodyParts'])

        print (person_body_parts, "\n")


    else:
        print('No person Found')



    # get the face bounding box
    person_box_coord = []
    for box in person_box:
        x1 = int(box['Left'] * width)
        y1 = int(box['Top'] * height)
        x2 = int(box['Left'] * width + box['Width'] * width)
        y2 = int(box['Top'] * height + box['Height'] * height)
        temp = [x1, y1, x2, y2]
        person_box_coord.append(temp)

    # print("Bounding box =  ", person_box_coord)


# '''
# Function find the emotions with max confidence
# emotions: emotion map from Aws response
# '''
#
#
# def find_emotion(emotions):
#     num = {'Confidence': 0.0, 'Type': None}
#     for item in emotions:
#         if item['Confidence'] > num['Confidence']:
#             num = item
#     return num


def main():
    # Capture set up
    capture_rate = 30
    cap = cv2.VideoCapture(0)
    pool = Pool(processes=2)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        # Resize the capture image
        frame = cv2.resize(frame, (width, height))

        # if we cant get the frame, exit
        if ret is False:
            break

        # capture rate for Aws rekognition petition
        if frame_count % capture_rate == 0:
            pool.apply_async(
                send_frame, (frame, frame_count, True), callback=get_data)

        frame_count += 1

        # if we have data, we draw it
        # print(gender)
        for x, y, w, h in person_box_coord:
            if x is not None and y is not None and w is not None and h is not None:
                cv2.rectangle(frame, (x - 3, y - 3), (w + 3, h + 3), (rand1, rand2, rand3), 1)
        #     font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        # cv2.putText(frame, gender, (x1, y1 + 20), font, .8,
        #             (255, 255, 255), 1, cv2.LINE_AA)


        # Show the frame
        cv2.imshow('Aws rekognition PPe detection demo', frame)
        # pressing q for exit
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # Release the camera and windows
    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    main()
