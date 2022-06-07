from datetime import datetime
import os
import pickle
import time
import cv2
import numpy as np
import pyrebase
import config

firebaseConfig = {
  "apiKey": "AIzaSyChD8WS1JORFnE68O5cMFi7ikX_NNDYLVs",
  "authDomain": "park-63a8f.firebaseapp.com",
  "databaseURL": "https://park-63a8f-default-rtdb.firebaseio.com",
  "projectId": "park-63a8f",
  "storageBucket": "park-63a8f.appspot.com",
  "messagingSenderId": "1008052510556",
  "appId": "1:1008052510556:web:4f1a71d52d02d1f1ad2b99",
  "measurementId": "G-FR3Y59Q2WT"
}

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()


def convert_date(timestamp):
    d = datetime.utcfromtimestamp(timestamp)
    formated_date = d.strftime('%d-%m-%Y, %H:%M:%S')
    return formated_date


while True:
    time.sleep(5)
    entries = os.listdir("Arduino")
    #Get All the png files.
    images = list(filter(lambda x: '.png' in x, entries))
    # treated images marked by __r in the end 
    images = list(filter(lambda x: '__r' not in x, images))
    entries = list(filter(lambda x: 'entry_' in x, images))
    exits = list(filter(lambda x: 'exit_' in x, images))
    images = [*entries, *exits]
    print(images)
    if os.path.getsize("Arduino\spots.pkl") > 0: 
        spots = open("Arduino\spots.pkl", "rb")
        spots = pickle.load(spots)
        db.child("spots").set(spots)

    if len(images) > 0:
        for entry in images:

            file_name = entry.replace('.png', '')
            date = convert_date(os.path.getmtime('Arduino\\' + file_name + ".png"))
            detections_file = open("Arduino\\" + file_name + ".pkl", "rb")
            detections = pickle.load(detections_file)

            img = cv2.imread('Arduino\\' + file_name + ".png")
            image_np_with_detections = np.array(img)

            license_plate, region = config.ocr_it(image_np_with_detections,detections, 0.45, 0.7)

            data = {
                "file_name": file_name,
                "license_plate": license_plate,
                "date": date,
                "action": list(file_name.split("_"))[0]  # Or Exit
            }
            db.child("licenses").child(file_name).set(data)
            os.rename("Arduino\\" + file_name + ".png",
                      "Arduino\\" + file_name + "__r" + ".png")

            images.remove(entry)