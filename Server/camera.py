from asyncio.windows_events import NULL
from distutils.command.config import config
import pickle
import cv2
import numpy as np
import config
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import uuid
import serial 


category_index = label_map_util.create_category_index_from_labelmap("ANPR\Tensorflow\workspace\\annotations\label_map.pbtxt")

image_to_ocr = NULL

arduino = serial.Serial('COM5', 9600)
x = 0 
while True:
    if x == 0 : 
        print("dkhlt")
        x=1
    line = arduino.readline()
    str = line.decode()
    str = str.replace('\r\n', '')

    # str = input("1 or 0 (Entry or Exit)")

    if (str == "1"):
        img_counter = 0
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        image_np = NULL
        while cap.isOpened():
            
            ret, frame = cap.read()
            image_np = np.array(frame)
            cv2.waitKey(800)  # Decrease the frame rate 1000/200= 5 fps

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),dtype=tf.float32)
            detections = config.detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {
                key: value[0, :num_detections].numpy()
                for key, value in detections.items()
            }
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections[
                'detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.7,
                agnostic_mode=False)

            cv2.imshow('object detection',cv2.resize(image_np_with_detections, (600, 450)))

            if len(list(filter(lambda x: x > 0.8,detections['detection_scores']))) > 0:
                image_to_ocr = image_np_with_detections
                id = uuid.uuid1()
                cv2.imwrite('Arduino/entry_{}.png'.format(id), image_to_ocr)
                output_file = open("Arduino/entry_{}.pkl".format(id), "wb")
                pickle.dump(detections, output_file)
                output_file.close()
                
                cap.release()
                cv2.destroyAllWindows()
                break 

        #Open the door
        arduino.write("O".encode())
        print("O")

    if (str == "0"):
        img_counter = 0
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        image_np = NULL
        while cap.isOpened():
            
            ret, frame = cap.read()
            image_np = np.array(frame)
            # cv2.waitKey(800)  # Decrease the frame rate 1000/200= 5 fps

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),dtype=tf.float32)
            detections = config.detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {
                key: value[0, :num_detections].numpy()
                for key, value in detections.items()
            }
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections[
                'detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.7,
                agnostic_mode=False)

            cv2.imshow('object detection',cv2.resize(image_np_with_detections, (600, 450)))

            if len(list(filter(lambda x: x > 0.8,detections['detection_scores']))) > 0:
                image_to_ocr = image_np_with_detections
                id = uuid.uuid1()
                cv2.imwrite('Arduino/exit_{}.png'.format(id), image_to_ocr)
                output_file = open("Arduino/exit_{}.pkl".format(id), "wb")
                pickle.dump(detections, output_file)
                output_file.close()
                cap.release()
                cv2.destroyAllWindows()
                break 

        #Open the door
        print("C")
        arduino.write("C".encode())
    

    spots = {
        "spot 1" : "Empty",
        "spot 2" : "Empty"

    }

    if str == "1 : Full - 2 : Full" : 
        spots['spot 1'] = "Full"
        spots['spot 2'] = "Full"

    if str == "1 : Empty - 2 : Full" : 
        spots['spot 1'] = "Empty"
        spots['spot 2'] = "Full"

    if str == "1 : Full - 2 : Empty" : 
        spots['spot 1'] = "Full"
        spots['spot 2'] = "Empty"

    if str == "1 : Empty - 2 : Empty" : 
        spots['spot 1'] = "Empty"
        spots['spot 2'] = "Empty"


    str ="" 

    output_file = open("Arduino/spots.pkl", "wb")
    pickle.dump(spots, output_file)
