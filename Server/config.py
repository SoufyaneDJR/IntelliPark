import easyocr
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder
import cv2 
import numpy as np
import time 
import uuid 
import csv
import os

configs = config_util.get_configs_from_pipeline_file("ANPR\\Tensorflow\\workspace\\models\\my_ssd_mobnet\\pipeline.config")
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore("ANPR\\Tensorflow\\workspace\\models\\my_ssd_mobnet\\ckpt-5").expect_partial()

category_index = label_map_util.create_category_index_from_labelmap("ANPR\Tensorflow\workspace\\annotations\label_map.pbtxt")


detection_threshold = 0.6
region_threshold = 0.6

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
   
def filter_text(region, ocr_result,region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    plate = []
    
    for result in ocr_result: 
        lenght = np.sum(np.subtract(result[0][1],result[0][0]))
        height = np.sum(np.subtract(result[0][2],result[0][1]))
        
        if lenght*height /rectangle_size > region_threshold :
            plate.append(result[1])
    
    return plate

# Final Ocr Function 
def ocr_it(image_np_with_detections,detections,detection_threshold,region_threshold):
    image = image_np_with_detections
    scores = list(filter(lambda x : x> detection_threshold,detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    width = image.shape[1]
    height = image.shape[0]

    for idx, box in enumerate(boxes):
        #Croping the ROI
        roi = box*[height,width,height,width]
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        cv2.imwrite('Arduino/region.png', region)
        # Applying OCR
        reader = easyocr.Reader(['fr'])
        ocr_result = reader.readtext(region)
        print(ocr_result)

        text = filter_text(region,ocr_result,0)

        plate= ""
        for idx,x in enumerate(text):
            if idx== 0 : 
                plate = x
            else : 
                plate= plate +" | "+ x

        return plate,region 

def predict(image) : 
    IMAGE_PATH = 'ANPR\Tensorflow\workspace\images\\test' + image
    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=0.8,
            agnostic_mode=False)

    text,region = ocr_it(image_np_with_detections,detections,detection_threshold,region_threshold)

    return text , region 

def save_results(text,region,csv_filename,folder_path):
    
    img_name = '{}.jpg'.format(uuid.uuid1())

    time_string = time.strftime("%m/%d/%Y, %H:%M:%S",time.localtime())
    
    cv2.imwrite(os.path.join(folder_path,img_name),region)
    
    with open(csv_filename,mode='a',newline='') as f:
        csv_writer = csv.writer(f,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name,time_string,text])

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def real_time_detection(image_np) : 
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

    try : 
        text, region = ocr_it(image_np_with_detections,detections,detection_threshold,region_threshold)
        save_results(text,region,'detection_results.csv','Detection_Images')
        return text
    except: 
        pass
    
    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
