import os
import sys
import cv2
import json
import numpy as np
import urllib.request

# load the labels
coco91_labels = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# this will help us create a different color for each class
coco91_colors = np.random.uniform(0, 255, size=(len(coco91_labels), 3))


# this is required if you didn't load the model through hub before
yolov5_path = f'{os.environ["HOME"]}/.cache/torch/hub/ultralytics_yolov5_master/'
if os.path.exists(yolov5_path):
    if yolov5_path not in sys.path: sys.path.insert(0, yolov5_path)
    # imports from Yolov5 --> You need to install requirements.txt from
    # https://pytorch.org/hub/ultralytics_yolov5/
    try:
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_coords
        from utils.plots import Annotator, colors
    except Exception as e:
        pass

def preprocess_coco(img, img_size=(640,640), disable_letterbox=False, keep_aspect=True):
    x = img
    # preprocessing based on AutoShape's forward() call https://github.com/ultralytics/yolov5/blob/master/models/common.py#L560
    if disable_letterbox:
        if keep_aspect:
            h,w,c=img.shape
            if h!=w: # squared - make it square to avoid distortions
                max_side = max(h,w)
                new_img = np.zeros((max_side,max_side,c), dtype=np.uint8)
                new_img[0:h, 0:w] = img[:]
                x = new_img
        x = cv2.resize(x, img_size)
    else:
        x = letterbox(x[...,::-1], new_shape=img_size, auto=False)[0] # letterbox to the customer's desired image size
    x = np.expand_dims(x, 0)
    x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
    x = (x / 255).astype(np.float32)
    return x     


def postprocess_yolov5(predictions, raw_img, size=(640,640)):
    labels = load_coco80_labels()
    det = non_max_suppression(predictions)[0]
    img = raw_img.copy()
    annotator = Annotator(img, line_width=3, example=str(labels))
    det[:, :4] = scale_coords(size, det[:, :4], img.shape).round()
    # Write results
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)  # integer class        
        label = f'{labels[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
def preprocess_imagenet(img, chw=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], img_size=(224,224), make_square=True):
    x = img.copy()
    h,w,c = x.shape
    if make_square:
        max_side = 0
        if h!=w: # squared - make it square to avoid distortions
            max_side = max(h,w)
            new_img = np.zeros((max_side,max_side,c), dtype=np.uint8)
            new_img[0:h, 0:w] = x[:]
            x = new_img
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, img_size)
    x = (x / 255)
    x = (x - mean) / std # normalize
    if chw: x = x.transpose(2,0,1) # HWC to CHW    
    x = np.expand_dims(x, 0).astype(np.float32)
    return x

def draw_boxes(boxes, classes, labels, image, img_size=300):
    """
    Draws the bounding box around a detected object.
    """
    h,w,c = image.shape
    max_side = max(h,w)
    fx,fy = max_side/img_size,max_side/img_size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        box = (box.astype(np.float32) * (fx,fy,fx,fy)).astype(np.int32)
        #box += (0,-r,0,r)
        color = coco91_colors[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, w//150
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, int(0.5*fx), color, int(1*fx), 
                    lineType=cv2.LINE_AA)
    return image

def load_sample_imgA():
    if not os.path.exists("goldfish.jpg"):
        urllib.request.urlretrieve("https://images.unsplash.com/photo-1579161256825-57ba3094f057", "goldfish.jpg")
    return cv2.imread("goldfish.jpg")

def load_sample_imgB():
    if not os.path.exists("zidane.jpg"):
        urllib.request.urlretrieve("https://ultralytics.com/images/zidane.jpg", "zidane.jpg")
    return cv2.imread("zidane.jpg")

def load_sample_imgC():
    if not os.path.exists("man_walking.jpg"):
        urllib.request.urlretrieve("https://images.unsplash.com/photo-1611324204543-ecc01e953173", "man_walking.jpg")
        #urllib.request.urlretrieve("https://raw.githubusercontent.com/samir-souza/laboratory/master/06_PoseEstimation/man_walking.jpg", "man_walking.jpg")
    return cv2.imread("man_walking.jpg")

def load_imagenet1k_labels():
    # Download the labels for Imagenet 1k
    if not os.path.exists("imagenet1k.json"):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json",
            "imagenet1k.json"
        )
    return json.loads(open('imagenet1k.json', 'r').read())

def load_coco80_labels():
    # load the labels
    if not os.path.exists("coco80.json"):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt",
            "coco80.labels"
        )
    labels = open('coco80.labels', 'r').readlines()
    return [l.strip() for l in labels]
