# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# yolov5 + deep sort
import sys
sys.path.insert(0, './yolov5')

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

import argparse
import threading
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import random
import math

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from customerObject import Customer
from dataSender import ThreadedClient
from dataServer import ServerSocket
from reid import MultiReID

COUNT_THRESHOLD = 2
led_order = [0, 4, 4, 3, 2, 3, 4, 4, 1, 2, 1] #아직 임의의 수
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

class ImageInfo:
    def __init__(self, shape):
        self.shape = shape
        self.mask = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)

        # crop box
        self.ROI = []
        self.in_line = []
        self.out_line = []
        self.case = 0
        self.IsSelected = False
        self.det_ROI = []
        
        self.n_customer = 0 ## 현재 방문자 수
        self.n_visited = 0 ## 현재까지의 방문자 수
        self.dwell_time = 0 ## 체류 시간
        self.CustomerList = {} ## 현재 방문자 정보
        self.visitedCustomers = {} ## 방문했던 id 목록
        self.remainedCustomers = {} ## 현재 방문자 id 목록
        self.timeRecord = [] ## 시간 기록
        
        self.count_in = 0
        self.count_out = 0
        self.prev_count = 0
        self.productNum = 0
        
        self.exist_ids = {}
        
    def update_info(self, Customers):
        for customer in Customers:
            id = customer.getID()
            local_id = customer.getLocalID() * -1
            [px, py] = customer.getCentralPoint()
            
            if (id > 0) and (self.CustomerList.get(id, False) == False) and (self.CustomerList.get(local_id, False) != False): # global ID가 있는데 global ID는 없이 local ID가 등록된 경우
                customer.merge(self.CustomerList[local_id]) # 정보를 합치고
                del self.CustomerList[local_id] # 기존 걸 지우고
                self.CustomerList[id] = customer # 저장
                if customer.getVisitTime() != None:
                    self.visitedCustomers[id] = True
                    del self.visitedCustomers[local_id] 
                if customer.__in__:
                    self.remainedCustomers[id] = True
                    del self.remainedCustomers[local_id]
                    
            if (self.CustomerList.get(id, False) == False) and (self.CustomerList.get(local_id, False) == False): # 둘 다 등록이 안된 경우 (possible?)
                self.CustomerList[id] = customer 
            else:
                [prev_x, prev_y] =  self.CustomerList[id].central
                self.CustomerList[id].move(px, py) # 방향을 계산한다
                color = [255, 255, 255]
                if id > 0:
                    color = compute_color_for_labels(id)
                if math.dist([prev_x, prev_y], [px, py]) < 20:
                    self.mask = cv2.line(self.mask, (prev_x, prev_y), (px, py), color, 1) # draw trajectroy
                    self.mask = cv2.arrowedLine(self.mask, (prev_x, prev_y), (px, py), color, 1, 8, 0, 0.3)
            
            n_customer = 0
            # check in 본격적인 방문자 등록
            if self.__check_in(px, py): # 영역 안에 있는 경우
                n_customer = n_customer + 1
                if self.visitedCustomers.get(id, False) != True: # 새로 온 사람이다
                    self.visitedCustomers[id] = True # id를 등록하고
                    self.CustomerList[id].visit(time.time()) # 방문 시간 기록
                    self.remainedCustomers[id] = True # 현재 있다고 적고
                    self.__check_inout_line(customer, 0) # 입구로 들어왔는지 확인하자                
            else: # 영역 밖에 있는 경우
                if self.remainedCustomers.get(id, False) != False: # 방금 나간 건지 체크하자
                    in_time = self.CustomerList[id].getVisitTime() # 몇시에 방문했는지 보고
                    out_time = time.time() # 나간 시간을 기록해서
                    self.CustomerList[id].leave(out_time) # 떠난 시간 기록
                    self.__check_inout_line(customer, 1) # 출구로 나간건지 확인하자
                    del self.remainedCustomers[id] # 체류자 목록에서 제거
                    self.timeRecord.append(out_time - in_time) # 체류 시간을 기록
        
        self.n_customer = n_customer
        self.n_visited = len(self.visitedCustomers)
    
    def __check_in(self, x, y):
        # (ROI[-2][0], ROI[-2][1]), (ROI[-1][0], ROI[-1][1])
        if self.IsSelected:
            if x >= min(self.ROI[-2][0], self.ROI[-1][0]) and x <= max(self.ROI[-2][0], self.ROI[-1][0]) and \
                            y >= min(self.ROI[-2][1], self.ROI[-1][1]) and y <= max(self.ROI[-2][1], self.ROI[-1][1]):
                return True
        return False
    
    def __check_inout_line(self, customer, type=0):
        """Check Line In & Out

        Args:
            customer ([CustomerObject])
            type (int, optional): If entrance to 0, exit to 1. Defaults to 0.

        Returns:
            int, int: In count, Out count
        """
        [px, py] = customer.getPassPoint()
        count_in = 0
        count_out=  0
        
        if type == 0:
            if self.case == 1 or self.case == 2:
                if abs(py - self.in_line[0][1]) <= 10 and px >= self.in_line[0][0] and px <= self.in_line[1][0]:
                    count_in = 1
            elif self.case == 3 or self.case == 4:
                if abs(px - self.in_line[0][0]) <= 10 and py >= self.in_line[0][1] and py <= self.in_line[1][1]:
                    count_in = 1
        elif type == 1:
            if self.case == 1 or self.case == 2:
                if abs(py - self.out_line[0][1]) <= 10 and px >= self.out_line[0][0] and px <= self.out_line[1][0]:
                    count_out = 1
            elif self.case == 3 or self.case == 4:
                if abs(px - self.out_line[0][0]) <= 10 and py >= self.out_line[0][1] and py <= self.out_line[1][1]:
                    count_out = 1
            
        self.count_in = self.count_in + count_in
        self.count_out = self.count_out + count_out
        
    def draw_line(self, img):
        if self.ROI[-2][1] <= self.ROI[-1][1]:
            # case up&down
            if self.ROI[-2][0] <= self.ROI[-1][0]:
                # down
                self.case = 1
                self.in_line = [[self.ROI[-2][0], self.ROI[-2][1]], [self.ROI[-1][0], self.ROI[-2][1]]]
                self.out_line = [[self.ROI[-2][0], self.ROI[-1][1]], [self.ROI[-1][0], self.ROI[-1][1]]]
            else:
                # up
                self.case = 2
                self.in_line = [[self.ROI[-1][0], self.ROI[-1][1]], [self.ROI[-2][0], self.ROI[-1][1]]]
                self.out_line = [[self.ROI[-1][0], self.ROI[-2][1]], [self.ROI[-2][0], self.ROI[-2][1]]]
        else:
            # case right&left
            if self.ROI[-2][0] <= self.ROI[-1][0]:
                # right
                self.case = 3
                self.in_line = [[self.ROI[-2][0], self.ROI[-1][1]], [self.ROI[-2][0], self.ROI[-2][1]]]
                self.out_line = [[self.ROI[-1][0], self.ROI[-1][1]], [self.ROI[-1][0], self.ROI[-2][1]]]
            else:
                #left
                self.case = 4
                self.in_line = [[self.ROI[-2][0], self.ROI[-1][1]], [self.ROI[-2][0], self.ROI[-2][1]]]
                self.out_line = [[self.ROI[-1][0], self.ROI[-1][1]], [self.ROI[-1][0], self.ROI[-2][1]]]
                
        cv2.line(img, (self.in_line[0][0], self.in_line[0][1]), (self.in_line[1][0], self.in_line[1][1]), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(img, (self.out_line[0][0], self.out_line[0][1]), (self.out_line[1][0], self.out_line[1][1]), (255, 0, 0), 2, cv2.LINE_AA)
        
    def draw_ROI(self, img):
        
        cv2.rectangle(img, (self.ROI[-2][0], self.ROI[-2][1]), (self.ROI[-1][0], self.ROI[-1][1]), (36, 255, 12), 1)
        
        if len(self.det_ROI):
            cv2.rectangle(img, (self.det_ROI[-2][0], self.det_ROI[-2][1]), (self.det_ROI[-1][0], self.det_ROI[-1][1]), (255, 255, 255), 1)
        
        label = "COUNT: " + str(self.n_customer) + "\nVISITED: " + str(self.n_visited) + "\nIN: %d\nOUT: %d" % (self.count_in, self.count_out)
        labelsize = []
        

        for i, line in enumerate(label.split('\n')):
            labelsize.append(cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, 0.5, 2)[0])
        # cv2.rectangle(img, (ROI[-1][0], ROI[-2][1]),
        #                 (ROI[-1][0] + max(labelsize[0][0], labelsize[1][0]) + 4, 
        #                 ROI[-2][1] + labelsize[0][1] + labelsize[1][1] + 25), (0,0, 128), -1)
        
        x, y, w, h = 10, 10, 170, 100
        sub_img = img[y:y+h, x:x+w]
        res_box = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.5, res_box, 0.5, 1.0)
        img[y:y+h, x:x+w] = res
        
        for i, line in enumerate(label.split('\n')):
            text_dy = 20
            # cv2.putText(img, line, (ROI[-1][0] + 2, ROI[-2][1] + labelsize[i][1] + 3 + i*text_dy), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255], 1)
            cv2.putText(img, line, (x + 2, y + labelsize[i][1] + 3 + i*text_dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1)
        
        if self.dwell_time != 0:
            timetext = "Dwell Time: " + str(int(self.dwell_time)) + "s"
            # cv2.putText(img, timetext, (ROI[-1][0] + 2, ROI[-2][1] + labelsize[0][1]*3 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255], 1)
            cv2.putText(img, timetext, (x + 2, y + labelsize[0][1]*5 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1)

    def reset(self):
        self.n_customer = 0
        self.n_visited = 0
        self.dwell_time = 0
        self.count_in = 0
        self.count_out = 0
        self.visitedCustomers.clear()
        self.remainedCustomers.clear()
        self.CustomerList.clear()
        self.timeRecord.clear()
        self.ROI.clear()
        self.det_ROI.clear()
        self.in_line.clear()
        self.out_line.clear()
        self.case = 0
        self.IsSelected = False
        
        self.mask = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
        
def getProduct(gender, age, count):
    n = 0
    if gender != -1:
        n = gender * 5 + age
    else:
        n = random.randint(1, 10)
    if count >= COUNT_THRESHOLD:
        while True:
            nn = random.randint(1, 10)
            if led_order[nn] != led_order[n]:
                n = nn
                break
    return n

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, box, id=None, offset=(0, 0), foot=True):
    x1, y1, x2, y2 = [int(i) for i in box]
    x1 += offset[0]
    x2 += offset[0]
    y1 += offset[1]
    y2 += offset[1]
    
    if foot:
        py = int(y2)
    else:
        py = int((y1 + y2)/2)
        
    # person point (p.p.)
    px = int((x1 + x2)/2)
    
    # box text and bar
    id = int(id) if id is not None else 0
    
    if id <= 0:
        color = [0, 0, 0]
    else:
        color = compute_color_for_labels(id)
        
    label = '{}{:d}'.format("", id)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(
        img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
    cv2.putText(img, label, (x1, y1 +
                                t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1.5, [255, 255, 255], 2)
    
    cv2.circle(img, (px, py), 3, color, -1)
    # cv2.putText(img, "(%d, %d)"%(px,py), (px, py), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    # coord = "(%d %d) " % (px, py)
    # cv2.putText(img, coord, (px, py), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    
    return img, px, py

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        sbox = [x, y]
        param.ROI.append(sbox)
        print('ButtonDownEvent:', sbox)
        param.IsSelected = False
    elif event == cv2.EVENT_LBUTTONUP:
        ebox = [x, y]
        param.ROI.append(ebox)
        print('ButtonUpEvent:', ebox)
        print(param.ROI)
        param.IsSelected = True
    # elif event == cv2.EVENT_RBUTTONUP:
    #     param.ROI.clear()
    #     param.IsSelected = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        sbox = [x, y]
        param.det_ROI.append(sbox)
    elif event == cv2.EVENT_RBUTTONUP:
        sbox = [x, y]
        param.det_ROI.append(sbox)

def RepresentsInt(s):
    try: 
        int(s)
        return s
    except ValueError:
        return False

def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
        project, exist_ok, update, save_crop, network = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop, opt.network
        
    webcam = source == '0' or '1' or '2' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
 
    if network:
        # initialize socket
        client = ThreadedClient()
        client.start_listen()
        server = ServerSocket()
        play_trig = -1
        count_time = 0

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    if type(yolo_model) is str:  # single yolo model
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        exp_name = yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, rotate=opt.rotate)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    image_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
        image_list.append(ImageInfo(dataset.imgs[i].shape))
    outputs = [None] * nr_sources
    features = [None] * nr_sources
    REID = MultiReID(nr_sources, dist_thresh=opt.reid_thres)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    # t0 = time.time()

    save_path = str(Path(out))
    ## extract what is in between the last '/' and last '.'
    # txt_file_name = source.split('/')[-1].split('.')[0]
    # txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
    # save_path = str(Path(out))
    # txt_path = str(Path(out)) + '/results' + time.strftime('%y%m%d_%H%M%S', time.localtime(time.time())) +'.txt'
    # log_path = str(Path(out)) + '/log_' + time.strftime('%y%m%d_%H%M%S', time.localtime(time.time())) +'.txt'
    
    
    # if opt.rotate:
    #     mask = np.zeros((dataset.imgs[0].shape[1], dataset.imgs[0].shape[0], 3), dtype=np.uint8)
    # else:

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                log_file_name = 'log' + p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    log_file_name = 'log' + p.parent.name
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            log_path = str(save_dir / 'tracks' / log_file_name)
            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            
            # if  frame_idx != 0 and frame_idx % 10000 == 0 and i == 0:
            #     last_global = REID.get_last_global_id()
            #     REID = MultiReID(nr_sources, dist_thresh=opt.reid_thres, init_global=last_global + 1)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs[i], features[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    # multi-camera fused reid
                    new_ID = REID.update(i, outputs[i], features[i], image_list[i].exist_ids, image_list[i].det_ROI)
                    # customers in frame
                    Customers = []
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4]
                        local_id = output[4]
                        id = new_ID[j]
                        cls = output[5]
                        
                        if image_list[i].exist_ids.get(local_id, -1) == -1:
                            image_list[i].exist_ids[local_id] = 1
                        else:
                            image_list[i].exist_ids[local_id] += 1
                        
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, local_id,
                                                               bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, i))
                            
                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            _, px, py = draw_boxes(im0, bboxes, id, (0, 0), opt.foot)
                            # label = f'{id} {names[c]} {conf:.2f}'
                            # annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                        
                        Customers.append(Customer(id, local_id, px, py))

                    image_list[i].update_info(Customers)

                    # Write MOT compliant results to file
                    if save_txt:
                        with open(log_path + '.txt', 'a') as f:
                            line = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + \
                                " count %d total %d dwell_time %f in %d out %d\n" % \
                                    (image_list[i].n_customer, image_list[i].n_visited, image_list[i].dwell_time, image_list[i].count_in, image_list[i].count_out)
                            f.write(line)
            else:
                deepsort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            if len(image_list[i].timeRecord) != 0:
                image_list[i].dwell_time = sum(image_list[i].timeRecord)/len(image_list[i].timeRecord)
            
            
            ### Per frame
            # client.add_message(str(n_customer))

            # Stream results
            # im0 = annotator.result()
            if show_vid:
                if image_list[i].IsSelected:
                    # visualize result of count
                    image_list[i].draw_ROI(im0)
                    image_list[i].draw_line(im0)
                im0 = cv2.addWeighted(im0, 1.0, image_list[i].mask, 0.5, 0)
                cv2.imshow(str(p), im0)

                ## resized window
                # dst = cv2.resize(im0, dsize=(0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                # cv2.imshow("resized", dst)

                cv2.setMouseCallback(str(p), on_mouse, image_list[i])
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    # raise StopIteration
                    quit()

                if cv2.waitKey(1) == ord('r'):  # r to reset
                    for info in image_list:
                        info.reset()
                    # if opt.rotate:
                    #     mask = np.zeros((dataset.imgs[0].shape[1], dataset.imgs[0].shape[0], 3), dtype=np.uint8)
                    # else:
                

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
                
            if network:
                if image_list[0].n_customer >= 1:
                    if prev_count == 0 or (prev_count != image_list[0].n_customer and play_trig == 1):
                        gender, age = server.getRecentRecogResult()
                        productNum = getProduct(gender, age, image_list[0].n_customer)
                        client.add_led_message(str(led_order[productNum]))
                        msg = 'play/' + str(productNum)
                        client.add_message(msg)
                        prev_count = image_list[0].n_customer
                        play_trig = 1
                        
                elif play_trig == 1:
                    client.add_message('stop/')
                    client.add_led_message('0')
                    play_trig = -1
                    prev_count = 0
                else:
                    play_trig = -1
                    prev_count = 0
                    
                
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_model)  # update model (to fix SourceChangeWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='crowdhuman_yolov5m.pt', help='model.pt path(s)')
    # parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_5_market1501')
    parser.add_argument('--source', type=str, default='streams.txt', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--reid-thres', type=float, default=0.16, help='person reidentification threshold') # 0.16
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', default=0, nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--config_alphapose", type=str, default="alphapose/configs/alphapose.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    # rotate 90
    parser.add_argument("--rotate", default=False, type=bool, help="rotate video 90")
    # foot tracker
    parser.add_argument("--foot", default=True, type=bool, help='track foot position')
    # network
    parser.add_argument("--network", default=False, type=bool, help='open client socket')
    # args = parser.parse_args()
    # args.img_size = check_img_size(args.img_size)
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
