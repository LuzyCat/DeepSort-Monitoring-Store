# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
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

from customerObject import Customer
from dataSender import ThreadedClient

COUNT_THRESHOLD = 2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
# crop box
ROI = []
in_line = []
out_line = []
case = 0
IsSelected = False
n_customer = 0

def check_in(x, y):
    # (ROI[-2][0], ROI[-2][1]), (ROI[-1][0], ROI[-1][1])
    
    if IsSelected:
        if x >= min(ROI[-2][0], ROI[-1][0]) and x <= max(ROI[-2][0], ROI[-1][0]) and y >= min(ROI[-2][1], ROI[-1][1]) and y <= max(ROI[-2][1], ROI[-1][1]):
            return True
    
    return False

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

def draw_boxes(img, bbox, identities=None, offset=(0, 0), foot=True):
    Customers = []
    for i, box in enumerate(bbox):
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
        id = int(identities[i]) if identities is not None else 0
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
        Customers.append(Customer(id, px, py))
        # coord = "(%d %d) " % (px, py)
        # cv2.putText(img, coord, (px, py), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img, Customers

def draw_line(img):
    global case
    global in_line, out_line
    if ROI[-2][1] <= ROI[-1][1]:
        # case up&down
        if ROI[-2][0] <= ROI[-1][0]:
            # down
            case = 1
            in_line = [[ROI[-2][0], ROI[-2][1]], [ROI[-1][0], ROI[-2][1]]]
            out_line = [[ROI[-2][0], ROI[-1][1]], [ROI[-1][0], ROI[-1][1]]]
        else:
            # up
            case = 2
            in_line = [[ROI[-1][0], ROI[-1][1]], [ROI[-2][0], ROI[-1][1]]]
            out_line = [[ROI[-1][0], ROI[-2][1]], [ROI[-2][0], ROI[-2][1]]]
    else:
        # case right&left
        if ROI[-2][0] <= ROI[-1][0]:
            # right
            case = 3
            in_line = [[ROI[-2][0], ROI[-1][1]], [ROI[-2][0], ROI[-2][1]]]
            out_line = [[ROI[-1][0], ROI[-1][1]], [ROI[-1][0], ROI[-2][1]]]
        else:
            #left
            case = 4
            in_line = [[ROI[-2][0], ROI[-1][1]], [ROI[-2][0], ROI[-2][1]]]
            out_line = [[ROI[-1][0], ROI[-1][1]], [ROI[-1][0], ROI[-2][1]]]
            
    cv2.line(img, (in_line[0][0], in_line[0][1]), (in_line[1][0], in_line[1][1]), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(img, (out_line[0][0], out_line[0][1]), (out_line[1][0], out_line[1][1]), (255, 0, 0), 2, cv2.LINE_AA)
    
def check_inout_line(customer, type=0):
    """Check Line In & Out

    Args:
        customer ([CustomerObject])
        type (int, optional): If entrance to 0, exit to 1. Defaults to 0.

    Returns:
        int, int: In count, Out count
    """
    [px, py] = customer.central
    ud = customer.getDirection_ud()
    rl = customer.getDirection_rl()
    count_in = 0
    count_out=  0
    
    if 0:
        if case == 1:
            if abs(py - in_line[0][1]) <= 5 and px >= in_line[0][0] and px <= in_line[1][0]:
                if ud == 1:
                    # up & out
                    count_out =1
                elif ud == 2:
                    # down & in
                    count_in = 1
                
        elif case == 2:
            if abs(py - in_line[0][1]) <= 5 and px >= in_line[0][0] and px <= in_line[1][0]:
                if ud == 1:
                    # up & in
                    count_in =1
                elif ud == 2:
                    # down & out
                    count_out = 1
        elif case == 3:
            if abs(px - in_line[0][0]) <= 5 and py >= in_line[0][1] and py <= in_line[1][1]:
                if rl == 1:
                    # right & in
                    count_in =1
                elif rl == 2:
                    # left & out
                    count_out = 1
        elif case == 4:
            if abs(px - in_line[0][0]) <= 5 and py >= in_line[0][1] and py <= in_line[1][1]:
                if rl == 2:
                    # left & in
                    count_in =1
                elif rl == 1:
                    # right & out
                    count_out = 1
    elif 1:
        if case == 1:
            if abs(py - out_line[0][1]) <= 5 and px >= out_line[0][0] and px <= out_line[1][0]:
                if ud == 2:
                    # down & out
                    count_out =1
                elif ud == 1:
                    # up & in
                    count_in = 1  
        elif case == 2:
            if abs(py - out_line[0][1]) <= 5 and px >= out_line[0][0] and px <= out_line[1][0]:
                if ud == 1:
                    # up & out
                    count_out =1
                elif ud == 2:
                    # down & in
                    count_in = 1
        elif case == 3:
            if abs(px - out_line[0][0]) <= 5 and py >= out_line[0][1] and py <= out_line[1][1]:
                if rl == 1:
                    # right & out
                    count_out =1
                elif rl == 2:
                    # left & in
                    count_in = 1
        elif case == 4:
            if abs(px - out_line[0][0]) <= 5 and py >= out_line[0][1] and py <= out_line[1][1]:
                if rl == 2:
                    # left & out
                    count_out =1
                elif rl == 1:
                    # right & in
                    count_in = 1
        
    return count_in, count_out
    
def draw_ROI(img, n_cst, n_vst, d_time, count_in, count_out):
    # cv2.rectangle(img, (ROI[-2][0], ROI[-2][1]), (ROI[-1][0], ROI[-1][1]), (0,0, 128), 3)
    cv2.rectangle(img, (ROI[-2][0], ROI[-2][1]), (ROI[-1][0], ROI[-1][1]), (36, 255, 12), 1)
    # coord = "(%d %d) " % (ROI[-2][0], ROI[-2][1])
    # cv2.putText(im0, coord, (ROI[-2][0], ROI[-2][1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    
    label = "COUNT: " + str(n_cst) + "\nVISITED: " + str(n_vst) + "\nIN: %d\nOUT: %d" % (count_in, count_out)
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
    
    if d_time != 0:
        timetext = "Dwell Time: " + str(int(d_time)) + "s"
        # cv2.putText(img, timetext, (ROI[-1][0] + 2, ROI[-2][1] + labelsize[0][1]*3 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255], 1)
        cv2.putText(img, timetext, (x + 2, y + labelsize[0][1]*5 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1)
    
    # label_inout = "\nIN: %d\nOUT: %d" % (count_in, count_out)
    # # cv2.putText(img, label_inout, (ROI[-1][0] + 2, ROI[-2][1] + labelsize[0][1]*3 + 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, [255, 255, 255], 1)
    # cv2.putText(img, label_inout, (x + 2, y + labelsize[0][1]*3 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, [0, 0, 0], 1)

def on_mouse(event, x, y, flags, params):
    global IsSelected      
    if event == cv2.EVENT_LBUTTONDOWN:
        sbox = [x, y]
        ROI.append(sbox)
        print('ButtonDownEvent:', sbox)
        IsSelected = False
    elif event == cv2.EVENT_LBUTTONUP:
        ebox = [x, y]
        ROI.append(ebox)
        print('ButtonUpEvent:', ebox)
        print(ROI)
        IsSelected = True
    elif event == cv2.EVENT_RBUTTONUP:
        ROI.clear()
        IsSelected = False

def RepresentsInt(s):
    try: 
        int(s)
        return s
    except ValueError:
        return False

def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, half = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate, opt.half
    webcam = source == '0' or '1' or '2' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize socket
    client = ThreadedClient()
    client.start_listen()
    play_trig = -1
    count_time = 0
    
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

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

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        show_vid = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, rotate=opt.rotate)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    ## extract what is in between the last '/' and last '.'
    # txt_file_name = source.split('/')[-1].split('.')[0]
    # txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results' + time.strftime('%y%m%d_%H%M%S', time.localtime(time.time())) +'.txt'
    log_path = str(Path(out)) + '/log_' + time.strftime('%y%m%d_%H%M%S', time.localtime(time.time())) +'.txt'
    
    # if opt.rotate:
    #     mask = np.zeros((dataset.imgs[0].shape[1], dataset.imgs[0].shape[0], 3), dtype=np.uint8)
    # else:
    mask = np.zeros((dataset.imgs[0].shape[0], dataset.imgs[0].shape[1], 3), dtype=np.uint8)

    n_customer = 0
    n_visited = 0
    dwell_time = 0
    visitedCustomers = {}
    CustomerList = {}
    remainedCustomers = {}
    timeRecord = []
    count_in = 0
    count_out = 0
    

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()

        n_customer = 0

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]
                    _, Customers = draw_boxes(im0, bbox_xyxy, identities, (0, 0), opt.foot)
                    for customer in Customers:
                        id = customer.getID()
                        [px, py] = customer.getCentralPoint()
                        c_iin = 0
                        c_iout = 0
                        c_oin = 0
                        c_oout = 0
                        
                        ## direction check
                        if CustomerList.get(id, False) == False:
                            CustomerList[id] = customer
                        else:
                            [prev_x, prev_y] =  CustomerList[id].central
                            mask = cv2.line(mask, (prev_x, prev_y), (px, py), compute_color_for_labels(id), 2)
                            # mask = cv2.arrowedLine(mask, (prev_x, prev_y), (px, py), compute_color_for_labels(id), 2, 8, 0, 0.3)
                            CustomerList[id].move(px, py)
                        if check_in(px, py):
                            n_customer = n_customer + 1
                            # print("n_customer: " , n_customer, ", len: ", len(outputs))
                            if visitedCustomers.get(id, False) != True:
                                visitedCustomers[id] = True
                                n_visited = n_visited + 1
                                CustomerList[id].visit(time.time())
                                remainedCustomers[id] = True
                                c_iin, c_iout = check_inout_line(CustomerList[id], 0)
                                c_oin, c_oout = check_inout_line(CustomerList[id], 1)
                            # print("remainedCustomers len: ", len(remainedCustomers))
                            
                        else:
                            if remainedCustomers.get(id, False) != False:
                                in_time = CustomerList[id].getVisitTime()
                                out_time = time.time()
                                CustomerList[id].leave(out_time)
                                c_iin, c_iout = check_inout_line(CustomerList[id], 0)
                                c_oin, c_oout = check_inout_line(CustomerList[id], 1)
                                del remainedCustomers[id]
                                timeRecord.append(out_time - in_time)

                        count_in += c_iin
                        count_out += c_oout

                    # Write MOT compliant results to file
                    if save_txt and len(outputs) != 0:
                        for j, output in enumerate(outputs):
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2]
                            bbox_h = output[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                        with open(log_path, 'a') as f:
                            line = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " count %d total %d dwell_time %f in %d out %d\n" % (n_customer, n_visited, dwell_time, count_in, count_out)
                            f.write(line)
            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            if len(timeRecord) != 0:
                dwell_time = sum(timeRecord)/len(timeRecord)
            
            if n_customer >= 1:
                if count_time < 10:
                    count_time += 1
                    continue
                else:
                    if n_customer >= COUNT_THRESHOLD:
                        client.add_message('play/')
                    else:
                        vod_num = str(1) # 이후에 age/gender에 따른 영상 번호로 변경
                        msg = 'play/' + vod_num
                        client.add_message(msg)
                    play_trig = 1
            elif play_trig == 1:
                client.add_message('stop/')
                play_trig = -1
                count_time = 0
            else:
                play_trig = -1
                count_time = 0
            ### Per frame
            # client.add_message(str(n_customer))

            # Stream results
            im0 = annotator.result()
            if show_vid:
                if IsSelected:
                    # visualize result of count
                    draw_ROI(im0, n_customer, n_visited, dwell_time, count_in, count_out)
                    draw_line(im0)
                im0 = cv2.addWeighted(im0, 1.0, mask, 0.5, 0)
                cv2.imshow(p, im0)

                ## resized window
                # dst = cv2.resize(im0, dsize=(0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                # cv2.imshow("resized", dst)

                cv2.setMouseCallback(p, on_mouse, 0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    # raise StopIteration
                    quit()

                if cv2.waitKey(1) == ord('r'):  # r to reset
                    n_customer = 0
                    n_visited = 0
                    dwell_time = 0
                    count_in = 0
                    count_out = 0
                    visitedCustomers.clear()
                    remainedCustomers.clear()
                    CustomerList.clear()
                    timeRecord.clear()
                    # if opt.rotate:
                    #     mask = np.zeros((dataset.imgs[0].shape[1], dataset.imgs[0].shape[0], 3), dtype=np.uint8)
                    # else:
                    mask = np.zeros((dataset.imgs[0].shape[0], dataset.imgs[0].shape[1], 3), dtype=np.uint8)
                

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=str, default='crowdhuman_yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', default=0, nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    # rotate 90
    parser.add_argument("--rotate", default=False, type=bool, help="rotate video 90")
    # foot tracker
    parser.add_argument("--foot", default=True, type=bool, help='track foot position')
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
