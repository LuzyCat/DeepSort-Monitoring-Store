import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def RepresentsInt(s):
    try: 
        int(s)
        return s
    except ValueError:
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

def getCentralPoint(box, offset=(0, 0), foot=True):
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
    
    return px, py

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

def xywh_to_xyxy(bbox_xywh, width, height):
    x, y, w, h = bbox_xywh
    x1 = max(int(x - w / 2), 0)
    x2 = min(int(x + w / 2), width - 1)
    y1 = max(int(y - h / 2), 0)
    y2 = min(int(y + h / 2), height - 1)
    return x1, y1, x2, y2

def xywh_to_foot(bbox_xywh, width, height):
    x, y, w, h = bbox_xywh
    xc = min(int(x + (w / 2)), width - 1)
    yc = min(int(y + h), height - 1)
    return xc, yc
