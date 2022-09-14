import numpy as np
import cv2
from iutils import compute_color_for_labels
import math
import time
import threading
from scipy import stats, integrate, spatial, ndimage
from scipy.ndimage.filters import gaussian_filter 
import matplotlib.pyplot as plt
from matplotlib import cm as CM

class ImageInfo:
    def __init__(self, shape):
        self.shape = shape
        self.mask_traj = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
        # self.mask_heat = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)

        # crop box
        self.ROI = []
        self.in_line = []
        self.out_line = []
        self.case = 0
        self.IsSelected = False
        self.nondet_ROI = []
        
        self.n_customer = 0 ## 현재 방문자 수
        self.n_visited = 0 ## 현재까지의 방문자 수
        self.dwell_time = 0 ## 체류 시간
        self.CustomerList = {} ## 현재 방문자 정보
        self.visitedCustomers = {} ## 방문했던 id 목록
        self.remainedCustomers = {} ## 현재 방문자 id 목록
        self.timeRecord = [] ## 시간 기록
        # self.movementRecord = []
        
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
            # self.movementRecord.append(customer.getCentralPoint())
            
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
                    # self.mask_traj = cv2.line(self.mask_traj, (prev_x, prev_y), (px, py), color, 1) # draw trajectroy
                    self.mask_traj = cv2.arrowedLine(self.mask_traj, (prev_x, prev_y), (px, py), color, 1, 8, 0, 0.3)
            
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
        
    # def density(self):
    #     density = np.zeros(self.shape, dtype=np.float32)
    #     points = np.array(self.movementRecord)
    #     counts = len(self.movementRecord)
    #     if counts > 0:
    #         tree = spatial.KDTree(points)
    #         distances, locations = tree.query(points, k=4)
    #         for i, pt in enumerate(points):
    #             pt2d = np.zeros(self.shape, dtype=np.float32)
    #             pt2d[pt[1],pt[0]] = 1.
    #             if counts > 1:
    #                 sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
    #             else:
    #                 sigma = np.average(np.array(self.shape))/2./2. #case: 1 point
    #             density += gaussian_filter(pt2d, sigma, mode='constant')
    #     K = gaussian_filter(density, 15)
    #     re = plt.imshow(K,cmap=CM.jet)
    #     return density
        
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
        
    def draw_ROI(self, img, cap=0):
        cv2.rectangle(img, (self.ROI[-2][0], self.ROI[-2][1]), (self.ROI[-1][0], self.ROI[-1][1]), (36, 255, 12), 1)
        
        if len(self.nondet_ROI):
            cv2.rectangle(img, (self.nondet_ROI[-2][0], self.nondet_ROI[-2][1]), (self.nondet_ROI[-1][0], self.nondet_ROI[-1][1]), (255, 255, 255), 1)
        
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
        self.nondet_ROI.clear()
        self.in_line.clear()
        self.out_line.clear()
        self.case = 0
        self.IsSelected = False
        
        self.mask_traj = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
     