class Customer:
    """
    Information of customer
    - customerID: detection ID
    - central: customer position in now
    - visitTime: system time when the customer entered in ROI    
    """
    def __init__(self, customerID, px, py):
        self.customerID = customerID
        self.central = [px, py]
        self.visitTime = None
        self.leaveTime = None
        self.__in__ = None
        self.direction_ud = 0 # 1:up 2:down
        self.direction_rl = 0 # 0:tight 2:left
    
    def move(self, new_px, new_py):
        [px, py] = self.central
        self.central = [new_px, new_py]
        if px < new_px:
            self.direction_rl = 1
        elif px > new_px:
            self.direction_rl = 2
        else:
            self.direction_rl = 0
        if py > new_py:
            self.direction_ud = 1
        elif py < new_px:
            self.direction_ud = 2
        else:
            self.direction_ud = 0
            
    def visit(self, time):
        self.visitTime = time
        self.__in__ = True
    def leave(self, time):
        self.leaveTime = time
        self.__in__ = False
    def getVisitTime(self):
        return self.visitTime
    def getID(self):
        return self.customerID
    def getCentralPoint(self):
        return self.central
    def getDirection_ud(self):
        return self.direction_ud
    def getDirection_rl(self):
        return self.direction_rl