class Customer:
    """
    Information of customer
    - customerID: detection ID
    - central: customer position in now
    - visitTime: system time when the customer entered in ROI    
    """
    def __init__(self, customerID, localID, px, py):
        self.customerID = customerID
        self.localID = localID
        self.central = [px, py]
        self.old_central = [px, py]
        self.visitTime = None
        self.leaveTime = None
        self.__in__ = None
        self.direction_ud = 0 # 1:up 2:down
        self.direction_rl = 0 # 0:tight 2:left
    
    def move(self, new_px, new_py):
        [px, py] = self.central
        self.old_central = [px, py]
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
    
    def merge(self, local):
        self.old_central = local.central
        self.visitTime = local.visitTime
        self.leaveTime = local.leaveTime
        self.__in__ = local.__in__
            
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
    def getLocalID(self):
        return self.localID
    def getCentralPoint(self):
        return self.central
    def getPassPoint(self):
        return self.old_central
    def getDirection_ud(self):
        return self.direction_ud
    def getDirection_rl(self):
        return self.direction_rl
    def IsCustomer(self):
        return self.customerID > 0