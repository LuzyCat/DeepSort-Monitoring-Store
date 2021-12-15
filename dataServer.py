import time
import sys
import threading
import socket

IP = "192.168.1.101"
# IP = 'localhost'
PORT = 8986

class ServerSocket(threading.Thread):
    
    def __init__(self, ip=IP, port=PORT):
        self.UDP_IP = ip
        self.UDP_PORT = port
        # self.conn = False
        self.age = -1
        self.gender = -1
        self.vod = 0
        
        self.receiveThread = threading.Thread(target=self.receiveMsg, daemon=True)
        self.receiveThread.start()
    
    def socketClose(self):
        self.sock.close()
        print(u'Server socket [ UDP_IP: ' + self.UDP_IP + ', UDP_PORT: ' + str(self.UDP_PORT) + ' ] is close')

    def socketOpen(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))
        # self.sock.listen(1)
        print(u'Server socket [ UDP_IP: ' + self.UDP_IP + ', UDP_PORT: ' + str(self.UDP_PORT) + ' ] is open')
        # self.conn, self.addr = self.sock.accept()
        # print(u'Server socket [ UDP_IP: ' + self.UDP_IP + ', UDP_PORT: ' + str(self.UDP_PORT) + ' ] is connected with client')
        # self.sock.settimeout(1.0)
    
    def getRecentRecogResult(self):
        """
        가장 최근에 인식한 age/gender 결과를 출력
        시간을 넣을까 고민되네
        """
        if self.age != -1:
            self.vod = self.gender * 5 + self.age
        return self.vod
        
    def receiveMsg(self):
        
        self.socketOpen()
        
        while True:
            data, addr = self.sock.recvfrom(1024)
            if data:
                msg = data.decode("utf-8")
                
                if msg != "":
                    # (C++) msg = ID + "-" + GENDER + "-" + AGE;
                    id, gender, age = msg.split("-")
                    print("Client: ", msg)
                    if int(age) <= 1:
                        a = 1
                    elif int(age) >= 5:
                        a = 5
                    else:
                        a = int(age)
                    g = int(gender) - 1
                    
                    self.age = a
                    self.gender = g
                else:
                    self.age = -1
                    self.gender = -1
        
        # try:
        #     while True:
        #         data = self.conn.recv(1024)
        #         # data = self.recvall(self.conn, 1024)
        #         if data:
        #             # msg = data.decode("utf-8")
        #             msg = data.decode("utf-8")
                    
        #             if msg != "":
        #                 # (C++) msg = ID + "-" + GENDER + "-" + AGE;
        #                 id, gender, age = msg.split("-")
        #                 print("Client: ", msg)
        #                 if int(age) <= 1:
        #                     a = 0
        #                 elif int(age) >= 5:
        #                     a = 5
        #                 else:
        #                     a = int(age)
        #                 g = int(gender) - 1
        #                 print("Age: %d Gender: %d".format(g, a))

        # except Exception as e:
        #     print(e)
        #     self.video.stop()
        #     self.socketClose()
        #     self.socketOpen()
        #     self.receiveThread = threading.Thread(target=self.receiveMsg, daemon=True)
        #     self.receiveThread.start()