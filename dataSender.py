import socket
import threading
import sys
import time

IP = "192.168.1.54"
# IP = 'localhost'
PORT = 8987

class ThreadedClient(threading.Thread):
    def __init__(self, host=IP, port=PORT):
        self.msg = ""
        self.led_msg = ""
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.socket.settimeout(.1)
        self.led_sockt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.led_sockt.connect(('192.168.0.6', 9001))
        print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + self.host + ', TCP_SERVER_PORT: ' + str(self.port) + ' ]')
    def listen(self):
        while True:
            try:
                self.send_message()
                recv_msg = self.socket.recv(4096)
            except socket.timeout:
                pass
    def listen_led(self):
        while True:
            try:
                self.send_message_to_led()
                recv_msg = self.socket.recv(4096)
            except socket.timeout:
                pass   
    def start_listen(self):
        t = threading.Thread(target=self.listen, daemon=True)
        t.start()
        t2 = threading.Thread(target=self.listen_led, daemon=True)
        t2.start()
        # print("START Client")
    def add_message(self, msg):
        self.msg = msg
    def add_led_message(self, msg):
        self.led_msg = msg
    def send_message(self):
        if self.msg != "":
            # self.socket.sendto(self.msg.encode('utf-8'), (self.host, self.port))
            self.socket.send(self.msg.encode('utf-8'))
            self.msg = ""
    def send_message_to_led(self):
        if self.led_msg != "":
            self.led_sockt.send(self.led_msg.encode('utf-8'))
            self.led_msg = ""

# class ThreadedClient(threading.Thread):
#     def __init__(self, host=IP, port=PORT):
#         self.msg = ("", "")
#         self.host = host
#         self.port = port
#         self.connectCount = 0
#         self.connectServer()
#         # self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         # self.socket.connect((self.host, self.port))
#         # self.socket.settimeout(.1)
    
#     def connectServer(self):
#         try:
#             self.sock = socket.socket()
#             self.sock.connect((self.host, self.port))
#             print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + self.host + ', TCP_SERVER_PORT: ' + str(self.port) + ' ]')
#             self.connectCount = 0
#         except Exception as e:
#             print(e)
#             self.connectCount += 1
#             if self.connectCount == 10:
#                 print(u'Connect fail %d times. exit program'%(self.connectCount))
#                 sys.exit()
#             print(u'%d times try to connect with server'%(self.connectCount))
#             self.connectServer() 
    
#     def start_listen(self):
#         t = threading.Thread(target=self.send_message, daemon=True)
#         t.start()

#     def add_message(self, msg):
#         self.msg[1] = msg
        
#     def send_message(self):
#         while True:
#             msg = self.msg[1]
#             if msg != self.msg[0]:
#                 try:
#                     # self.socket.sendto(self.msg.encode('utf-8'), (self.host, self.port))
#                     self.sock.send(msg.encode('utf-8').ljust(64))
#                     print('Send to Client: ', msg)
#                     self.msg[0] = msg
#                 except Exception as e:
#                     print(e)
#                     self.sock.close()
#                     time.sleep(1)
#                     self.connectServer()
#                     self.send_message()
            