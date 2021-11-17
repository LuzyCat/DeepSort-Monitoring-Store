import socket
import threading
import queue
import time

IP = "192.168.1.101"
PORT = 8987

class ThreadedClient(threading.Thread):
    def __init__(self, host=IP, port=PORT):
        self.msg = ""
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.connect((self.host, self.port))
        self.socket.settimeout(.1)
    def listen(self):
        while True:
            try:
                self.send_message()
                recv_msg = self.socket.recv(4096)
            except socket.timeout:
                pass
    def start_listen(self):
        t = threading.Thread(target=self.listen)
        t.start()
        print("START Client")
    def add_message(self, msg):
        self.msg = msg
    def send_message(self):
        if self.msg != "":
            # self.socket.sendto(self.msg.encode('utf-8'), (self.host, self.port))
            self.socket.send(self.msg.encode('utf-8'))
            self.msg = ""