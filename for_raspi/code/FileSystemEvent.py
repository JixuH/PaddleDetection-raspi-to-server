import time

from watchdog.events import *
from watchdog.observers import Observer

from client import Client
from play_audio import Play_Audio

# 
class MyDirEventHandler(FileSystemEventHandler):

    def __init__(self):
        FileSystemEventHandler.__init__(self)
        self.audio = './audio/warning.mp3'
        self.warning = Play_Audio(self.audio)
    # 是否创建文件、如是则报警并传送文件至终端
    def on_created(self, event):

        print("file created:{0}".format(event.src_path))
        a = str(event.src_path)
        #
        self.warning.run(self.audio)
        client = Client(a)
        client.sock_client_image(file_path = a)
        #deal_image(sock, address)
