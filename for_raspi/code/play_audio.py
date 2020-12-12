import time

import pygame


# 播放警示音频
class Play_Audio():
    def __init__(self,audio):
        self.audio = audio
    def run(self,audio):
        pygame.mixer.init()
        pygame.mixer.music.load(self.audio) # './audio/warning.mp3'
        pygame.mixer.music.play()
        time.sleep(3)
        pygame.mixer.music.stop()
