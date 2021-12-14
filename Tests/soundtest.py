import pygame


for i in range(0,10):

    print(i)
    pygame.mixer.init()
    pygame.mixer.music.load("bee.mp3")
    pygame.mixer.music.play()
    while(pygame.mixer.music.get_busy()==True):
          continue
