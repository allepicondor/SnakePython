import pygame
import numpy as np
WIDTH = 900
HEIGHT = 900

#pos is relating to board so [1,1]
#pix is relating to window in pixels so [30,30]
class Board:
    def __init__(self,width,height,segment):
        self.pixwidth = width#300
        self.pixheight = height #300
        self.segment = segment#30
        self.rows = int(((self.pixheight+(segment*2))/self.segment))
        self.coloums = int(((self.pixwidth+(segment*2))/self.segment))
        self.board = np.zeros((self.rows,self.coloums,2),dtype=np.int32)
        for x in range(self.rows):
            for y in range(self.coloums):
                self.board[x][y] = [x*self.segment,y*self.segment]
    def grabPix(self,pos):#takes in a pos ex. [3,3] returns pixel value of the pos so [90,90] by calling self.board 
        pass
    def grabPos(self,pix):#takes in a pix ex. [90,90] returns pos value of the pix so [3,3] by dividing by segment size
        pass
    def draw(self,win):
        for x in range(self.rows):
            for y in range(self.coloums):
                pos = self.board[x][y]
                pygame.draw.rect(win,(55,0,0),(pos[0]+2,pos[1]+2,self.segment - 2,self.segment - 2))

board = Board(WIDTH,HEIGHT,15)
win = pygame.display.set_mode((WIDTH,HEIGHT))
clock = pygame.time.Clock()

while True:
    clock.tick(60)
    win.fill((0,0,0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        y -= 10
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        y += 10
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        x -= 10
    if keys[pygame.K_d] or keys [pygame.K_RIGHT]:
        x += 10
    board.draw(win)
    pygame.display.update()


