import pygame
import assistant as assistant
import numpy as np
import random
WIDTH = 900
HEIGHT = 900

#pos is relating to board so [1,1]penis
#pix is relating to window in pixels so [30,30]
class Board:
    def __init__(self,width,height,segment):
        self.pixwidth = width#300#line 11 to 16 is taking how big the window is and divinding it into rows and coloums for the snake 
        self.pixheight = height #300
        self.segment = segment#30
        self.rows = int(((self.pixheight+(segment*2))/self.segment))
        self.coloums = int(((self.pixwidth+(segment*2))/self.segment))
        self.board = np.zeros((self.rows,self.coloums,2),dtype=np.int32)#Creating a table for the snake to move around in aka the grid
        for x in range(self.rows):#populating each cell of the table with there corrosponding pixel location ex. cell 1,1 == 30,30 if the segment size is 30 
            for y in range(self.coloums):
                self.board[x][y] = [x*self.segment,y*self.segment]#this is where we populate the cell
    def draw_square(self,win,cords,color):#cords = [x,y]# takes in a cord such as 2,2 or 1,1 and draws a square at the location on the board
        pos = self.board[cords[0]][cords[1]]# Get the pixel location of the cord by calling the table we made in line 16
        pygame.draw.rect(win,color,(pos[0]+2,pos[1]+2,self.segment - 2,self.segment - 2))# pygame rect drawing function
    def draw(self,win):#draws all squares on board for debugging and asthetic 
        for x in range(self.rows):
            for y in range(self.coloums):
                pos = self.board[x][y]
                pygame.draw.rect(win,(55,0,0),(pos[0]+2,pos[1]+2,self.segment - 2,self.segment - 2))
    def CheckInBounds(self,snake):
        if snake.current_pos[0] < 0:
            return True
        elif snake.current_pos[1] < 0:
            return True
        elif snake.current_pos[0] >= self.coloums-2:
            return True
        elif snake.current_pos[1] >= self.rows -2:
            return True
        else:
            return False

class Snake:
    def __init__(self,board,color):
        self.board = board #we grab this variable so we have a refrence to the board object so we can use the draw square function
        self.current_pos = np.array([3,3])#this hold current head position
        self.body_pos = [[2,3],[1,3],[0,3]]#this hold current body positions
        self.vel = np.array([1,0])# Velocity of snake [x,y] x = how many squares itll move every frame on the x axis,y = how many squares itll move every frame on the y axis
        self.color = color#color of snake cuz why not
        self.nextVel = np.array([1,0])
    def move(self):
        self.vel = self.nextVel
        copyOfBodyPos = self.body_pos.copy()#copys the current body_pos so we dont change the real body_pos until were done moving all the positions
        for i,cord in enumerate(self.body_pos):#go through every piece of the body
            if i == 0:#if the body is the first piece aka the piece right behind the head
                self.body_pos[i] = self.current_pos.copy()#if it is then we set its pos to the current head pos
            else:#otherwise
                self.body_pos[i] = copyOfBodyPos[i-1].copy()#we set the body pos to the pos of the body in front of it
        self.current_pos += self.vel#apply the vel to the head so we can change directions
    def CheckCollisionWithSelf(self):
        if assistant.NumpyInList(self.current_pos,self.body_pos):
            return True
    def add_length(self):
        self.body_pos.append(self.body_pos[-1])#literaly jus duplicate the last body cord
    def draw(self,win):
        self.board.draw_square(win,self.current_pos,self.color)#draws head
        for cord in self.body_pos:
            self.board.draw_square(win,cord,self.color)#draws body

class Food:
    def __init__(self,board):
        self.currentPos = []
        self.board = board
        self.color = [0,255,0]
    def spawn(self,snake):
        while  True:#loops through
            x = random.randrange(0,self.board.coloums -2,step=1)#randomize x
            y = random.randrange(0,self.board.rows -2,step=1)#randomize y
            good = True
            for cord in snake.body_pos:
                if (cord[0] == x and cord[1] == y):#if the [x,y] is not apart of the snake then break
                    good=False
            if(good):
                break
        self.currentPos = [x,y]
    def draw(self,win):
        self.board.draw_square(win,self.currentPos,self.color)
class VanilaGame:
    def __init__(self,WIDTH,HEIGHT,SEGMENT):
        self.board = Board(WIDTH,HEIGHT,SEGMENT)# creating the board
        self.snake = Snake(self.board,(220,220,220))# creating the snake
        self.food = Food(self.board)#create food
        self.win = pygame.display.set_mode((WIDTH,HEIGHT))#all you need to make a window in pygame
        self.clock = pygame.time.Clock()#creates clock for fps setting
        self.food.spawn(self.snake)#spawn FOOD
        for i in range(20):
            self.snake.add_length()
    def reset(self):
        self.snake = Snake(self.board,(220,220,220))# creating the snake
        self.food = Food(self.board)#create food
        self.food.spawn(self.snake)#spawn FOOD
    def GameLoop(self):
        while True:
            self.clock.tick(15)#sets FPS
            self.win.fill((0,0,0))#clears window
            for event in pygame.event.get():#you haved to have this loop in every pygame game loop|
                if event.type == pygame.QUIT:#this is vitial                                      | these make it so you can end window
                    break#this is vitial                                                          | if you dont have these lines in your pygame loop your game will crash
            keys = pygame.key.get_pressed() #Grab Pressed Keys
            if(keys[pygame.K_LEFT]) and self.snake.vel[0] != 1 and self.snake.vel[1] != 0:#left arrow
                self.snake.nextVel = [-1,0]
            if(keys[pygame.K_UP]) and self.snake.vel[0] != 0 and self.snake.vel[1] != 1:#up arrow
                self.snake.nextVel = [0,-1]
            if(keys[pygame.K_RIGHT]) and self.snake.vel[0] != -1 and self.snake.vel[1] != 0:#right arrow
                self.snake.nextVel = [1,0]
            if(keys[pygame.K_DOWN]) and self.snake.vel[0] != 0 and self.snake.vel[1] != -1:#down arrow
                self.snake.nextVel = [0,1]
            self.snake.move()#move snake
            #put Game logic her like collisions etc
            if(self.snake.current_pos[0] == self.food.currentPos[0] and self.snake.current_pos[1] == self.food.currentPos[1]):#checks if the snake head is on top of the food aka eating it this should be a fucntion but who cares its like 2 lines
                self.food.spawn(self.snake)
                self.snake.add_length()
            if (self.snake.CheckCollisionWithSelf()):
                self.reset()
            if (self.board.CheckInBounds(self.snake)):
                self.reset()
            #draw stuff here
            self.board.draw(self.win)#        -|
            self.snake.draw(self.win)#        -|
            self.food.draw(self.win)#         -| all of this jus draws stuff to screen
            pygame.display.update()#-|


game = VanilaGame(WIDTH,HEIGHT,15)
game.GameLoop()