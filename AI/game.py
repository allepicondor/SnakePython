import pygame
from PIL import Image
import assistant as assistant
import numpy as np
import cv2
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
    def move(self):
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
    MOVE_PENALTY = 1
    WALL_PENALTY = 300
    COLLISION_PENALTY = 400
    FOOD_REWARD = 25
    def __init__(self,WIDTH,HEIGHT,SEGMENT):
        self.board = Board(WIDTH,HEIGHT,SEGMENT)# creating the board
        self.snake = Snake(self.board,(220,220,220))# creating the snake
        self.food = Food(self.board)#create food
        self.OBSERVATION_SPACE_VALUES = (self.board.rows,self.board.coloums,3)
        self.ACTION_SPACE_SIZE  = 4
        self.win = pygame.display.set_mode((WIDTH,HEIGHT))#all you need to make a window in pygame
        self.clock = pygame.time.Clock()#creates clock for fps setting
        self.food.spawn(self.snake)#spawn FOOD
        self.done = False
        self.episode_step = 0
    def reset(self):
        self.episode_step = 0
        self.snake = Snake(self.board,(220,220,220))# creating the snake
        self.food = Food(self.board)#create food
        self.food.spawn(self.snake)#spawn FOOD
        self.done = False
        return np.array(self.Get_State())
    def render(self):
        img = self.Get_State()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)
    def Get_State(self):
        env = np.zeros((self.board.coloums, self.board.rows, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.currentPos[0]][self.food.currentPos[1]] = (0, 255, 0)  # sets the food location tile to green color
        for i in self.snake.body_pos:
            env[i[0]][i[1]] = (255, 0, 0)
        env[self.snake.current_pos[0]][self.snake.current_pos[1]] = (0, 0, 255)  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img
    def step(self,action):
        if(action == 0):#left arrow
            self.snake.vel = [-1,0]
        if(action == 1):#up arrow
            self.snake.vel = [0,-1]
        if(action == 2):#right arrow
            self.snake.vel = [1,0]
        if(action == 3):#down arrow
            self.snake.vel = [0,1]
        self.snake.move()#move snake
        #put Game logic her like collisions etc
        reward = -self.MOVE_PENALTY
        if(self.snake.current_pos[0] == self.food.currentPos[0] and self.snake.current_pos[1] == self.food.currentPos[1]):#checks if the snake head is on top of the food aka eating it this should be a fucntion but who cares its like 2 lines
            self.food.spawn(self.snake)
            reward = self.FOOD_REWARD
            self.done = True
            self.snake.add_length()
        if (self.snake.CheckCollisionWithSelf()):
            reward = -self.COLLISION_PENALTY
            self.done = True
        if (self.board.CheckInBounds(self.snake)):
            reward = -self.WALL_PENALTY
            self.done = True
        if self.episode_step >= 25:
            self.done = True
        self.episode_step += 1
        return np.array(self.Get_State()),reward ,self.done


