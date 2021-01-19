from PIL import Image
import PIL
import assistant as assistant
import numpy as np
#import cv2
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
        self.body_pos = []#this hold current body positions
        self.vel = np.array([1,0])# Velocity of snake [x,y] x = how many squares itll move every frame on the x axis,y = how many squares itll move every frame on the y axis
        self.nextVel = np.array([1,0])
        self.color = color#color of snake cuz why not
        self.viewDistance = 5
    def DangerIn(self,direction,food):
        if direction =="U":
            for i in range(self.viewDistance):
                newY = self.current_pos[1] + i
                for cord in self.body_pos:
                    if cord[0] == self.current_pos[0] and newY == cord[1]:
                        return -1
                if food.currentPos[0] == self.current_pos[0] and newY == food.currentPos[1]:
                    return 1
                if newY < 0 or newY > self.board.rows-2:
                    return -1
                return 0
        if direction =="D":
            for i in range(self.viewDistance):
                newY = self.current_pos[1] - i
                for cord in self.body_pos:
                    if cord[0] == self.current_pos[0] and newY == cord[1]:
                        return -1
                if food.currentPos[0] == self.current_pos[0] and newY == food.currentPos[1]:
                    return 1
                if newY < 0 or newY > self.board.rows-2:
                    return -1
                return 0
        if direction =="R":
            for i in range(self.viewDistance):
                newX = self.current_pos[0] + i
                for cord in self.body_pos:
                    if cord[1] == self.current_pos[1] and newX == cord[0]:
                        return -1
                if food.currentPos[1] == self.current_pos[1] and newX == food.currentPos[0]:
                    return 1
                if newX < 0 or newX > self.board.coloums-2:
                    return -1
                return 0
        if direction =="L":
            for i in range(self.viewDistance):
                newX = self.current_pos[0] - i
                for cord in self.body_pos:
                    if cord[1] == self.current_pos[1] and newX == cord[0]:
                        return -1
                if food.currentPos[1] == self.current_pos[1] and newX == food.currentPos[0]:
                    return 1
                if newX < 0 or newX > self.board.coloums-2:
                    return -1
                return 0
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
class VanilaGame:
    MOVE_PENALTY = 1
    WALL_PENALTY = 300
    COLLISION_PENALTY = 300
    FOOD_REWARD = 50
    def __init__(self,WIDTH,HEIGHT,SEGMENT):
        self.board = Board(WIDTH,HEIGHT,SEGMENT)# creating the board
        self.snake = Snake(self.board,(220,220,220))# creating the snake
        self.food = Food(self.board)#create food
        self.OBSERVATION_SPACE_VALUES = (self.board.rows-2,self.board.coloums-2,3)
        self.ACTION_SPACE_SIZE  = 4
        self.food.spawn(self.snake)#spawn FOOD
        self.done = False
        self.episode_step = 0
    def reset(self):
        self.episode_step = 0
        self.snake = Snake(self.board,(220,220,220))# creating the snake
        self.food = Food(self.board)#create food
        self.food.spawn(self.snake)#spawn FOOD
        self.done = False
        return np.array(self.Get_Image())
    def render(self):
        pass
        # img = self.Get_Image()
        # img = img.resize((300, 300),PIL.Image.BOX)  # resizing so we can see our agent in all its glory.
        # cv2.imshow("image", np.array(img))  # show it!
        # cv2.waitKey(1)
    def Get_Image(self):
        env = np.zeros((self.board.coloums-2, self.board.rows-2, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.currentPos[0]-1][self.food.currentPos[1]-1] = (0, 255, 0)  # sets the food location tile to green color
        for i in self.snake.body_pos:
            env[i[0]-1][i[1]-1] = (255, 0, 0)
        env[self.snake.current_pos[0]-1][self.snake.current_pos[1]-1] = (0, 0, 255)  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img
    def Get_State(self):
        image = True
        if image:
            return self.Get_Image()
        else:
            return [self.snake.current_pos[0],self.snake.current_pos[1],
                    self.food.currentPos[0],self.food.currentPos[1],
                    self.snake.DangerIn("U",self.food),self.snake.DangerIn("D",self.food),
                    self.snake.DangerIn("R",self.food),self.snake.DangerIn("L",self.food)]
    def step(self,action):
        if(action == 0) and self.snake.vel[0] != 1 and self.snake.vel[1] != 0:#left arrow
            self.snake.nextVel = [-1,0]
        if(action == 1) and self.snake.vel[0] != 0 and self.snake.vel[1] != 1:#up arrow
            self.snake.nextVel = [0,-1]
        if(action == 2) and self.snake.vel[0] != -1 and self.snake.vel[1] != 0:#right arrow
            self.snake.nextVel = [1,0]
        if(action == 3) and self.snake.vel[0] != 0 and self.snake.vel[1] != -1:#down arrow
            self.snake.nextVel = [0,1]
        self.snake.move()#move snake
        #put Game logic her like collisions etc
        reward = -self.MOVE_PENALTY
        if(self.snake.current_pos[0] == self.food.currentPos[0] and self.snake.current_pos[1] == self.food.currentPos[1]):#checks if the snake head is on top of the food aka eating it this should be a fucntion but who cares its like 2 lines
            self.food.spawn(self.snake)
            reward = self.FOOD_REWARD
            self.done = True
            #self.snake.add_length()
        if (self.snake.CheckCollisionWithSelf()):
            reward = -self.COLLISION_PENALTY
            self.done = True
        if (self.board.CheckInBounds(self.snake)):
            reward = -self.WALL_PENALTY
            self.done = True
        if self.episode_step >= 50:
            self.done = True
        self.episode_step += 1
        return np.array(self.Get_Image()),reward ,self.done


