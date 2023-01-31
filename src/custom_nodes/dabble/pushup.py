from typing import Any, Dict, List, Tuple
import cv2
import math
from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from tkinter import *
from PIL import Image, ImageTk
#global constants
ORG_TIMER = "0"
TIME_GIVEN = 5
FONT_NAME = "Courier"
window = Tk()
window.title("PushUp CV")
window.geometry("500x500")

def show_frames():
   # Get the latest frame and convert into Image
   cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
   img = Image.fromarray(cv2image)
   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk) 
   # Repeat after an interval to capture continiously
   label.after(20, show_frames)
   
def close_window_to_start():
   window.destroy()

def start_timer():
   global TIME_GIVEN
   time.config(text="GET INTO POSITION")
   count_down(TIME_GIVEN)

def count_down(count):
    count_min = math.floor(count/60)
    count_sec = count % 60
    if len(str(count_sec)) < 2:
        count_sec = "0" + str(count_sec)
    canvas.itemconfig(timer_text, text=f"{count_min}:{count_sec}")
    if count > 0:
        global timer
        timer = window.after(1000, count_down, count - 1)
    elif count == 0:
        close_window_to_start()

time = Label(text="Timer", font=(FONT_NAME, 50, "bold"))
time.pack()

start_button = Button(text="Start", command=start_timer)
start_button.pack()

label =Label(window)
label.pack()
cap= cv2.VideoCapture(0)

canvas = Canvas(width=100, height= 150, highlightthickness=0)
timer_text = canvas.create_text(50,75, text="00:00", fill="black", font=(FONT_NAME, 35, "bold"))
canvas.pack()

show_frames()
window.mainloop()

# setup global constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
RED = (0,0,255)  
GREEN = (0,255,0)     
BLACK = (0, 0,0)
THRESHOLD = 0.3               # ignore keypoints below this threshold
KP_RIGHT_EAR = 4              # PoseNet's skeletal keypoints
KP_RIGHT_SHOULDER = 6         
KP_RIGHT_WRIST = 10
KP_RIGHT_ELBOW = 8
KP_RIGHT_HIP = 12
KP_RIGHT_KNEE = 14
KP_RIGHT_ANKLE = 16
keypointList = [4,6,10,8,12,14,16]

def map_keypoint_to_image_coords(         #Second helper function to convert relative keypoint coordinates to absolute image coordinates.
   keypoint: List[float], image_size: Tuple[int, int]
) -> List[int]:
   width, height = image_size[0], image_size[1]
   x, y = keypoint
   x *= width
   y *= height
   return int(x), int(y)

def draw_text(img, x, y, text_str: str, color_code):
   cv2.putText(
      img=img,
      text=text_str,
      org=(x, y),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=0.7,
      color=color_code,
      thickness=2,
   )
class Node(AbstractNode):
   def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
      super().__init__(config, node_path=__name__, **kwargs)
      self.upCondition = set()   #check if all conditions have been fufilled to consider a successful up
      self.downCondition = set() #check if all conditions have been fufilled to consider a successful down
      self.direction = "up"      #initialize as up, so later on we are looking our for a successful "down" 
      self.num_pushups = 0
      self.shoulderElbowDistance = None

   def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
      # get required inputs from pipeline
      img = inputs["img"]
      keypoints = inputs["keypoints"]
      keypoint_scores = inputs["keypoint_scores"]
      img_size = (img.shape[1], img.shape[0])  # image width, height
      
      #the section below is on keypoint detection
      the_keypoints = keypoints[0]                 # image only has one person
      the_keypoint_scores = keypoint_scores[0]     # only one set of scores
      right_ear = None    
      right_shoulder = None
      right_wrist = None
      right_elbow = None
      right_hip = None
      right_knee = None
      right_ankle = None


      for i, keypoints in enumerate(the_keypoints):
         keypoint_score = the_keypoint_scores[i]

         if keypoint_score >= THRESHOLD:
            if i in keypointList:
               # drawing the coordinates of each keypoint
               x, y = map_keypoint_to_image_coords(keypoints.tolist(), img_size)
               x_y_str = f"({x}, {y})"
               draw_text(img, x, y, x_y_str, WHITE)

            # matching keypoints to body parts
            if i == KP_RIGHT_EAR:
               right_ear = keypoints
            if i == KP_RIGHT_SHOULDER:
               right_shoulder = keypoints
            if i == KP_RIGHT_WRIST:
               right_wrist = keypoints
            if i == KP_RIGHT_ELBOW:
               right_elbow = keypoints
            if i == KP_RIGHT_HIP:
               right_hip = keypoints
            if i == KP_RIGHT_KNEE:
               right_knee = keypoints
            if i == KP_RIGHT_ANKLE:
               right_ankle = keypoints

      def getAngle(A, B, C):
         AB = math.sqrt((B[0]-A[0])**2 + (B[1]-A[1])**2)
         BC = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)
         AC = math.sqrt((C[0]-A[0])**2 + (C[1]-A[1])**2)
         return math.degrees(math.acos((BC**2 + AB**2 - AC**2)/(2*AB*BC)))      

      def noFlare(shoulder,elbow,distance):
          # Index is 0 now because looking at x-coordinates.
          if (shoulder[0]- elbow[0]) < (0.20*distance):
              return False
          return True

      if right_shoulder is not None and right_elbow is not None:
        if self.shoulderElbowDistance == None:
            self.shoulderElbowDistance = right_elbow[1] - right_shoulder[1]
      
      if self.direction == "up": #to check for a proper down
         draw_text(img, 20, 70, "Going down", BLACK)
         
         #conditions that need to be met to be considered a successful "down" position
         if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
            angle = getAngle(right_shoulder,right_elbow,right_wrist)
            if angle <= 100:
               self.downCondition.add("a")
               draw_text(img, 20, 100, "Pushup is deep enough", GREEN)
            else:
               draw_text(img, 20, 100, "Pushup is deep enough", BLACK)
         if right_wrist is not None and right_shoulder is not None and right_ankle is not None:
            angle = getAngle(right_shoulder,right_ankle,right_wrist)
            if angle <= 20:
               self.downCondition.add("b")
               draw_text(img, 20, 130, "Body is level to the ground", GREEN)
            else:
               draw_text(img, 20, 130, "Body is level to the ground", BLACK)

         if right_shoulder is not None and right_elbow is not None and self.shoulderElbowDistance is not None:
            if noFlare(right_shoulder,right_elbow,self.shoulderElbowDistance):
                 self.downCondition.add("c")
                 draw_text(img, 20, 160, "Elbows are not flared", GREEN)
            else:
               draw_text(img, 20, 160, "Elbows are not flared", BLACK)
         #at any point,if the individuals back is curved, reset the conditions that need to be met
         if right_shoulder is not None and right_hip is not None and right_ankle is not None:
            angle = getAngle(right_shoulder,right_hip,right_ankle)
            if angle < 165:
               self.downCondition = set()
               draw_text(img, 400, 30, "BACK NOT STRAIGHT", RED)

         #to check if there is a proper down position. if there is, we acknowledge the down, count the rep, and begin to lookout for an up
         if len(self.downCondition) == 3:
            self.num_pushups +=1
            self.downCondition = set()
            self.direction = "down"

      if self.direction == "down": #to check for a proper up
         draw_text(img, 20, 70, "Going up", BLACK)

         #conditions that need to be met in order to be considered a proper "up" position
         if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
            angle = getAngle(right_shoulder,right_elbow,right_wrist)
            if angle >= 170:
               self.upCondition.add("a")
               draw_text(img, 20, 100, "Arm fully extended", GREEN)
            else:
               draw_text(img, 20, 100, "Arm fully extended", BLACK)
         if right_wrist is not None and right_shoulder is not None and right_ankle is not None:
            angle = getAngle(right_shoulder,right_ankle,right_wrist)
            if angle >= 35:
               self.upCondition.add("b")
               draw_text(img, 20, 130, "Body is level to the ground", GREEN)
            else:
               draw_text(img, 20, 130, "Body is level to the ground", BLACK)

         #at any point,if the individuals back is curved, reset the conditions that need to be met
         if right_shoulder is not None and right_hip is not None and right_ankle is not None:
            angle = getAngle(right_shoulder,right_hip,right_ankle)
            if angle < 165:
               self.upCondition = set()
               draw_text(img, 400, 30, "BACK NOT STRAIGHT", RED)
         #to check if there is a proper up position. if there is, we begin to lookout for an proper down position
         if len(self.upCondition) == 2:
            self.upCondition = set()
            self.direction = "up"      
               
      #print the number of pushup
      pushup_str = f"pushups = {self.num_pushups}"
      draw_text(img, 20, 30, pushup_str, BLACK)

      return {}

