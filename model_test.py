#imports
import torch
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import time
import pygame
import threading
import telegram_send
from datetime import datetime
#variables
is_elephant_bool=False
is_bird_bool=False

#disctionary mapping for the classes
import re

def dict_mapping(mapping_file_path):
    a_dictionary = {}
    a_file = open(mapping_file_path)
    for line in a_file:
        key, value = line.split(":")
        key=re.findall(r'\d+',key)[0]
        key=int(key)
        a_dictionary[key] = value
    return a_dictionary

a_dictionary=dict_mapping("mapping.txt")


#opencv videocapture
vc=cv2.VideoCapture(0)
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])


def is_elephant_fn(class_id):
    if class_id in [385,386]:
        return True
    else:
        return False
    
def is_bird_fn(class_id):
    if class_id in [134,15,11,8,7]:
        return True
    else:
        return False
    

#image prediction function
def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to('cpu')
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

def play_sound(sound_file):
    pygame.mixer.init()
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    while(pygame.mixer.music.get_busy()==True):
          continue

def is_play_sound(is_elephant_bool,is_bird_bool,frame):
    
    if is_elephant_bool==True:
        print('playng elephant sound')
        sound_file="bee.mp3"
        play_sound(sound_file)
        cv2.imwrite("captured_images/image.jpg",frame)
        send_telegram_message('Elephant')
    elif is_bird_bool==True:
        print('playng hawk sound...')
        sound_file="hawk.mp3"
        play_sound(sound_file)
        cv2.imwrite("captured_images/image.jpg",frame)
        send_telegram_message('Bird')
    else:
        pass


def get_date_time():
    now=datetime.now()
    dt_string=now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string

def send_telegram_message(animal_type):
    telegram_send.send(messages=[animal_type+" Found!!"])
    with open ("captured_images/image.jpg","rb") as f:
        telegram_send.send(images=[f])
    telegram_send.send(messages=[get_date_time()])
    
    
#pytorch prediction
model = models.googlenet(pretrained=True)
model.eval()
flag_time=time.time()
while(1):

    ret,frame=vc.read()
    frame=cv2.rotate(frame,cv2.ROTATE_180)
    frame = cv2.resize(frame, (800,800), interpolation = cv2.INTER_AREA)
    cv2.imshow("im",frame)
    if cv2.waitKey(1)==ord('q'):
        break
    frame = cv2.resize(frame, (225,225), interpolation = cv2.INTER_AREA)

    
    if time.time()-flag_time>3:
        flag_time=time.time()
        frame_time=time.time()
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image=Image.fromarray(color_coverted)
        result=predict_image(pil_image)
        print(a_dictionary[result])
        
        is_elephant_bool=is_elephant_fn(result)
        is_bird_bool=is_bird_fn(result)
        #is_play_sound(is_elephant_bool,is_bird_bool)
        x0=threading.Thread(target=is_play_sound,args=(is_elephant_bool,is_bird_bool,frame))
        x0.start()
        prediction_time=time.time()
        time_difference=prediction_time-frame_time
        print("Prediction time", time_difference)
        

vc.release()
