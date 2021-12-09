#imports
import torch
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2

#disctionary mapping for the classes
import re
a_dictionary = {}
a_file = open("mapping.txt")
for line in a_file:
    key, value = line.split(":")
    key=re.findall(r'\d+',key)[0]
    key=int(key)
    a_dictionary[key] = value


#opencv videocapture
vc=cv2.VideoCapture(0)
img=Image.open("elephant.png")

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])

#image prediction function
def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to('cpu')
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

#pytorch prediction
model = models.googlenet(pretrained=True)
model.eval()
while(1):
	ret,frame=vc.read()
	frame = cv2.resize(frame, (800,800), interpolation = cv2.INTER_AREA)
	cv2.imshow("im",frame)
	if cv2.waitKey(1)==ord('q'):
		break
	frame = cv2.resize(frame, (225,225), interpolation = cv2.INTER_AREA)
	
	
	color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	pil_image=Image.fromarray(color_coverted)
	result=predict_image(pil_image)
	print(a_dictionary[result])


vc.release()
