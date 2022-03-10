import cv2
import torchvision
import os
import torch

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

img = cv2.imread(filename=r'man_with_cup.jpg')
img_input = torch.tensor(img).movedim(-1, 0).unsqueeze(dim=0) / 255
with torch.no_grad():
    output = model(img_input)[0]
    
treshold_mask = (output['scores'] > 0.3)
boxes = torch.reshape(torch.masked_select(output['boxes'], treshold_mask.unsqueeze(1)), (-1, 4)).type(torch.int32).cpu().numpy()
for box in boxes:
    pt1 = (box[0], box[1])
    pt2 = (box[2], box[3])
    cv2.rectangle(img, pt1, pt2, color=(0, 255, 0))
cv2.imwrite('test.jpg', img)
pass
