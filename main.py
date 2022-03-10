import cv2
import torchvision
import os
import torch

device = 'cuda'
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval().to(device)


def rect_area(x1, y1, x2, y2):
    return abs(x1-x2) * abs(y1-y2)



def get_saliency(img):
    # TODO: add scene clustering
    with torch.no_grad():
        output = model(img)[0]
    boxes = output['boxes']
    scores = output['scores']
        
    saliency = 0
    for box, score in zip(boxes, scores):
        saliency += score * rect_area(*box)
        
    return saliency


def main():

    # loop through all frames
    FOLDER_PATH = 'frames'
    scores = {}
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith('.jpg'):
            file_path = os.path.join(FOLDER_PATH, filename)
            img = cv2.imread(file_path)
            img = torch.tensor(img).to(device).movedim(-1, 0).unsqueeze(dim=0) / 255
            scores[filename] = get_saliency(img)
    scores = dict(sorted(scores.items(), key=lambda item: item[1]))
    pass

if __name__ == '__main__':
    main()