import cv2
import torchvision
import os
import torch
from sklearn.cluster import KMeans

def get_file_paths(dir):
    'returns list of .jpg files in a directory'
    filenames = os.listdir(dir)
    file_paths = [os.path.join(dir, filename) for filename in filenames\
        if filename.endswith('.jpg')]
    
    return file_paths


def rect_area(x1, y1, x2, y2):
    'returns an area of a rectnagle by its coordinates'
    return abs(x1-x2) * abs(y1-y2)

def prepare_for_model(img, device):
    '''
    converts an array to torch tensor
    moves it to the given device
    manages its dimensions to suit the (B, C, H, W) format
    and converts 0-255 uint8 to 0-1 float32
    '''
    
    img = cv2.resize(img, (img.shape[0]//4, img.shape[1]//4))
    img = torch.tensor(img).to(device).movedim(-1, 0).unsqueeze(dim=0) / 255
    
    return img

def get_features(img, device, detection_model, feature_extractor):
    '''passes given image through
    object detection and classification models
    to extract some features for clustering and
    saliency estimation
    '''
    img = prepare_for_model(img, device)
    with torch.no_grad():
        output = detection_model(img)[0]
        feature_vector = feature_extractor(img).flatten()
    boxes = output['boxes']
    scores = output['scores']
        
    return boxes, scores, feature_vector


def get_saliency(boxes, scores):
    'returns a custom saliency criterion'
    
    saliency = 0
    for box, score in zip(boxes, scores):
        saliency += score * rect_area(*box)
    
    return saliency.item()

def clusterize(vectors, n=2):
    'clusterizes a set of verctors and returns its labels'
    kmeans = KMeans(n_clusters=n, random_state=0).fit(vectors)
    return kmeans.labels_


def main():

    device = 'cuda'

    detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    detection_model.eval().to(device)

    resnet = torchvision.models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1])).eval().to(device)
    
    dir = 'frames'
    img_properties = {}  # dict structure: {file_path: {'saliency': value, 'feature_vector': vector}}
    
    # save info on saliency score and feature vector (clustering input)
    # for each image in a directory into a img_properties dictionary
    for file_path in get_file_paths(dir):
        img = cv2.imread(file_path)
        scores, boxes, feature_vector = get_features(img, device, detection_model, feature_extractor)
        img_properties[file_path] = {'saliency': get_saliency(scores, boxes),
                                    'feature_vector': feature_vector.cpu().numpy()}
    
    # form a list of feature vectors to feed it into a clusterizator
    vectors = []
    for v in img_properties.values():
        vectors.append(v['feature_vector'])

    cluster_labels = clusterize(vectors)
    
    # for each cluster find an image (filename) with the highest saliency
    max_saliencies = {}  # dict format: {cluster_label: {file_path: path, saliency: val}, ...}
    for cluster in set(cluster_labels):
        max_saliencies[cluster] = {'saliency': 0}
        for label, file_path in zip(cluster_labels, img_properties.keys()):
            if label == cluster:
                if img_properties[file_path]['saliency'] > max_saliencies[cluster]['saliency']:
                    max_saliencies[cluster]['file_path'] = file_path
                    max_saliencies[cluster]['saliency'] = img_properties[file_path]['saliency']
    
    print(max_saliencies)
    pass

if __name__ == '__main__':
    main()