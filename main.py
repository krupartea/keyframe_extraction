from turtle import distance
import cv2
import torchvision
import os
import torch
from sklearn.cluster import KMeans
import shutil


def get_jpeg_paths(dir):
    'returns list of .jpg files in a directory'
    filenames = os.listdir(dir)
    file_paths = [os.path.join(dir, filename) for filename in filenames\
        if filename.endswith('.jpg')]
    
    return file_paths


def get_feature_vector(path, model, device, transforms):
    '''reads and prepares the image for the model,
    performs forward pass of an feature extractor
    to get the feature vector
    '''
    
    # prepare the image for the model
    img = torchvision.io.read_image(path).to(device).unsqueeze(0)  # img is 0-255
    img = img / 255  # 0-255 uint8 to 0-1 float32
    img = transforms(img)
    
    # perform forward pass
    with torch.no_grad():
        feature_vector = model(img)[0]

    return feature_vector


def cluster(vectors, n_clusters):
    '''clusters a set of verctors and returns its labels'''
    kmeans = KMeans(n_clusters, random_state=0).fit(vectors)
    return kmeans.labels_


def euclidean_distance(tensor1, tensor2):
    '''calculates eucledean distance
    between two n-dimensinal points'''
    return (tensor1 - tensor2).pow(2).sum().sqrt()


def find_closest(sequence, point):
    '''Finds an index of an element of the given sequence
    which is the closest to the given point
    Assumes that a sequence element and the point
    have equal dimensions (i.e., they are in the same space).
    NOTE: a sequence element is a point itself
    '''
    
    # form a list (--> tensor) of distances between
    # elements of a tensor and the given point
    distances = torch.tensor([euclidean_distance(element, point) for element in sequence])
    return torch.argmin(distances)


def make_cluster_subfolders(output_dir, file_paths, labels):
    '''Creates a subfolder for each cluster,
    and copies files into corresponding subfolder,
    based on their cluster labels.
    NOTE: only for the visualization, not the inference
    '''
    for label in set(labels):
        os.mkdir(os.path.join(output_dir, f'scene_{label}'))
        
    for label, file_path in zip(labels, file_paths):
        _, filename = os.path.split(file_path)
        shutil.copyfile(src=file_path, dst=os.path.join('output', f'scene_{label}', filename))
        

def main():

    # parameters
    device = 'cuda'
    dir = 'frames2'
    output_dir = 'output'
    n_clusters = 4
    img_size = (1920, 1440)  # H, W
    downscale_factor = 4
    subfolders = True  # to make or not to make cluster subfolders

    # create the resnet model (feature extractor) instance
    model = torchvision.models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1])).eval().to(device)
    
    # define transforms for image preprocessing
    new_size = (img_size[0] // downscale_factor, img_size[1] // downscale_factor)
    transforms = torch.nn.Sequential(torchvision.transforms.Resize(new_size))
    
    # form a list of feature vectors
    feature_vectors = []
    file_paths = get_jpeg_paths(dir)
    for path in file_paths:
        feature_vectors.append(get_feature_vector(path, model, device, transforms))
    feature_vectors = torch.stack(feature_vectors).squeeze()
    
    # cluster feature vectors (images),
    # get their labels, and centers of clusters
    kmeans = KMeans(n_clusters, random_state=0).fit(feature_vectors.cpu().numpy())
    labels = kmeans.labels_  # ordered wrt file_paths
    centers = kmeans.cluster_centers_
    
    # make cluster subfolders if needed
    if subfolders:
        make_cluster_subfolders(output_dir, file_paths, labels)
    
    # find a feature vector (an image),
    # which is the closest to each cluster center.
    # These closest images are keyframes
    # TODO: restructure with a find_closest function variation
    keyframes_paths = []
    for cluster, center in zip(set(labels), centers):
        center = torch.tensor(center).to(device)
        min_distance = -1  # placeholder
        for file_path, label, feature_vector in zip(file_paths, labels, feature_vectors):
            if label == cluster:
                distance = euclidean_distance(feature_vector, center)
                if distance < min_distance or min_distance == -1:
                    min_distance = distance
                    keyframe = file_path
        keyframes_paths.append(keyframe)
    
    # create a subfolder and fill it with keyframes
    os.mkdir(os.path.join(output_dir, 'keyframes'))
    for idx, keyframe_path in enumerate(keyframes_paths):
        shutil.copyfile(src=keyframe_path, dst=os.path.join('output', 'keyframes', f'{idx}.jpg'))
        
        
if __name__ == '__main__':
    main()
