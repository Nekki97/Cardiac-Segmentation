import re
import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.core._multiarray_umath import ndarray
from skimage import segmentation as seg
from skimage import color
from PIL import Image

start_x = 4
end_x = 100
start_y = 4
end_y = 100
#patient = 2


def read_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b'(^P7\s(?:\s*#.*[\r\n])*'
            b'(\d+)\s(?:\s*#.*[\r\n])*'
            b'(\d+)\s(?:\s*#.*[\r\n])*'
            b'(\d+)\s(?:\s*#.*[\r\n]\s)*)', buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def get_labelled_img_paths(patient, type):
    if patient<10:
        patient = '0' + str(patient)
    if type == 'train':
        path = 'data/train/Pat' + str(patient)
    if type == 'test':
        path = 'data/test/Pat' + str(patient)
    if type == 'val':
        path = 'data/validation/Pat' + str(patient)
    filepaths = []
    labels = []
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if "out" in filename.lower():  # check whether the file's DICOM
                filepaths.append(os.path.join(dirName, filename))
                labels.append('Patient: ' + str(patient) + ' Image: ' + str(filename[-6:-4]))
    return filepaths, labels


def get_labelled_mask_paths(patient, type):
    if patient < 10:
        patient = '0' + str(patient)
    if type == 'train':
        path = 'data/train/Pat' + str(patient)
    if type == 'test':
        path = 'data/test/Pat' + str(patient)
    if type == 'val':
        path = 'data/validation/Pat' + str(patient)
    outerpaths = []
    innerpaths = []
    outerlabels = []
    innerlabels = []
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if "cavity" in filename.lower():  # check whether the file's DICOM
                innerpaths.append(os.path.join(dirName, filename))
                innerlabels.append('Patient: ' + str(patient) + ' Image: ' + str(filename[-6:-4]))
            if "epilisse" in filename.lower():
                outerpaths.append(os.path.join(dirName, filename))
                outerlabels.append('Patient: ' + str(patient) + ' Image: ' + str(filename[-6:-4]))
    return innerpaths, outerpaths, innerlabels, outerlabels


def read_pgm_paths(filepaths, byteorder='>'):
    images = []
    for i in range(len(filepaths)):
        images.append(read_pgm(filepaths[i], byteorder))
    return images


def crop_imgs(images,startx,endx,starty,endy):
    cropped_imgs = []
    for i in range(len(images)):
        image = images[i]
        cropped_image = image[startx:endx, starty:endy]
        cropped_imgs.append(cropped_image)
    return cropped_imgs

def normalize(images):
    norm_images = []
    for image in images:
        norm_image = np.clip(image,None,np.percentile(image,98))
        norm_image = norm_image - np.amin(norm_image)
        norm_image = norm_image / np.amax(norm_image)
        norm_images.append(norm_image)
    return norm_images

def show_imgs(images):
    #print("      MRI Image       |     Inner Mask        |     Outer Mask      ")
    for i in range(len(images)):
        #print(labels[0][i] + ' | ' + labels[1][i] +  ' | ' + labels[2][i])
        plt.axes().set_aspect('equal', 'datalim')
        plt.imshow(images[i], plt.cm.gray)
        plt.show()


def check_labels(labels1, labels2):
    assert len(labels1) == len(labels2)
    for i in range(len(labels1)):
        assert labels1[i].endswith(str(i+1))
        assert labels2[i].endswith(str(i+1))


def get_labelled_imgs(patient, type):
    img_paths, img_labels = get_labelled_img_paths(patient, type)
    images = crop_imgs(read_pgm_paths(img_paths),start_x,end_x,start_y,end_y)
    return images, img_labels


def get_labelled_masks(patient, type):
    masks = []
    inner_mask_paths, outer_mask_paths, inner_mask_labels, outer_mask_labels = get_labelled_mask_paths(patient, type)
    check_labels(inner_mask_labels, outer_mask_labels)
    inner_masks = crop_imgs(read_pgm_paths(inner_mask_paths),start_x,end_x,start_y,end_y)
    outer_masks = crop_imgs(read_pgm_paths(outer_mask_paths),start_x,end_x,start_y,end_y)
    assert len(outer_masks) == len(inner_masks)
    for i in range(len(outer_masks)):
        masks.append(outer_masks[i] - inner_masks[i])
    return masks, inner_mask_labels, outer_mask_labels


def get_segm_imgs(images, masks):
    segm_images = []
    for i in range(len(images)):
        segm_image = seg.mark_boundaries(images[i], masks[i], color=(0, 0, 0))
        segm_images.append(segm_image)
    return segm_images

#TODO: center all images with function using center of mass (Hendrik)


def get_data(type):
    all_norm_imgs = []
    all_norm_masks = []
    for patient in range(1, 19):
        imgs, img_labels = get_labelled_imgs(patient, type)
        masks, inner_mask_labels, outer_mask_labels = get_labelled_masks(patient, type)
        check_labels(img_labels, inner_mask_labels)
        norm_imgs = normalize(imgs)
        norm_masks = normalize(masks)
        for norm_img in norm_imgs:
            all_norm_imgs.append(norm_img)
        for norm_mask in norm_masks:
            all_norm_masks.append(norm_mask)
    return (all_norm_imgs, all_norm_masks)

def visualize(images, masks, together):
    return None


#patient = 3
#images, img_labels = get_labelled_imgs(patient, 'train')
#masks, inner_mask_labels, outer_mask_labels = get_labelled_masks(patient, 'train')
#norm_images = normalize(images)
#show_imgs(norm_images)
