import re
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import segmentation as seg
from scipy import ndimage as nd


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


def get_labelled_img_paths(patient):
    if patient<10:
        patient = '0' + str(patient)
    path = 'data/all_data/Pat' + str(patient)
    filepaths = []
    labels = []
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if "out" in filename.lower():  # check whether the file's DICOM
                filepaths.append(os.path.join(dirName, filename))
                labels.append('Patient: ' + str(patient) + ' Image: ' + str(filename[-6:-4]))
    return filepaths, labels


def get_labelled_mask_paths(patient):
    if patient < 10:
        patient = '0' + str(patient)
    path = 'data/all_data/Pat' + str(patient)
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
        norm_image = np.clip(image,None,np.percentile(image,99))
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


def get_labelled_imgs(patient):
    img_paths, img_labels = get_labelled_img_paths(patient)
    images = crop_imgs(read_pgm_paths(img_paths),start_x,end_x,start_y,end_y)
    return images, img_labels


def get_labelled_masks(patient):
    masks = []
    inner_mask_paths, outer_mask_paths, inner_mask_labels, outer_mask_labels = get_labelled_mask_paths(patient)
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


def get_data(cropper_size):
    all_norm_imgs = []
    all_norm_masks = []

    for patient in range(1, 19):
        #Get images
        imgs, img_labels = get_labelled_imgs(patient)
        masks, inner_mask_labels, outer_mask_labels = get_labelled_masks(patient)
        check_labels(img_labels, inner_mask_labels)
        #Scale images up to 2mm per pixel
        scaled_imgs, scaled_masks = scale(imgs, masks, patient)
        scaled_masks = np.array(scaled_masks,float)
        scaled_imgs = np.array(scaled_imgs, float)
        #Pad images up to 128x128
        coms = find_center_of_mass(scaled_masks)
        cropped_imgs = crop_images(cropper_size, coms, scaled_imgs)
        cropped_masks = crop_images(cropper_size, coms, scaled_masks)
        #Normalize images to [0, 1]
        norm_imgs = normalize(cropped_imgs)
        norm_masks = normalize(cropped_masks)

        for norm_img in norm_imgs:
            all_norm_imgs.append(norm_img)
        for norm_mask in norm_masks:
            all_norm_masks.append(norm_mask)
    return all_norm_imgs, all_norm_masks


def scale(images, masks, patient):
    # scale every image up to 2mm per pixel
    if patient < 10:
        patient = '0' + str(patient)
    path = 'data/all_data/Pat' + str(patient) + '/info.txt'
    file = open(path)
    file.readline()
    file.readline()
    resolution = float(file.readline()[0:4])
    multiplier = 2 / resolution
    scaled_images = []
    scaled_masks = []
    for image, mask in zip(images, masks):
        scaled_images.append(nd.zoom(image, multiplier))
        #print(nd.zoom(image, multiplier).shape)
        #print(nd.zoom(mask, multiplier).shape)
        scaled_masks.append(nd.zoom(mask, multiplier))
    return scaled_images, scaled_masks


def find_center_of_mass(masks):
    coms = []
    for i in range(masks.shape[0]):
        coms.append(nd.measurements.center_of_mass(masks[i]))
    coms = np.array(coms)
    return coms


def crop_images(cropper_size, center_of_masses, data):
    """
    :param cropper_size:
    :param center_of_masses:
    :param data:
    :return: returns np.array of cropped images
    """
    cropped_data = []
    counter = 0

    temp = np.empty((data.shape[0], 2*cropper_size, 2*cropper_size))
    for i in range(data.shape[0]):

        center_i = int(center_of_masses[i][0])
        center_j = int(center_of_masses[i][1])

        if center_j - cropper_size > 0 and center_i - cropper_size > 0:

        # print('center_i - cropper_size', center_i - cropper_size)
        # print('center_j - cropper_size', center_j - cropper_size)

            temp[i] = data[i,:][center_i - cropper_size: center_i + cropper_size, center_j - cropper_size: center_j + cropper_size]

            # imageio.imwrite('visualisation/data/while_cropping/' + str(counter) + 'label' + '.png', temp[i,:,:])
            counter = counter + 1
        else:
            padded = np.pad(data[i,:], ((cropper_size,cropper_size),(cropper_size,cropper_size)), 'constant')

            temp[i] = padded[center_i + 64 - cropper_size: center_i + 64 + cropper_size, center_j + 64 - cropper_size: center_j + 64 + cropper_size]

    cropped_data.append(temp)
    cropped_data = np.array(cropped_data)
    cropped_data = np.moveaxis(cropped_data, 0, 3)
    #print(str(cropped_data.shape) + " cropped data shape")

    return cropped_data
