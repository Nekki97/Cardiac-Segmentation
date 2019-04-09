import re
import numpy as np
from matplotlib import pyplot as plt
from skimage import segmentation as seg


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
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


def get_paths_labels(patient, type):
    image_paths = []
    labels = []
    for i in range(1, 26):
        path = '/Users/nektarioswinter/Downloads/HeartDatabase/Pat'
        if patient < 10:
            path += '0'
        elif i == 23 and patient != 12:
            break
        path += str(patient)
        if type == 'img':
            path += '/img/out0'
        if type == 'mask1':
            path += '/mse4d/lvc/seg4db_cavity00'
        if type == 'mask2':
            path += '/mse4d/lvcm/seg4db_epilisse00'
        if i < 10:
            path += '0'
        path += str(i) + '.pgm'
        image_paths.append(path)
        label = 'Patient ' + str(patient) + ' Image ' + str(i)
        labels.append(label)
    return image_paths, labels


def read_pgm_paths(filepaths, byteorder='>'):
    images = []
    for i in range(len(filepaths)):
        images.append(read_pgm(filepaths[i], byteorder))
    return images


def crop_imgs(images):
    cropped_imgs = []
    for i in range(len(images)):
        image = images[i]
        cropped_image = image[4:100, 4:100]
        cropped_imgs.append(cropped_image)
    return cropped_imgs


def show_imgs(images):
    for i in range(len(images)):
        plt.imshow(images[i], plt.cm.gray)
        plt.show()


def get_marked_imgs(images, masks):
    marked_images = []
    for i in range(len(images)):
        marked_image = seg.mark_boundaries(images[i], masks[i], color=(1, 0, 0))
        marked_images.append(marked_image)
    return marked_images


def get_imgs_labels(patient):
    img_paths, labels = get_paths_labels(patient, 'img')
    images = crop_imgs(read_pgm_paths(img_paths))
    return images, labels


def get_marked_imgs_labels(patient, mask):
    images, labels = get_imgs_labels(patient)
    mask_paths, mask_labels = get_paths_labels(patient, mask)
    masks = crop_imgs(read_pgm_paths(mask_paths))
    marked_images = get_marked_imgs(images, masks)
    return marked_images, labels


imgs, labels = get_marked_imgs_labels(3, 'mask2')
show_imgs(imgs)


'''
path = '/Users/nektarioswinter/Downloads/HeartDatabase/Pat12/mse4d/lvc/seg4db_cavity0023.pgm'
image = read_pgm(path)
plt.imshow(image, plt.cm.gray)
plt.show()
'''

# Patient 12, Image 23 got faulty segmentation with both masks
