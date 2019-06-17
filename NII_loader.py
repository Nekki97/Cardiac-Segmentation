import matplotlib.pyplot as plt
import nibabel as nib
import os
import numpy as np
from scipy import ndimage as nd
import imageio

def load_paths(img_root, mask_root):
    """
    :param data_dir, both paths to systole and diastole images with corresponding labels
    :return: array of paths to unlabeled images and array with path to corresponding labels
    """
    img_paths = []
    mask_paths = []
    for root, subdirList, fileList in os.walk(img_root):
        for filename in fileList:
           img_paths.append(os.path.join(root, filename))
    for root, subdirList, fileList in os.walk(mask_root):
        for filename in fileList:
           mask_paths.append(os.path.join(root, filename))

    return img_paths, mask_paths


def load_data(data_paths):
    data = []
    for i in data_paths:
        img = nib.load(i)
        img_data = img.get_data()
        data.append(img_data)

    return data


def display(img_data, block = True):
    """
    Plots middle slice of 3D image
    :param img_data: data to be plotted
    :param block: if programm should be blocked
    """
    def show_slices(slices):
        """ Function to display row of image slices """
        fig, axes = plt.subplots(1)
        for i, slice in enumerate(slices):
            axes.imshow(slice.T, cmap="gray", origin="lower")

    n_i, n_j, n_k = img_data.shape
    center_i = (n_i - 1) // 2  # // for integer division
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2
    # print('Image center: ', center_i, center_j, center_k)
    center_vox_value = img_data[center_i, center_j, center_k]
    # print('Image center value: ', center_vox_value)

    slice_0 = img_data[center_i, :, :]
    slice_1 = img_data[:, center_j, :]
    slice_2 = img_data[:, :, center_k]

    show_slices([slice_0, slice_1, slice_2])
    show_slices([slice_0])


    plt.suptitle("Center slices for image")
    plt.show(block = block)


def find_center_of_mass(data):
    """
    :param data: array of images with shape (x, y, z)
    :return: list of arrays (of length z)  with the center of mass for each of these images (x, y) and list of empty labels (which patient, which slice)
    """
    coms = []
    empty_labels = []
    for s in range(0, len(data)):
        com = []
        data[s] = np.moveaxis(data[s], 0, -1)
        for i in range(0, data[s].shape[2]):
            bool_temp = np.sum(data[s][..., i])
            if bool_temp != 0:         #verify that label is non empty
                com.append(nd.measurements.center_of_mass(data[s][..., [i]][..., 0]))

            else:
                empty_labels.append((s, i))

        coms.append(com)
        data[s] = np.moveaxis(data[s], -1, 0)

    return coms, empty_labels


def crop_images(cropper_size, center_of_masses, data):
    """
    :param cropper_size:
    :param center_of_masses: list of lists of length z with the com (x,y)
    :param data: array of images with shape (x, y, z)
    :return: returns np.array of cropped images (z, x, y)
    """
    cropped_data = []
    for s in range(len(data)):
        data[s] = np.moveaxis(data[s], 0, -1)
        temp = np.empty((data[s].shape[2], 2*cropper_size, 2*cropper_size))
        for i in range(data[s].shape[2]):
            padded = np.pad(data[s][..., i], cropper_size, 'constant')
            center_i = int(padded.shape[0] / 2)
            center_j = int(padded.shape[1] / 2)
            x_start = center_i - cropper_size
            x_end = center_i + cropper_size
            y_start = center_j - cropper_size
            y_end = center_j + cropper_size
            temp[i] = padded[x_start:x_end, y_start:y_end]
        cropped_data.append(temp)
    return cropped_data


def save_datavisualisation2(img_data, myocar_labels, save_folder, counter, index_first=False, normalized=False):
    if index_first:
        for i in range(len(img_data)):
            img_data[i] = np.moveaxis(img_data[i], 0, -1)
            myocar_labels[i] = np.moveaxis(myocar_labels[i], 0, -1)
    print(img_data.shape)
    img_data = np.expand_dims(img_data,0)
    myocar_labels = np.expand_dims(myocar_labels, 0)
    print(img_data.shape)
    for i, j in zip(img_data[:,:,:], myocar_labels[:,:,:]):
        print(counter)
        print(i.shape)
        i_patch = i[0, :, :]
        if normalized:
            i_patch = i_patch*255
        # np.squeeze(i_patch)

        j_patch = j[0, :, :]
        # np.squeeze(j_patch)
        j_patch = j_patch * 255
        for slice in range(1, i.shape[0]):
            temp_i = i[slice, :, :]
            temp_j = j[slice, :, :]
            if normalized:
                temp_i = temp_i * 255
            temp_j = temp_j * 255

            i_patch = np.hstack((i_patch, temp_i))
            j_patch = np.hstack((j_patch, temp_j))

        image = np.vstack((i_patch, j_patch))
        imageio.imwrite(save_folder + '%d.png' % (counter,), image)


def remove_empty_label_data(data, empty_labels):
    """
    :param data:
    :param empty_labels: (which_person, which_slice)
    :return: data without the slices that have empty segmentations
    """
    new_empty = []
    for i in np.asarray(empty_labels):
        new_empty.append([i[0],i[1]])

    new_i = []
    new_data = []
    for i in range(len(data)):
        new_i = []
        for j in range(data[i].shape[0]):
            if [i,j] not in new_empty:
                new_i.append(data[i][j,:,:])
        new_data.append(np.array(new_i,dtype=float))

    return new_data


def remove_other_segmentations(labels):
    """
    loads labels and removes all other segmentations apart from the myocardium one
    :param label: list of strings with paths to labels
    :return: array with all the myocar labels
        https://nipy.org/nibabel/images_and_memory.html
    """
    myocar_data = []
    for i in labels:
        img = nib.load(i)
        img_data = img.get_data()
        img_data[img_data != 1] = 0
        myocar_data.append(img_data)
    return myocar_data


def data_normalization(data):
    for i in data:
        for j in range(i.shape[0]):
            i[j] = i[j]*1.
            i[j] = np.clip(i[j], 0, np.percentile(i[j], 99))
            i[j] = i[j] - np.amin(i[j])
            if np.amax(i[j]) != 0:
                i[j] = i[j] / np.amax(i[j])
    return data



def get_nii_data():

    img_root = "D:/Pycharm_dir/Cardiac-Segmentation/nii_data/images"
    mask_root = "D:/Pycharm_dir/Cardiac-Segmentation/nii_data/masks"

    img_paths, mask_paths = load_paths(img_root, mask_root)
    img_paths.sort()
    mask_paths.sort()

    print("Datapaths ready")

    masks = remove_other_segmentations(mask_paths)
    images = load_data(img_paths)

    print("after loading")
    coms, emptymasks = find_center_of_mass(masks)

    print("before label removing")
    masks = remove_empty_label_data(masks, emptymasks)
    images = remove_empty_label_data(images, emptymasks)

    print("before cropping")
    images = crop_images(100, coms, images)
    masks = crop_images(100, coms, masks)
    print("after cropping")

    print("before normalizing")
    images = data_normalization(images)
    print("after normalizing")


    return images, masks



#masks = data_normalization(masks)
#counter = 0
#for patient in range(len(images)):
#    save_datavisualisation2(images[patient], masks[patient], "nii_data/visualize/", counter, normalized=True)
#    counter += 1
#print("Visualized data")
