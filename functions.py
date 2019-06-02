from matplotlib import pyplot as plt
import numpy as np
import random
#import imageio
from sklearn import model_selection

'''
def save_datavisualisation3(img_data, myocar_labels, predited_labels, save_folder, name, bunch_size, rows, percentage, normalized = False):
    amount = int(img_data.shape[0] * percentage)
    ind = random.sample(range(img_data.shape[0]), amount)
    img_data = img_data[ind]
    myocar_labels = myocar_labels[ind]
    predited_labels = predited_labels[ind]
    pics = int(img_data.shape[0] / (bunch_size * rows)) + 1
    remaining_imgs = img_data.shape[0]
    for pic in range(pics):
        indx_start = bunch_size * rows * pic
        indx_end = bunch_size * rows * (pic + 1)
        i_bunch = []
        j_bunch = []
        k_bunch = []
        while len(i_bunch) < bunch_size * rows and len(i_bunch) < img_data.shape[0]:
            while len(j_bunch) < bunch_size * rows and len(j_bunch) < myocar_labels.shape[0]:
                while len(k_bunch) < bunch_size * rows and len(k_bunch) < predited_labels.shape[0]:
                    for i, j, k in zip(img_data[indx_start:indx_end, :, :, 0], myocar_labels[indx_start:indx_end, :, :, 0], predited_labels[indx_start:indx_end, :, :, 0]):
                        if normalized == True:
                            i = i * 255
                            j = j * 255
                            k = k * 255
                        i_bunch.append(i)
                        j_bunch.append(j)
                        k_bunch.append(k)
        i_bunch = np.array(i_bunch, dtype=float)
        j_bunch = np.array(j_bunch, dtype=float)
        k_bunch = np.array(k_bunch, dtype=float)
        full_rows = []
        for row_number in range(rows):
            if remaining_imgs > 0:
                i_row = i_bunch[bunch_size * row_number, :, :]
                j_row = j_bunch[bunch_size * row_number, :, :]
                k_row = k_bunch[bunch_size * row_number, :, :]
                remaining_imgs = max(remaining_imgs - 1, 0)
                for index in range(bunch_size * row_number + 1,bunch_size * (row_number + 1)):
                    if remaining_imgs > 0:
                        image = i_bunch[index, :, :]
                        myocar_label = j_bunch[index, :, :]
                        predicted_label = k_bunch[index, :, :]
                    else:
                        x = img_data.shape[1]
                        y = img_data.shape[2]
                        image = np.zeros((x, y))
                        myocar_label = np.zeros((x, y))
                        predicted_label = np.zeros((x,y))
                    i_row = np.hstack((i_row, image))
                    j_row = np.hstack((j_row, myocar_label))
                    k_row = np.hstack((k_row, predicted_label))
                    remaining_imgs = max(remaining_imgs - 1, 0)
                full_row = np.vstack((i_row, j_row, k_row))
                if row_number == 0:
                    full_rows = full_row
                else:
                    full_rows = np.array(full_rows, dtype=float)
                    full_rows = np.append(full_rows, full_row, axis=0)
        full_image = full_rows[0]
        for full_row in full_rows[1:full_rows.shape[0]]:
            full_image = np.vstack((full_image, full_row))
        imageio.imwrite(save_folder + "/" + name + '%d_%d.png' % (indx_start, indx_end-1), full_image)
        print("Visualised a bunch")
    print("******Finished visualising " + str(amount) + " images******")


def save_datavisualisation2(img_data, myocar_labels, save_folder, name, bunch_size, rows, percentage, normalized = False):
    amount = int(img_data.shape[0] * percentage)
    ind = random.sample(range(img_data.shape[0]), amount)
    img_data = img_data[ind]
    myocar_labels = myocar_labels[ind]
    pics = int(img_data.shape[0]/(bunch_size*rows))+1
    remaining_imgs = img_data.shape[0]
    for pic in range(pics):
        indx_start = bunch_size * rows * pic
        indx_end = bunch_size * rows * (pic + 1)
        i_bunch = []
        j_bunch = []
        while len(i_bunch) < bunch_size * rows and len(i_bunch) <= img_data.shape[0]:
            while len(j_bunch) < bunch_size * rows and len(j_bunch) <= myocar_labels.shape[0]:
                for i, j in zip(img_data[indx_start:indx_end,:,:,0], myocar_labels[indx_start:indx_end,:,:,0]):
                    if normalized == True:
                        i = i * 255
                        j = j * 255
                    j_bunch.append(j)
                    i_bunch.append(i)
        i_bunch = np.array(i_bunch, dtype=float)
        j_bunch = np.array(j_bunch, dtype=float)
        full_rows = []
        for row_number in range(rows):
            if remaining_imgs > 0:
                i_row = i_bunch[bunch_size*row_number, :, :]
                j_row = j_bunch[bunch_size*row_number, :, :]
                remaining_imgs = max(remaining_imgs - 1, 0)
                for index in range(bunch_size*row_number+1, bunch_size*(row_number+1)):
                    if remaining_imgs > 0:
                        image = i_bunch[index, :, :]
                        label = j_bunch[index, :, :]
                    else:
                        x = img_data.shape[1]
                        y = img_data.shape[2]
                        image = np.zeros((x,y))
                        label = np.zeros((x,y))
                    i_row = np.hstack((i_row, image))
                    j_row = np.hstack((j_row, label))
                    remaining_imgs = max(remaining_imgs - 1, 0)
                full_row = np.vstack((i_row, j_row))
                if row_number == 0:
                    full_rows = full_row
                else:
                    full_rows = np.array(full_rows, dtype=float)
                    full_rows = np.append(full_rows, full_row, axis=0)
        full_image = full_rows[0]
        for full_row in full_rows[1:full_rows.shape[0]]:
            full_image = np.vstack((full_image, full_row))
        imageio.imwrite(save_folder + "/" + name + '%d_%d.png' % (indx_start,indx_end-1), full_image)
    print("******Finished visualising " + str(amount) + " images******")
'''

def get_split(images, masks, split, seed):
    # split in form of (0.2,0.2)
    test_amount = int(images.shape[0]*split[0])
    val_amount = int(images.shape[0]*split[1])
    train_amount = images.shape[0] - test_amount - val_amount
    print("******************************************")
    print("TRAINING DATA: " + str(train_amount) + " images")
    print("VALIDATION DATA: " + str(val_amount) + " images")
    print("TEST DATA: " + str(test_amount) + " images")
    print("******************************************")

    train_val_images, test_images, train_val_masks, test_masks = \
            model_selection.train_test_split(images, masks, test_size=test_amount, random_state=seed)

    train_images, val_images, train_masks, val_masks = \
            model_selection.train_test_split(train_val_images, train_val_masks, test_size=val_amount, random_state=seed)

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def get_splits(images, masks, splits, seed):
    # split in form of {1:(0.2, 0.2), 2:(0.3, 0.4)}
    # return: {split1: {train_images_split1: (100,96,96), train_masks_split1: ... }, split2: {}} for every split

    split_dicts = {}
    j = 0
    for split in splits.values():
        split_data = {}
        labels = ["train_images", "train_masks", "val_images", "val_masks",
                "test_images", "test_masks"]
        train_images, train_masks, val_images, val_masks, test_images, test_masks = get_split(images, masks, split, seed)
        data = [train_images, train_masks, val_images, val_masks, test_images, test_masks]
        for i in range(len(data)):
            split_data[labels[i]] = data[i]
        split_dicts["Split#"+str(j)] = split_data
        j += 1
    return split_dicts


def get_datasets(data, split_number):
    index = "Split#"+str(split_number)
    train_images = data[index].get("train_images")
    train_masks = data[index].get("train_masks")
    val_images = data[index].get("val_images")
    val_masks = data[index].get("val_masks")
    test_images = data[index].get("test_images")
    test_masks = data[index].get("test_masks")
    
    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def getalldata(images, masks, data_percs, splits, seed):
    images_dict = {}
    masks_dict = {}
    split_dicts = {}
    for i in range(len(data_percs)):
        assert images.shape[0] == masks.shape[0]
        amount = int(images.shape[0] * data_percs[i])
        random.seed(seed)
        ind = random.sample(range(images.shape[0]), amount)
        images_dict[i] = images[ind]
        masks_dict[i] = masks[ind]

    for j in range(len(images_dict)):
        split_dicts[str(data_percs[j]) + "Perc"] = get_splits(images_dict[j], masks_dict[j], splits, seed)

    return split_dicts
