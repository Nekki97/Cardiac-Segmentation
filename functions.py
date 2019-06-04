import numpy as np
import random
#import imageio
from sklearn import model_selection
import keras.backend as K

'''
def save_datavisualisation3(img_data, myocar_labels, predited_labels, save_folder, name, bunch_size, rows,
                            percentage, normalized=False):
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
                    for i, j, k in zip(img_data[indx_start:indx_end, :, :, 0],
                                       myocar_labels[indx_start:indx_end, :, :, 0],
                                       predited_labels[indx_start:indx_end, :, :, 0]):
                        if normalized:
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
                for index in range(bunch_size * row_number + 1, bunch_size * (row_number + 1)):
                    if remaining_imgs > 0:
                        image = i_bunch[index, :, :]
                        myocar_label = j_bunch[index, :, :]
                        predicted_label = k_bunch[index, :, :]
                    else:
                        x = img_data.shape[1]
                        y = img_data.shape[2]
                        image = np.zeros((x, y))
                        myocar_label = np.zeros((x, y))
                        predicted_label = np.zeros((x, y))
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


def save_datavisualisation2(img_data, myocar_labels, save_folder, name, bunch_size, rows,
                            percentage, normalized=False):
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
                for i, j in zip(img_data[indx_start:indx_end, :, :, 0], myocar_labels[indx_start:indx_end, :, :, 0]):
                    if normalized:
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
                        image = np.zeros((x, y))
                        label = np.zeros((x, y))
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
        imageio.imwrite(save_folder + "/" + name + '%d_%d.png' % (indx_start, indx_end-1), full_image)
    print("******Finished visualising " + str(amount) + " images******")
'''


def get_split(images, masks, split, seed):
    # split in form of (0.2,0.2)
    test_amount = max(int(len(images)*split[0]), 1)
    val_amount = max(int(len(images)*split[1]), 1)
    #train_amount = len(images) - test_amount - val_amount

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
        train_images, train_masks, val_images, val_masks, test_images, test_masks = \
            get_split(images, masks, split, seed)

        '''
        print(train_images[0].shape, "train_image")
        correctarray(train_images)
        correctarray(train_masks)
        correctarray(test_masks)
        correctarray(test_images)
        correctarray(val_images)
        correctarray(val_masks)
        '''

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

    train_images = collectimages(train_images)
    train_masks = collectimages(train_masks)
    test_images = collectimages(test_images)
    test_masks = collectimages(test_masks)
    val_images = collectimages(val_images)
    val_masks = collectimages(val_masks)

    train_images = np.array(train_images, dtype=float)
    test_images = np.array(test_images, dtype=float)
    val_images = np.array(val_images, dtype=float)
    train_masks = np.array(train_masks, dtype=float)
    test_masks = np.array(test_masks, dtype=float)
    val_masks = np.array(val_masks, dtype=float)

    print(train_images.shape)
    print(test_images.shape)

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def getalldata(images, masks, data_percs, splits, seed):
    images_dict = {}
    masks_dict = {}
    split_dicts = {}
    for i in range(len(data_percs)):
        assert len(images) == len(masks)
        amount = int(len(images) * data_percs[i])
        perc = amount/len(images)*100
        remaining = data_percs[i]*100 - perc
        if remaining/100*len(images) > 0.5*images[max(amount-1,0)].shape[0]:
            amount += 1
        random.seed(seed)
        ind = random.sample(range(len(images)), amount)
        temp_imgs = []
        temp_masks = []
        for k in range(len(ind)):
            temp_imgs.append(images[k])
            temp_masks.append(masks[k])
        images_dict[i] = temp_imgs
        masks_dict[i] = temp_masks

    for j in range(len(images_dict)):
        split_dicts[str(data_percs[j]) + "Perc"] = get_splits(images_dict[j], masks_dict[j], splits, seed)

    return split_dicts


def getdicescore(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def collectimages(mylist):
    data = []
    for patient in range(len(mylist)):
        for image in range(len(mylist[patient])):
            data.append((mylist[patient])[image])
    return data


def getpatpercs(images, masks, patperc):
    new_imgs = []
    new_masks = []
    for pat in range(len(images)):
        temp_imgs = []
        temp_masks = []
        for index in range(int(patperc*len(images[pat]))):
            temp_imgs.append((images[pat])[index])
            temp_masks.append((masks[pat])[index])
        new_imgs.append(temp_imgs)
        new_masks.append(temp_masks)
    return new_imgs, new_masks

'''
def correctarray(myarray):
    min = myarray[0].shape[0]
    for pat in range(len(myarray)):
        if myarray[pat].shape[0] < min:
            min = myarray[pat].shape[0]

    for pat in range(len(myarray)):
        for j in range(pat):
            print(pat, j)
            if j == pat:
                continue
            while myarray[j].shape[0] > myarray[pat].shape[0]:
                myarray[j] = (myarray[j])[0:(myarray[j].shape[0]-1), :, :, :]
                print("from", myarray[j].shape[0], "to", myarray[j].shape[0]-1)
            while myarray[j].shape[0] < myarray[pat].shape[0]:
                myarray[pat] = (myarray[pat])[0:(myarray[pat].shape[0]-1), :, :, :]
                print("from", myarray[pat].shape[0], "to", myarray[pat].shape[0] - 1)
        print("Corrected to", myarray[pat].shape[0])
    return myarray
    '''
