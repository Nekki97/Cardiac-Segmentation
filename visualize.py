import numpy as np
import imageio
import os
import keras.backend as K
from functions import getdicescore
from functions import matthews_coeff
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


def threshold(images, upper, lower):
    images[images > upper] = 1
    images[images < lower] = 0
    return images


def save_visualisation(img_data, myocar_labels, predicted_labels, rounded_labels,m_coeff, name, save_folder):
    counter = 0
    img_data = np.moveaxis(img_data,-1,0)
    myocar_labels = np.moveaxis(myocar_labels, -1, 0)
    predicted_labels = np.moveaxis(predicted_labels, -1, 0)
    rounded_labels = np.moveaxis(rounded_labels, -1, 0)
    for i, j, k, l in zip(img_data, myocar_labels, predicted_labels, rounded_labels):
        i_patch = i[0, :, :]*255
        j_patch = j[0, :, :]*255
        k_patch = k[0, :, :]*255
        l_patch = l[0, :, :]*255
        for slice in range(1, i.shape[0]):
            i_patch = np.hstack((i_patch, i[slice, :, :]*255))
            j_patch = np.hstack((j_patch, j[slice, :, :]*255))
            k_patch = np.hstack((k_patch, k[slice, :, :]*255))
            l_patch = np.hstack((l_patch, l[slice, :, :]*255))

        image = np.vstack((i_patch, j_patch, k_patch, l_patch))

        imageio.imwrite(save_folder + '/%s_%s.png' % (round(m_coeff, 3), name), image)
        counter = counter + 1
        #print("Done visualising at", save_folder, '/%s_%s.png' % (round(m_coeff, 3), name))


def findpath(type, path):
    epochs = 0
    while not os.path.exists(path + '/' + str(epochs) + type):
        if epochs > 200:
            print("No data found")
            return False, None

        else:
            epochs += 1
    return True, path + '/' + str(epochs) + type


def read_score(filepath):
    if os.path.exists(filepath):
        result = open(filepath)
        resulttext = result.read()
        result.close()
        ind = resulttext.find("score")
        score = resulttext[ind + 8:ind + 14]
        zero = False
        nonzero = False
        if score[0].isdigit() and not score[2:].isdigit():
            zero = True
            score = score[0:2]
        if score[2:].isdigit():
            nonzero = True
        if nonzero or zero:
            return score
        else:
            return 0
    else:
        return 0


def save_rounded_pred(path):
    pred_bool, pred_path = findpath('epochs_mask_prediction.npy', path)
    rounded_pred_bool, rounded_pred_path = findpath('rounded_mask_pred.npy', path)
    if pred_bool and not rounded_pred_bool:
        predictions = np.load(pred_path)
        rounded_pred = threshold(predictions, 0.5, 0.5)
        ind = pred_path.find('epochs_mask')
        if pred_path[ind - 3].isdigit():
            epochs = pred_path[ind - 3:ind]
        else:
            epochs = pred_path[ind - 2:ind]
        np.save(os.path.join(path, str(epochs) + "rounded_mask_pred"), rounded_pred)
        #print("Rounded Prediction saved at", path)
    #elif rounded_pred_bool:
        # print("Rounded Prediction exists")


def check_scores(path):
    dice_exists = False
    m_exists = False
    for dirname, subdirlist, filelist in os.walk(path):
        for filename in filelist:
            if 'dice' in filename.lower():
                dice_exists = True
            if 'matthews' in filename.lower():
                m_exists = True
    return dice_exists, m_exists


def get_data(path):
    test_masks = []
    test_imgs = []
    predictions = []
    rounded_pred = []
    test_img_bool, test_img_path = findpath('test_images.npy', path)
    test_mask_bool, test_mask_path = findpath('test_masks.npy', path)
    pred_bool, pred_path = findpath('epochs_mask_prediction.npy', path)
    rounded_pred_bool, rounded_pred_path = findpath('rounded_mask_pred.npy', path)
    if test_img_bool and test_mask_bool and pred_bool and rounded_pred_bool:
        test_imgs = np.load(test_img_path)
        test_masks = np.load(test_mask_path)
        predictions = np.load(pred_path)
        rounded_pred = np.load(rounded_pred_path)
        if len(rounded_pred.shape) == 5:
            rounded_pred = np.squeeze(rounded_pred, 1)
    else:
        print("DATA ERROR at", path)
    return np.array(test_imgs), np.array(test_masks), np.array(predictions), np.array(rounded_pred)


def get_paths(path):
    filepaths = []
    labels = []
    for dirName, subdirList, fileList in os.walk(path):
        for subdirname in subdirList:
            if "seed" in subdirname.lower():  # check whether the file's DICOM
                filepaths.append(os.path.join(dirName, subdirname))
    return filepaths


def save_dice(paths):
    index = 0
    for path in paths:
        perc = int((index + 1) / len(nodice) * 100)
        if perc % 5 == 0: print('***********', perc, '% DICE ***********')
        index += 1
        test_imgs, test_masks, predictions, rounded_pred = get_data(path)
        dice_scores = []
        for image in range(test_masks.shape[0]):
            dice_score = getdicescore(test_masks[image, :, :, 0], rounded_pred[image, :, :, 0])
            dice_scores.append(dice_score)
        med_dice_score = np.median(dice_scores)
        save_visualisation(test_imgs, test_masks, predictions, rounded_pred, med_dice_score, 'dice', path)


def save_matthew(paths):
    index = 0
    for path in paths:
        perc = int((index + 1) / len(nom) * 100)
        if perc % 5 == 0: print('***********', perc, '% Matthews ***********')
        index += 1
        test_imgs, test_masks, predictions, rounded_pred = get_data(path)
        m_coeffs = []
        for image in range(test_masks.shape[0]):
            m_coeff = matthews_coeff(test_masks[image, :, :, 0], rounded_pred[image, :, :, 0])
            m_coeffs.append(m_coeff)
        med_m_coeff = np.median(m_coeffs)
        save_visualisation(test_imgs, test_masks, predictions, rounded_pred, med_m_coeff, 'matthews', path)


path = 'results/4_layers'

filepaths = get_paths(path)

nom = []
nodice = []

for path in filepaths:
    save_rounded_pred(path)

    dice_exists, m_exists = check_scores(path)
    if not dice_exists and m_exists:
        nodice.append(path)
    if dice_exists and not m_exists:
        nom.append(path)
    if not dice_exists and not m_exists:
        nodice.append(path)
        nom.append(path)

save_dice(nodice)
#save_matthew(nom)
