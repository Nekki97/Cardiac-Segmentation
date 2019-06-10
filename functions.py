import numpy as np
import random
#import imageio
from sklearn import model_selection
import keras.backend as K
import tensorflow as tf
import imageio
import os


def save_visualisation(img_data, myocar_labels, predicted_labels, rounded_labels, score, score_name, save_folder):
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
        imageio.imwrite(save_folder + '%s_%s.png' % (score,score_name), image)
        counter = counter + 1
        print("Done visualising at", save_folder, '%s_%s.png' %(score,score_name))



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
        if remaining/100*len(images) > 0.5*images[max(amount-1, 0)].shape[0]:
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
    return (2. * intersection + smooth) / (K.sum(y_true, -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def weighted_cross_entropy(y_true, y_pred, beta=0.7):
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

    return tf.reduce_mean(loss)

  return loss(y_true, y_pred)


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

def matthews_coeff(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    y_true = tf.convert_to_tensor(y_true, np.float32)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return K.eval(numerator / (denominator + K.epsilon()))


def threshold(images, upper, lower):
    images = np.array(images)
    images[images > upper] = 1
    images[images < lower] = 0
    return images
