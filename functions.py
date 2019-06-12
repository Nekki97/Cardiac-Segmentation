import numpy as np
import random
import keras.backend as K
import tensorflow as tf
import imageio
import sklearn


def save_visualisation(img_data, myocar_labels, predicted_labels, rounded_labels, score, score_name, save_folder):
    counter = 0
    img_data = np.moveaxis(img_data, -1, 0)
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
        print("Done visualising at", save_folder, '%s_%s.png' % (score, score_name))


def get_patient_split(pats_amount, split):

    test_perc = split[0]
    val_perc = split[1]
    train_perc = 1 - test_perc - val_perc

    test_pat_amount = max(int(test_perc * pats_amount), 1)
    val_pat_amount = max(int(val_perc * pats_amount), 1)
    train_pat_amount = max(int(train_perc * pats_amount), 1)

    indices = list(range(pats_amount))

    train_inds = random.sample(indices, train_pat_amount)
    train_pats = []
    for index in train_inds:
        train_pats.append(index)
        indices.remove(index)

    test_inds = random.sample(indices, test_pat_amount)
    test_pats = []
    for index in test_inds:
        test_pats.append(index)
        indices.remove(index)

    val_inds = random.sample(indices, val_pat_amount)
    val_pats = []
    for index in val_inds:   # remaining patients go to val (remaining after rounding train_perc and test_perc to int)
        val_pats.append(index)
        indices.remove(index)

    pat_splits = [train_pats, test_pats, val_pats]

    train_diff = train_perc - len(train_pats) / pats_amount*100
    test_diff = test_perc - len(test_pats) / pats_amount * 100
    val_diff = val_perc - len(val_pats) / pats_amount * 100
    diffs = [train_diff, test_diff, val_diff]

    if len(indices) > 0:
        pat_splits[diffs.index(max(diffs))].append(indices[0])
        diffs.remove(max(diffs))
    if len(indices) > 1:
        pat_splits[diffs.index(max(diffs))].append(indices[1])
        diffs.remove(max(diffs))
    if len(indices) > 2:
        pat_splits[diffs.index(max(diffs))].append(indices[2])

    return train_pats, test_pats, val_pats


def get_total_perc_pats(pats, perc):
    amount = max(int(perc*len(pats)), 1)
    if perc - amount / len(pats) >= 0.5 * 1 / len(pats):
        amount += 1
    perc_pats = random.sample(pats, amount)
    return perc_pats


def get_patient_perc_split(total_imgs, total_masks, pats, perc, test):
    images = []
    masks = []
    img_slices = []
    mask_slices = []
    for patient in pats:
        amount = max(int(perc * min_length), 1)
        if perc - amount / min_length >= 0.5 * 1 / min_length:
            amount += 1
        indices = random.sample(range(min_length), amount)
        for index in indices:
            img_slices.append(total_imgs[patient][index])
            mask_slices.append(total_masks[patient][index])
    if test:
        images.append(np.array(img_slices, dtype=float))
        masks.append(np.array(mask_slices, dtype=float))
    else:
        for slice in img_slices:
            images.append(slice)
        images = np.array(images, dtype=float)
        for slice in mask_slices:
            masks.append(slice)
        masks = np.array(masks, dtype=float)
    return images, masks


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


def dice_coeff(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(y_true, -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coeff_loss(y_true, y_pred):
    return 1-dice_coeff(y_true, y_pred)


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

'''
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
