from matplotlib import pyplot as plt
import numpy as np
import random
import imageio
from sklearn import model_selection

def save_datavisualisation3(img_data, myocar_labels, predicted_labels, save_folder, index_first = False, normalized = False):
    img_data_temp = []
    myocar_labels_temp = []
    predicted_labels_temp = []
    if index_first == True:
        for i in range(0, len(img_data)):
            img_data_temp.append(np.moveaxis(img_data[i], 0, -1))
            myocar_labels_temp.append(np.moveaxis(myocar_labels[i], 0, -1))
            predicted_labels_temp.append(np.moveaxis(predicted_labels[i], 0, -1))
    counter = 0
    for i, j, k in zip(img_data_temp[:], myocar_labels_temp[:], predicted_labels_temp[:]):
        print(counter)
        print(i.shape)
        i_patch = i[:, :, 0]
        if normalized == True:
            i_patch = i_patch*255
        # np.squeeze(i_patch)

        j_patch = j[:, :, 0]
        # np.squeeze(j_patch)
        j_patch = j_patch * 255

        k_patch = k[:,:,0]
        k_patch = k_patch*255

        for slice in range(1, i.shape[2]):
            temp = i[:, :, slice]
            # np.squeeze(temp)
            if normalized == True:
                temp = temp * 255
            i_patch = np.hstack((i_patch, temp))


            temp = j[:, :, slice]
            # np.squeeze(temp)
            temp = temp * 255
            j_patch = np.hstack((j_patch, temp))

            temp = k[:,:,slice]
            temp = temp*255
            k_patch = np.hstack((k_patch, temp))

        image = np.vstack((i_patch, j_patch, k_patch))

        print(image.shape)
        imageio.imwrite(save_folder + '%d.png' % (counter,), image)
        counter = counter + 1

'''
plt.imshow(test_images[0][:,:,0], plt.cm.gray)
plt.show()
plt.imshow(test_masks[0], plt.cm.gray)
plt.show()
plt.imshow(results[0][:,:,0], plt.cm.gray)
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

'''

def split(images, masks, perc):
    '''
    :param perc: percentage of total data that are turned into test-dataset
    '''
    imagecount = len(images)
    print(imagecount)

    seed = 698869
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.shuffle(masks)

    testdata_count = int(imagecount * perc)
    traindata_count = imagecount - testdata_count

    train_images = []
    test_images = []
    train_masks = []
    test_masks = []
    for image in images[:traindata_count]:
        train_images.append(image)
    for mask in masks[:traindata_count]:
        train_masks.append(mask)
    for image in images[traindata_count:traindata_count+testdata_count]:
        test_images.append(image)
    for mask in masks[traindata_count:traindata_count+testdata_count]:
        test_masks.append(mask)
    print(len(train_images))
    print(len(train_masks))
    print(len(test_images))
    print(len(test_masks))

    return train_images, train_masks, test_images, test_masks


def get_splits(images, masks, splits):
    # splits in form of {1: (0.2,0.2), 2: (0.3,0.4)} (die Erste ist train-test, die Zweite train-val)
    arr_train_images = []
    arr_train_masks = []
    arr_test_images = []
    arr_test_masks = []
    arr_val_images = []
    arr_val_masks = []

    for split in splits.values():     #TODO Look at it

        train_val_images, test_images, train_val_masks, test_masks = \
            model_selection.train_test_split(images, masks, test_size=split[0])

        train_images, val_images, train_masks, val_masks = \
            model_selection.train_test_split(train_val_images, train_val_masks, test_size=split[1])

        print(len(train_images))
        print(train_images.shape())
        np.concatenate(arr_train_images, train_images)
        np.concatenate(arr_train_masks, train_masks)
        np.concatenate(arr_test_images, test_images)
        np.concatenate(arr_test_masks, test_masks)
        np.concatenate(arr_val_images, val_images)
        np.concatenate(arr_val_masks, val_masks)

    return arr_train_images, arr_train_masks, arr_val_images, arr_val_masks, arr_test_images, arr_test_masks
