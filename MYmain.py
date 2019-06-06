from keras.models import *
from keras.callbacks import ModelCheckpoint
from MYmodel import param_unet, segnet
import PGM_Loader_new as pgm
import os
from functions import *
import scoring_utils as su
from matplotlib import pyplot as plt

from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow

# import Nii_Loader as nii
# import torch
from sklearn.metrics import roc_curve, auc
#from numba import guvectorize

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

import gc

def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))

filters = 64
dropout_rate = 0.5  # TODO: change to 0.4
cropper_size = 64 # --> 128x128 after padding

whichmodel = 'param_unet'
#whichmodel = 'segnet'
splits = {1: (0.3, 0.1)}

epochs = 100
basic_batch_size = 12 # auf 0.75*self ab 5 layers
seeds = [1, 2, 3]
data_percs = [0.25, 0.5, 0.75, 1] # PERCENTAGE OF PEOPLE (TOTAL DATA)
layers_arr = [3]
loss_funcs = ["weighted_cross_entropy"]
patient_percs = [0.25, 0.5, 0.75, 1]

#TODO: anfangen plots zu machen mit matplotlib

all_results = []

for layers in layers_arr:
    if layers > 4:
        batch_size = int(0.5*basic_batch_size)
    else:
        batch_size = basic_batch_size
    for loss_func in loss_funcs:
        for seed in seeds:
            for split_number in range(len(splits)):
                for perc_index in range(len(data_percs)):
                    for patient_perc in patient_percs:

                        reset_keras()

                        (pgm_images, pgm_masks) = pgm.get_data(cropper_size)
                        pgm_images, pgm_masks = getpatpercs(pgm_images, pgm_masks, patient_perc)

                        for patient in range(len(pgm_images)):
                            pgm_images[patient] = np.array(pgm_images[patient])
                            pgm_masks[patient] = np.array(pgm_masks[patient])

                        data_dict = getalldata(pgm_images, pgm_masks, data_percs, splits, seed)
                        data = data_dict[str(data_percs[perc_index])+"Perc"]
                        train_images, train_masks, val_images, val_masks, test_images, test_masks = \
                            get_datasets(data, split_number)

                        train_images = np.array(train_images, dtype=float)
                        test_images = np.array(test_images, dtype=float)
                        val_images = np.array(val_images, dtype=float)
                        # train_images = np.expand_dims(train_images,-1) --> add dimension after the last dimension

                        train_masks = np.array(train_masks, dtype=float)
                        test_masks = np.array(test_masks, dtype=float)
                        val_masks = np.array(val_masks, dtype=float)

                        print('Input Shape: ' + str(train_images.shape))
                        print('Mask Shape: ' + str(train_masks.shape))

                        input_size = train_images.shape[1:4]
                        train_val_test_split = splits.get(split_number + 1)

                        print("EXPERIMENT:", layers, "layers", patient_perc*100, "% per pat",
                              data_percs[perc_index]*100, "% total data")
                        print("TOTAL DATA SIZE", train_images.shape[0]+val_images.shape[0]+test_images.shape[0])

                        index = 1
                        path = "results/" + whichmodel + '-' + str(epochs) + "_epochs" + '-' + loss_func + '-' + \
                               str(int(data_percs[perc_index]*100)) + '%_total_data' + '-' + \
                               str(int(patient_perc*100)) + '%_per_pat' + '-' + str(train_val_test_split) + '_split' \
                               + '-' + str(layers) + '_layers' + '-' + 'seed_' + str(seed)
                        if not os.path.exists(path + "-" + str(index) + '/'):
                            os.makedirs(path + '-' + str(index) + '/')
                            save_dir = path + '-' + str(index) + '/'
                        else:
                            while os.path.exists(path + '-' + str(index) + '/'):
                                index += 1
                            os.makedirs(path + '-' + str(index)+ '/')
                            save_dir = path + '-' + str(index) + '/'

                        '''
                        if not os.path.exists(path):
                            os.makedirs(path)
                        path += '/' + whichmodel
                        if not os.path.exists(path):
                            os.makedirs(path)
                        path += '/' + loss_func
                        if not os.path.exists(path):
                            os.makedirs(path)
                        path += '/' + str(int(data_percs[perc_index]*100)) + '%_total_data'
                        if not os.path.exists(path):
                            os.makedirs(path)
                        path += '/' + str(int(patient_perc*100)) + '%_per_pat'
                        if not os.path.exists(path):
                            os.makedirs(path)
                        path += '/' + str(train_val_test_split) + '_split'
                        if not os.path.exists(path):
                            os.makedirs(path)
                        path += '/' + str(layers) + '_layers'
                        if not os.path.exists(path):
                            os.makedirs(path)
                        path += '/' + 'seed_' + str(seed)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        if not os.path.exists(path + '/' + str(index)):
                            os.makedirs(path + '/' + str(index))
                            save_dir = path + '/' + str(index)
                        else:
                            while os.path.exists(path + '/' + str(index)):
                                index += 1
                            os.makedirs(path + '/' + str(index))
                            save_dir = path + '/' + str(index)
                        '''
                        if whichmodel == "param_unet":
                            model = param_unet(input_size, filters, layers, dropout_rate, loss_func)
                        if whichmodel == "segnet":
                            model = segnet(input_size, 3, dropout_rate, loss_func)

                        history = model.fit(train_images, train_masks, epochs=epochs, batch_size=batch_size,
                                            validation_data=(val_images, val_masks), verbose=2, shuffle=True)

                        results = model.predict(test_images, verbose=0)
                        np.save("data", results)

                        '''
                        save_datavisualisation2(train_images, train_masks, save_dir, "train", 12, 3, 0.2, True)
                        save_datavisualisation2(test_images, test_masks, save_dir, "test", 12, 3, 0.2, True)
                        save_datavisualisation2(val_images, val_masks, save_dir, "val", 12, 3, 0.2, True)
                        plt.imshow(test_images[0][:, :, 0], cmap='gray')
                        plt.show()
                        plt.imshow(test_masks[0][:, :, 0], cmap='gray')
                        plt.show()
                        plt.imshow(results[0][:, :, 0], cmap='gray')
                        plt.show()
                        '''

                        # Plot training & validation accuracy values
                        plt.plot(history.history['acc'])
                        plt.plot(history.history['val_acc'])
                        plt.title('Model accuracy')
                        plt.ylabel('Accuracy')
                        plt.xlabel('Epoch')
                        plt.legend(['Train', 'Test'], loc='upper left')
                        plt.savefig(os.path.join(save_dir, str(epochs) + 'epochs_accuracy_values.png'))
                        # plt.show()

                        plt.close()

                        # Plot training & validation loss values
                        plt.plot(history.history['loss'])
                        plt.plot(history.history['val_loss'])
                        plt.title('Model loss')
                        plt.ylabel('Loss')
                        plt.xlabel('Epoch')
                        plt.legend(['Train', 'Test'], loc='upper left')
                        plt.savefig(os.path.join(save_dir, str(epochs) + 'epochs_loss_values.png'))
                        # plt.show()

                        plt.close()

                        mask_prediction = []
                        for i in test_images:
                            i = np.expand_dims(i, 0)
                            mask_prediction.append(model.predict(i, batch_size=1, verbose=0))

                        mask_prediction = np.array(mask_prediction)
                        mask_prediction = np.squeeze(mask_prediction, 1)
                        np.save(os.path.join(save_dir, str(epochs) + 'epochs_mask_prediction'), mask_prediction)
                        np.save(os.path.join(save_dir, str(epochs) + 'test_images'), test_images)
                        np.save(os.path.join(save_dir, str(epochs) + 'test_masks'), test_masks)

                        results = {
                            "epochs": epochs,
                            #"dice": "dice",
                            "median_dice_score": "median_dice_score",
                            "train_val_test_split": train_val_test_split,
                            "unet_layers": layers,
                            "data_perc": data_percs[perc_index],
                            "seed": seed,
                            "filters": filters,
                            "loss_function": loss_func,
                            "patient_perc": patient_perc
                        }
                        dice = []
                        #roc_auc = []
                        output = np.squeeze(mask_prediction)
                        test_masks_temp = np.squeeze(test_masks)
                        print(output.shape, "OUTPUT SHAPE")
                        print(test_masks_temp.shape, "TESTMASKTEMP SHAPE")
                        for s in range(test_masks.shape[0]):
                            dice.append(getdicescore(test_masks_temp[s, :, :], output[s, :, :]))
                            #fpr, tpr, thresholds = roc_curve(test_masks_temp[s, :, :], output[s, :, :])
                            #roc_auc.append(auc(fpr, tpr))

                        median_dice_score = np.median(dice)
                        #median_roc_score = np.median(roc_auc)

                        results["median_dice_score"] = median_dice_score

                        results_file = open(save_dir + "/results.txt", "w+")
                        results_file.write(str(results))
                        results_file.close()

                        #roc_auc_score = roc_auc_score(test_masks_temp[0, :, :], output[0, :, :])

                        #print('ROC SCORE: ' + str(median_roc_score))
                        print('DICE SCORE: ' + str(median_dice_score))

                        output = np.expand_dims(output, -1)
                        #save_datavisualisation3(test_images, test_masks, output, save_dir, "prediction", 12, 2, 0.4, True)

                        all_results.append(results)


best_idx = np.argmax([dict["median_dice_score"] for dict in all_results])

print(' BEST MEDIAN DICE SCORE:', round(all_results[best_idx]["median_dice_score"],2),
      'with', all_results[best_idx]["train_val_test_split"],
      'split with seed:', all_results[best_idx]["seed"],
      'using', all_results[best_idx]["loss_function"], 'as loss function',
      'and', all_results[best_idx]["data_perc"]*100,
      '% of the total data with', all_results[best_idx]["patient_perc"]*100,
      '% data per patient after', all_results[best_idx]["epochs"],  "epochs")

final_results_file = open("results/bestresults.txt", "w+")
final_results_file.write(' BEST MEDIAN DICE SCORE: ' + str(round(all_results[best_idx]["median_dice_score"],2)) +
                         ' with ' + str(all_results[best_idx]["train_val_test_split"]) +
                         ' split with seed: ' + str(all_results[best_idx]["seed"]) +
                         ' using ' + str(all_results[best_idx]["loss_function"]) + ' as loss function ' +
                         ' and ' + str(all_results[best_idx]["data_perc"]*100) +
                         '% of the total data with ' + str(all_results[best_idx]["patient_perc"]*100) +
                         '% data per patient after ' + str(all_results[best_idx]["epochs"]) + " epochs")
final_results_file.close()
