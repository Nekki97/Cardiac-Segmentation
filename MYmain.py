from keras.models import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from MYmodel import param_unet, segnet
import PGM_Loader_new as pgm
import os
from functions import *
from matplotlib import pyplot as plt
import NII_loader as nii
import pandas as pd
from medpy.metric.binary import dc, hd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow as tf

from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# import Nii_Loader as nii
# import torch
from sklearn.metrics import roc_curve, auc

from tensorflow.python.client import device_lib

#print(device_lib.list_local_devices())

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

    #print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

filters = 64
dropout_rate = 0.5

whichmodels = ['param_unet']
splits = {1:(0.3,0.1)}

maxepochs = 500
basic_batch_size = 24
seeds = [1,2,3,4,5,6,7,8,9,10,11,12]
data_percs = [1, 0.75, 0.5, 0.25] # PERCENTAGE OF PEOPLE (TOTAL DATA)
levels_arr = [4]
loss_funcs = ["binary_crossentropy"]
patient_percs = [0.75, 0.5, 0.25]
dataset = 'pgm'

all_results = []

for whichmodel in whichmodels:
    for split in splits:
        for loss_func in loss_funcs:
            if whichmodel == 'segnet':
                levels_arr = [0]
            for levels in levels_arr:
                if levels > 4:
                    batch_size = int((0.5**(levels-4))*basic_batch_size)
                else:
                    batch_size = basic_batch_size
                for patient_perc in patient_percs:
                    for perc in data_percs:
                        for seed in seeds:
                            reset_keras()
                            random.seed(seed)
                            np.random.seed(seed)
                            tf.set_random_seed(seed)
                            os.environ['PYTHONHASHSEED'] = str(seed)

                            if dataset == 'pgm':
                                (data_images, data_masks) = pgm.get_data()
                            if dataset == 'nii':
                                (images, masks) = nii.get_nii_data()
                                data_images = []
                                data_masks = []
                                for i in range(len(images)):
                                    data_images.append(np.expand_dims(images[i], -1))
                                for i in range(len(masks)):
                                    data_masks.append(np.expand_dims(masks[i], -1))

                            train_pats, test_pats, val_pats = get_patient_split(len(data_images), splits.get(split))

                            train_pats = get_total_perc_pats(train_pats, perc)
                            test_pats = get_total_perc_pats(test_pats, 1)
                            val_pats = get_total_perc_pats(val_pats, 1)

                            train_images, train_masks = get_patient_perc_split(data_images, data_masks, train_pats, patient_perc, False)
                            test_images, test_masks = get_patient_perc_split(data_images, data_masks, test_pats, 1, True)
                            val_images, val_masks = get_patient_perc_split(data_images, data_masks, val_pats, 1, False)

                            input_size = train_images.shape[1:4]

                            print("****************************************************************************************************************")
                            print("EXPERIMENT:", levels, "levels,", perc*100, "% total data,", patient_perc*100, "% per pat,", loss_func, "loss function,", "seed", seed)
                            print("TRAIN DATA SIZE", train_images.shape[0])
                            print("VALIDATION DATA SIZE", val_images.shape[0])
                            print("TEST DATA SIZE", test_images.shape[0])
                            print("****************************************************************************************************************")

                            index = 1
                            path = "results/new/" + whichmodel + '-' + str(maxepochs) + "_maxepochs" + '-' + loss_func + '-' + \
                                   str(int(perc*100)) + '%_total_data' + '-' + \
                                   str(int(patient_perc*100)) + '%_per_pat' + '-' + str(splits.get(split)) + '_split' \
                                   + '-' + str(levels) + '_levels' + '-' + 'seed_' + str(seed)
                            if not os.path.exists(path + "-" + str(index) + '/'):
                                os.makedirs(path + '-' + str(index) + '/')
                                save_dir = path + '-' + str(index) + '/'
                            else:
                                while os.path.exists(path + '-' + str(index) + '/'):
                                    index += 1
                                os.makedirs(path + '-' + str(index) + '/')
                                save_dir = path + '-' + str(index) + '/'

                            if whichmodel == "param_unet":
                                model = param_unet(input_size, filters, levels, dropout_rate, loss_func)
                            if whichmodel == "segnet":
                                model = segnet(input_size, 3, dropout_rate, loss_func)
                            es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               min_delta=0.005,
                                                               patience=30,
                                                               verbose=1,
                                                               mode='min',
                                                               baseline=1,
                                                               restore_best_weights=False)
                            learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                                        patience=20,
                                                                        verbose=1,
                                                                        factor=0.5,
                                                                        min_lr=0.0001)
                            model_checkpoint = ModelCheckpoint('models/bestmodel.hdf5', monitor='loss', verbose=0,
                                                               save_best_only=True)
                            history = model.fit(train_images, train_masks, epochs=maxepochs, batch_size=batch_size,
                                                validation_data=(val_images, val_masks), verbose=2,
                                                shuffle=True, callbacks=[es, learning_rate_reduction, model_checkpoint])

                            mask_prediction = model.predict(test_images, verbose=0)
                            np.save("data", mask_prediction)

                            # Plot training & validation accuracy values

                            plt.plot(history.history['acc'])
                            plt.plot(history.history['val_acc'])
                            plt.title('Model accuracy')
                            plt.ylabel('Accuracy')
                            plt.xlabel('Epoch')
                            plt.legend(['Train', 'Test'], loc='upper left')
                            plt.ylim(0, 1)
                            plt.savefig(os.path.join(save_dir, str(maxepochs) + 'maxepochs_accuracy_values.png'))
                            # plt.show()

                            plt.close()

                            # Plot training & validation loss values
                            plt.plot(history.history['loss'])
                            plt.plot(history.history['val_loss'])
                            plt.title('Model loss')
                            plt.ylabel('Loss')
                            plt.xlabel('Epoch')
                            plt.legend(['Train', 'Test'], loc='upper left')
                            plt.ylim(0, 1)
                            plt.savefig(os.path.join(save_dir, str(maxepochs) + 'maxepochs_loss_values.png'))
                            # plt.show()

                            plt.close()

                            rounded_pred = threshold(mask_prediction, 0.5, 0.5)
                            rounded_pred = np.squeeze(rounded_pred, 3)
                            mask_prediction = np.array(mask_prediction)
                            mask_prediction = np.squeeze(mask_prediction, 3)

                            np.save(os.path.join(save_dir, str(maxepochs) + 'maxepochs_mask_prediction'), mask_prediction)
                            np.save(os.path.join(save_dir, str(maxepochs) + 'test_images'), test_images)
                            np.save(os.path.join(save_dir, str(maxepochs) + 'test_masks'), test_masks)
                            np.save(os.path.join(save_dir, str(maxepochs) + 'rounded_mask_pred'), rounded_pred)

                            results = {
                                "maxepochs": maxepochs,
                                "median_dice_score": "median_dice_score",
                                "median_rounded_dice_score": "median_rounded_dice_score",
                                "train_val_test_split": split,
                                "unet_levels": levels,
                                "data_perc": perc*100,
                                "seed": seed,
                                "filters": filters,
                                "loss_function": loss_func,
                                "patient_perc": patient_perc,
                                "median_hausdorff": "median_hausdorff",
                                "median_thresholded_hausdorff": "median_thresholded_hausdorff",
                            }
                            rounded_dice = []
                            dice = []
                            thresholded_hausdorff = []
                            hausdorff = []
                            for s in range(test_masks.shape[0]):
                                    rounded_dice.append(getdicescore(test_masks[s, :, :, 0], rounded_pred[s, :, :]))
                                    dice.append(getdicescore(test_masks[s, :, :, 0], mask_prediction[s, :, :]))
                                    if np.max(rounded_pred[s, :, :]) != 0:
                                        thresholded_hausdorff.append(hd(rounded_pred[s, :, :], test_masks[s, :, :, 0]))
                                    if np.max(rounded_pred[s, :, :]) != 0:
                                        hausdorff.append(hd(rounded_pred[s, :, :], test_masks[s, :, :, 0]))

                            median_rounded_dice_score = np.median(rounded_dice)
                            median_dice_score = np.median(dice)
                            median_thresholded_hausdorff = np.mean(thresholded_hausdorff)
                            median_hausdorff = np.median(hausdorff)

                            results['median_rounded_dice_score'] = median_rounded_dice_score
                            results["median_dice_score"] = median_dice_score
                            results['median_thresholded_hausdorff'] = median_thresholded_hausdorff
                            results['median_hausdorff'] = median_hausdorff

                            results_file = open(save_dir + "/results.txt", "w+")
                            results_file.write(str(results))
                            results_file.close()


                            median_dice_score = round(median_dice_score, 3)
                            median_thresholded_hausdorff = round(median_thresholded_hausdorff, 2)
                            median_rounded_dice_score = round(median_rounded_dice_score, 3)
                            median_hausdorff = round(median_hausdorff, 2)

                            save_visualisation(test_images[:,:,:,0], test_masks[:,:,:,0], mask_prediction, rounded_pred,
                                               median_dice_score, median_rounded_dice_score, median_hausdorff, median_thresholded_hausdorff, save_dir)
                            all_results.append(results)

                            print('DICE SCORE: ' + str(median_dice_score))
                            print('HAUSDORFF DISTANCE', str(median_thresholded_hausdorff))

best_idx = np.argmax([dict["median_dice_score"] for dict in all_results])
print(' BEST MEDIAN DICE SCORE:', round(all_results[best_idx]["median_dice_score"], 2),
      'with', all_results[best_idx]["train_val_test_split"],
      'split with seed:', all_results[best_idx]["seed"],
      'using', all_results[best_idx]["loss_function"], 'as loss function',
      'and', all_results[best_idx]["data_perc"],
      '% of the total data with', all_results[best_idx]["patient_perc"]*100,
      '% data per patient after', all_results[best_idx]["maxepochs"],  "maxepochs")

final_results_file = open("results/bestresults.txt", "w+")
final_results_file.write(' BEST MEDIAN DICE SCORE: ' + str(round(all_results[best_idx]["median_dice_score"], 2)) +
                         ' with ' + str(all_results[best_idx]["train_val_test_split"]) +
                         ' split with seed: ' + str(all_results[best_idx]["seed"]) +
                         ' using ' + str(all_results[best_idx]["loss_function"]) + ' as loss function ' +
                         ' and ' + str(all_results[best_idx]["data_perc"]) +
                         '% of the total data with ' + str(all_results[best_idx]["patient_perc"]*100) +
                         '% data per patient after ' + str(all_results[best_idx]["maxepochs"]) + " maxepochs")
final_results_file.close()
