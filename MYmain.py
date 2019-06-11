from keras.models import *
from keras.callbacks import ModelCheckpoint
from MYmodel import param_unet, segnet
import PGM_Loader_new as pgm
import os
from functions import *
import scoring_utils as su
from matplotlib import pyplot as plt
import pandas as pd

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

#from tensorflow.python.client import device_lib

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

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

filters = 64
dropout_rate = 0.5
cropper_size = 64 # --> 128x128 after padding

whichmodel = 'param_unet'
#whichmodel = 'segnet'
splits = {1:(0.3,0.1)}

epochs = 100
basic_batch_size = 32
seeds = [1, 2, 3, 4]
data_percs = [1, 0.75, 0.5, 0.25] # PERCENTAGE OF PEOPLE (TOTAL DATA)
layers_arr = [7, 6, 5]
loss_funcs = ["weighted_cross_entropy", "binary_crossentropy", "dice"]
patient_percs = [1]

all_results = []

for split in splits:
    for loss_func in loss_funcs:
        for layers in layers_arr:
            if layers > 4:
                batch_size = int((0.5**(layers-4))*basic_batch_size)
            else:
                batch_size = basic_batch_size
            for perc in data_percs:
                for patient_perc in patient_percs:
                    for seed in seeds:
                        reset_keras()
                        random.seed(seed)
                        np.random.seed(seed)
                        tf.set_random_seed(seed)
                        os.environ['PYTHONHASHSEED'] = str(seed)


                        (pgm_images, pgm_masks) = pgm.get_data(cropper_size)

                        train_pats, test_pats, val_pats = get_patient_split(len(pgm_images), splits.get(split))

                        train_pats = get_total_perc_pats(train_pats, perc)
                        test_pats = get_total_perc_pats(test_pats, 1)
                        val_pats = get_total_perc_pats(val_pats, 1)

                        train_images, train_masks = get_patient_perc_split(pgm_images, pgm_masks, train_pats, patient_perc)
                        test_images, test_masks = get_patient_perc_split(pgm_images, pgm_masks, test_pats, 1)
                        val_images, val_masks = get_patient_perc_split(pgm_images, pgm_masks, val_pats, 1)

                        input_size = train_images.shape[1:4]

                        print("****************************************************************************************************************")
                        print("EXPERIMENT:", layers, "layers,", perc*100, "% total data,", patient_perc*100, "% per pat,", loss_func, "loss function,", "seed", seed)
                        print("TRAIN DATA SIZE", train_images.shape[0])
                        print("VALIDATION DATA SIZE", val_images.shape[0])
                        print("TEST DATA SIZE", test_images.shape[0])
                        print("****************************************************************************************************************")

                        index = 1
                        path = "results/new/" + whichmodel + '-' + str(epochs) + "_epochs" + '-' + loss_func + '-' + \
                               str(int(perc*100)) + '%_total_data' + '-' + \
                               str(int(patient_perc*100)) + '%_per_pat' + '-' + str(splits.get(split)) + '_split' \
                               + '-' + str(layers) + '_layers' + '-' + 'seed_' + str(seed)
                        if not os.path.exists(path + "-" + str(index) + '/'):
                            os.makedirs(path + '-' + str(index) + '/')
                            save_dir = path + '-' + str(index) + '/'
                        else:
                            while os.path.exists(path + '-' + str(index) + '/'):
                                index += 1
                            os.makedirs(path + '-' + str(index) + '/')
                            save_dir = path + '-' + str(index) + '/'

                        if whichmodel == "param_unet":
                            model = param_unet(input_size, filters, layers, dropout_rate, loss_func)
                        if whichmodel == "segnet":
                            model = segnet(input_size, 3, dropout_rate, loss_func)

                        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=20, verbose=1,
                                                           mode='min', baseline=0.3, restore_best_weights=False)

                        history = model.fit(train_images, train_masks, epochs=epochs, batch_size=batch_size,
                                            validation_data=(val_images, val_masks), verbose=2,
                                            shuffle=True, callbacks=[es])

                        results = model.predict(test_images, verbose=0)
                        np.save("data", results)

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
                            mask_prediction.append(model.predict(i, verbose=0))

                        rounded_pred = threshold(mask_prediction, 0.5, 0.5)

                        mask_prediction = np.array(mask_prediction)
                        mask_prediction = np.squeeze(mask_prediction, 1)
                        np.save(os.path.join(save_dir, str(epochs) + 'epochs_mask_prediction'), mask_prediction)
                        np.save(os.path.join(save_dir, str(epochs) + 'test_images'), test_images)
                        np.save(os.path.join(save_dir, str(epochs) + 'test_masks'), test_masks)
                        np.save(os.path.join(save_dir, str(epochs) + 'rounded_mask_pred'), rounded_pred)

                        results = {
                            "epochs": epochs,
                            "median_dice_score": "median_dice_score",
                            "median_matthew_coeff": "median_matthew_coeff",
                            "train_val_test_split": split,
                            "unet_layers": layers,
                            "data_perc": perc*100,
                            "seed": seed,
                            "filters": filters,
                            "loss_function": loss_func,
                            "patient_perc": patient_perc
                        }
                        dice = []
                        m_coeffs = []
                        output = np.squeeze(rounded_pred)
                        test_masks_temp = np.squeeze(test_masks)
                        for s in range(test_masks.shape[0]):
                            dice.append(getdicescore(test_masks_temp[s, :, :], output[s, :, :]))
                            #m_coeffs.append(matthews_coeff(test_masks_temp[s, :, :], output[s, :, :]))

                        median_dice_score = np.median(dice)
                        median_matthew_coeff = np.median(m_coeffs)

                        results["median_dice_score"] = median_dice_score
                        results["median_matthew_coeff"] = median_matthew_coeff

                        results_file = open(save_dir + "/results.txt", "w+")
                        results_file.write(str(results))
                        results_file.close()

                        output = np.expand_dims(output, -1)
                        #save_visualisation(test_images, test_masks, mask_prediction, output, median_matthew_coeff, 'matthews', save_dir)
                        save_visualisation(test_images, test_masks, mask_prediction, output, median_dice_score, 'dice', save_dir)
                        all_results.append(results)

median_matthew_coeffs = []
best_idx = np.argmax([dict["median_matthew_coeff"] for dict in all_results])
median_matthew_coeffs.append([dict["median_matthew_coeff"] for dict in all_results])
median_matthew_coeffs = np.array(median_matthew_coeffs, float)

test_list = np.ndarray.tolist(median_matthew_coeffs.squeeze(0))

df = pd.DataFrame({"Coefficients": test_list})
bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
plt.hist(df.values, bins=bins, edgecolor="k")
plt.xticks(bins)
plt.ylabel("# of Experiments")
plt.xlabel("Matthews Coefficient")
#plt.savefig('results/' + str(epochs) + 'matthews_coeffs.png')

print(' BEST MEDIAN DICE SCORE:', round(all_results[best_idx]["median_dice_score"], 2),
      ' BEST MEDIAN MATTHEW COEFF:', round(all_results[best_idx]["median_matthew_coeff"], 2),
      'with', all_results[best_idx]["train_val_test_split"],
      'split with seed:', all_results[best_idx]["seed"],
      'using', all_results[best_idx]["loss_function"], 'as loss function',
      'and', all_results[best_idx]["data_perc"],
      '% of the total data with', all_results[best_idx]["patient_perc"]*100,
      '% data per patient after', all_results[best_idx]["epochs"],  "epochs")

final_results_file = open("results/bestresults.txt", "w+")
final_results_file.write(' BEST MEDIAN DICE SCORE: ' + str(round(all_results[best_idx]["median_dice_score"], 2)) +
                         ' BEST MEDIAN MATTHEW COEFF:' + str(round(all_results[best_idx]["median_matthew_coeff"], 2)) +
                         ' with ' + str(all_results[best_idx]["train_val_test_split"]) +
                         ' split with seed: ' + str(all_results[best_idx]["seed"]) +
                         ' using ' + str(all_results[best_idx]["loss_function"]) + ' as loss function ' +
                         ' and ' + str(all_results[best_idx]["data_perc"]) +
                         '% of the total data with ' + str(all_results[best_idx]["patient_perc"]*100) +
                         '% data per patient after ' + str(all_results[best_idx]["epochs"]) + " epochs")
final_results_file.close()
