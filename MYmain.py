from MYmodel import *
import PGM_Loader_new as pgm
from functions import *
import scoring_utils as su
#import torch
#from sklearn.metrics import roc_curve, auc

epochs = 1
batch_size = 32
filters = 8
dropout_rate = 0.5
cropper_size = 64 # --> 128x128 after padding

whichmodel = "param_unet"
seed = 262626
data_percs = [0.1]
layers_arr = [2] # 1 layer = 2 Convolutions, 1 MaxPooling, 1 Dropout (technically 4 layers)
splits = {1: (0.2, 0.2)}

all_results = []

(images, masks) = pgm.get_data(cropper_size)
images = np.array(images)
masks = np.array(masks)

data_dict = getalldata(images, masks, data_percs, splits, seed)

for layers in layers_arr:
    for perc_index in range(len(data_percs)):
        for split_number in range(len(splits)):
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

            index = 1
            path = "results"
            if not os.path.exists(path):
                os.makedirs(path)
            path += '/' + whichmodel
            if not os.path.exists(path):
                os.makedirs(path)
            path += '/' + str(int(data_percs[perc_index]*100)) + '%_data'
            if not os.path.exists(path):
                os.makedirs(path)
            path += '/' + str(train_val_test_split) + '_split'
            if not os.path.exists(path):
                os.makedirs(path)
            path += '/' + str(layers) + '_layers_#'
            if not os.path.exists(path + str(index)):
                os.makedirs(path + str(index))
                save_dir = path + str(index)
            else:
                while os.path.exists(path + str(index)):
                    index += 1
                save_dir = path + str(index)
            if not os.path.exists(save_dir + '/checkpoints'):
                os.makedirs(save_dir + '/checkpoints')

            model = param_unet(input_size, filters, layers, dropout_rate)
            model_checkpoint = ModelCheckpoint(save_dir + '/checkpoints/unet{epoch:02d}.hdf5', monitor='loss',verbose=0, save_best_only=True)
            history = model.fit(train_images, train_masks, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_masks), verbose=1, shuffle=True, callbacks=[model_checkpoint])

            results = model.predict(test_images, verbose=1)
            np.save("data" ,results)

            '''
            save_datavisualisation2(train_images, train_masks, save_dir, "train", 12, 3, 0.2, True)
            save_datavisualisation2(test_images, test_masks, save_dir, "test", 12, 3, 0.2, True)
            save_datavisualisation2(val_images, val_masks, save_dir, "val", 12, 3, 0.2, True)

            plt.imshow(test_images[0][:,:,0], cmap='gray')
            plt.show()
            plt.imshow(test_masks[0][:,:,0], cmap='gray')
            plt.show()
            plt.imshow(results[0][:,:,0], cmap='gray')
            plt.show()
            '''

            # Plot training & validation accuracy values
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig(os.path.join(save_dir, str(epochs) +'epochs_accuracy_values.png'))
            #plt.show()
            plt.close()
            
            # Plot training & validation loss values
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig(os.path.join(save_dir, str(epochs) +'epochs_loss_values.png'))
            #plt.show()
            plt.close()
            
            mask_prediction = []
            for i in test_images:
                i = np.expand_dims(i, 0)
                mask_prediction.append(model.predict(i, batch_size=1, verbose=1))

            mask_prediction = np.array(mask_prediction)
            mask_prediction = np.squeeze(mask_prediction, 1)
            np.save(os.path.join(save_dir, str(epochs) + 'epochs_mask_prediction'), mask_prediction)

            results = {
                "epochs": epochs,
                "dice": "dice",
                "median_dice_score": "median_dice_score",
                "train_val_test_split": train_val_test_split,
                "unet_layers": layers,
                "data_perc": data_percs[perc_index],
                "filters": filters,
                "input_size": input_size
            }
            dice = []

            output = np.squeeze(mask_prediction)
            test_masks_temp = np.squeeze(test_masks)
            for s in range(test_masks.shape[0]):
                dice.append(su.dice(output[s,:], test_masks_temp[s,:]))

            median_dice_score = np.median(dice)

            results["median_dice_score"] = median_dice_score
            results["dice"] = dice

            #torch.save(results, os.path.join(save_dir, str(epochs) + 'epochs_evaluation_results'))

            #TODO: torch library doesnt work yet

            print('DICE SCORE: ' + str(median_dice_score))

            output = np.expand_dims(output, -1)
            #save_datavisualisation3(test_images, test_masks, output, save_dir, "prediction", 12, 2, 0.4, True)

            all_results.append(results)


best_idx = np.argmax([dict["median_dice_score"] for dict in all_results])

print(' BEST MEDIAN DICE SCORE:', round(all_results[best_idx]["median_dice_score"],2), 'with', all_results[best_idx]["train_val_test_split"],
          'split and ', all_results[best_idx]["data_perc"]*100, '% of the data after ', all_results[best_idx]["epochs"], " epochs")
