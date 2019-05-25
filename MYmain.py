from MYmodel import *
import PGM_Loader_new as pgm
from functions import *
import scoring_utils as su
#import torch
from sklearn.metrics import roc_curve, auc


train_val_split = 0.2 # how much is val
epochs = 5
batch_size = 32
filters = 64
# max 5 layers with 96x96
dropout_rate = 0.5
whichmodel = "param_unet"

all_results = []

layers_arr = [5] # 1 layer = 2 Convolutions, 1 MaxPooling, 1 Dropout (technically 4 layers)
splits = {1: (0.2, 0.2), 2: (0.3, 0.3)}

(images, masks) = pgm.get_data()
images = np.array(images)
masks = np.array(masks)

train_images, train_masks, val_images, val_masks, test_images, test_masks = get_splits(images, masks, splits)

for layers in layers_arr:
    for split in range(len(splits)):

        train_images = train_images[split]
        train_masks = train_masks[split]
        val_images = val_images[split]
        val_masks = val_masks[split]
        test_images = test_images[split]
        test_masks = test_masks[split]

        train_images = np.array(train_images)
        test_images = np.array(test_images)
        train_images = train_images[...,np.newaxis]
        test_images = test_images[...,np.newaxis]
        # train_images = np.expand_dims(train_images,-1) --> add dimension after the last dimension

        train_masks = np.array(train_masks)
        test_masks = np.array(test_masks)
        train_masks = train_masks[...,np.newaxis]
        test_masks = test_masks[..., np.newaxis]
        print('Input Shape: ' + str(train_images.shape))
        print('Mask Shape: ' + str(train_masks.shape))


        input_size = train_images.shape[1:4]


        model = param_unet(input_size, filters, layers, dropout_rate)
        model_checkpoint = ModelCheckpoint('unet.{epoch:02d}.hdf5', monitor='loss',verbose=0, save_best_only=True)
        history = model.fit(train_images, train_masks, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_masks), verbose=1, shuffle=True, callbacks=[model_checkpoint])


        results = model.predict(test_images, verbose=1)
        np.save("data" ,results)



        path = "results"
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path + '/' + whichmodel):
            os.makedirs(path + '/' + whichmodel)
        if not os.path.exists(path + '/'+whichmodel+'/'+str(split)+'train-test-val-split'):
            os.makedirs(path + '/'+whichmodel+'/'+ str(split)+'train-test-val-split')
        if not os.path.exists(path + '/'+whichmodel+'/'+ str(split)+'train-test-val-split/' + str(layers*4)+'layers'):
            os.makedirs(path + '/'+whichmodel+'/'+ str(split)+'train-test-val-split/' + str(layers*4)+'layers')
        save_dir = path + '/' + whichmodel +'/' + str(split)+'train-test-val-split/'+ str(layers*4)+'layers'



        plt.imshow(test_images[0][:,:,0], cmap='gray')
        plt.show()
        plt.imshow(test_masks[0][:,:,0], cmap='gray')
        plt.show()
        plt.imshow(results[0][:,:,0], cmap='gray')
        plt.show()



        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.path.join(save_dir, str(epochs) +'epochs_accuracy_values.png'))
        plt.show()



        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.path.join(save_dir, str(epochs) +'epochs_loss_values.png'))
        plt.show()



        mask_prediction = []
        for i in test_images:
            i = np.expand_dims(i, 0)
            mask_prediction.append(model.predict(i, batch_size=1, verbose=1))

        # TODO: Implement 32 piece batches

        mask_prediction = np.array(mask_prediction)
        mask_prediction = np.squeeze(mask_prediction, 1)
        np.save(os.path.join(save_dir, str(epochs) + 'epochs_mask_prediction'), mask_prediction)


        results = {
            "median_ROC_AUC_": "median_ROC_AUC",
            "train-test-val-split" : split,
            "epochs": epochs,
            "dice": "dice",
            "roc_auc": "roc_auc",
            "median_dice_score": "median_dice_score",
            "validation_split_val": train_val_split,
            "unet_layers": layers,
            "filters": filters,
            "input_size": input_size
        }
        dice = []
        roc_auc = []

        # TODO: make this work (both DICE and ROC dont work yet)

        output = np.squeeze(mask_prediction)
        test_masks_temp = np.squeeze(test_masks)
        for s in range(test_masks.shape[0]):
            dice.append(su.dice(output[s,:], test_masks_temp[s,:]))
            #fpr, tpr, thresholds = roc_curve(test_masks_temp[s,:], output[s,:])
            #roc_auc.append(auc(fpr, tpr))

        median_ROC_AUC = np.median(roc_auc)
        median_dice_score = np.median(dice)

        results["median_dice_score"] = median_dice_score
        results["median_ROC_AUC"] = median_ROC_AUC
        results["dice"] = dice
        results["roc_auc"] = roc_auc
        results["epochs"] = epochs
        #torch.save(results, os.path.join(save_dir, str(epochs) + 'epochs_evaluation_results'))

        #TODO: torch library doesnt work yet

        print('DICE SCORE: ' + str(median_dice_score))
        print('ROC AUC:', str(median_ROC_AUC))


        output = np.expand_dims(output, -1)
        save_datavisualisation3(test_images, test_masks, output, save_dir+'/' + str(layers)+'layers', True, True)

        all_results.append(results)


best_idx = np.argmax([dict["median_dice_score"] for dict in all_results])

print(' BEST MEDIAN DICE SCORE:', all_results[best_idx]["median_dice_score"], 'with', all_results[best_idx]["train_test_split"],
          'test split, epochs = ',
          all_results[best_idx]["epochs"])
