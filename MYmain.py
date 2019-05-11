from MYmodel import *
import PGM_Loader as pgm
from matplotlib import pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

(train_images, train_masks) = pgm.get_data('train')
(test_images, test_masks) = pgm.get_data('train')
(val_images, val_masks) = pgm.get_data('val')

train_images = np.array(train_images)
test_images = np.array(test_images)
train_images = train_images[...,np.newaxis]
test_images = test_images[...,np.newaxis]
#train_images = np.expand_dims(train_images,-1) --> add dimension after the last dimension

train_masks = np.array(train_masks)
train_masks = train_masks[...,np.newaxis]
print('Input Shape: ' + str(train_images.shape))
print('Mask Shape: ' + str(train_masks.shape))

#TODO: find out how cluster works and run it on there (via VPN when away from ETH)

#TODO: plot loss/accuracy functions per epoch

filters = 64
# max 5 layers with 96x96
layers = 5

model = unet(train_images.shape[1:4], filters, layers)
model_checkpoint = ModelCheckpoint('unet.{epoch:02d}.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit(train_images, train_masks, steps_per_epoch=1, epochs=8, callbacks=[model_checkpoint])

results = model.predict(test_images, verbose=1)
np.save("data/test",results)

plt.imshow(test_images[0][:,:,0], plt.cm.gray)
plt.show()
plt.imshow(test_masks[0], plt.cm.gray)
plt.show()
plt.imshow(results[0][:,:,0], plt.cm.gray)
plt.show()
