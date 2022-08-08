import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image

import rasterio

import segmentation_models as sm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import MeanIoU

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

train_image_dir = 'images_train'
train_mask_dir = 'masks_train'

test_image_dir = 'images_test'
test_mask_dir = 'masks_test'


# print(os.listdir(train_image_dir))
# print(os.listdir(train_mask_dir))
# print(os.listdir(test_image_dir))
# print(os.listdir(test_mask_dir))

patch_size = 128  # 128, 256 


# # read all the images and get patches of 128


# # with rasterio.open(image_dir + '/' + os.listdir(image_dir)[0], 'r') as ds:
# #     arr = ds.read() 
# #     img = np.moveaxis(arr, 0, -1)
# # # print(img.shape)


train_image_dataset = []
test_image_dataset = []


def make_image_patches(folder, image_dataset):

	for path, subdirs, files in os.walk(folder):
		images = os.listdir(path)

		for i, image_name in enumerate(images):

			# print(i, image_name)
			with rasterio.open(folder + '/' + image_name, 'r') as ds:
				arr = ds.read() 

			image = np.moveaxis(arr, 0, -1)
			# print(image.shape)
			# # image = cv2.imread(path+'/'+image_name, 1) # read image as RGB
			# # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
			# SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
			# SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
			# print(SIZE_X, SIZE_Y)
			# image = Image.fromarray(image)
			# image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
			#image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
			image = np.array(image)
			# print(i, image.shape)


			# extract the patched for each image
			# print(i, " Now patchifying image: ", path+"/"+image_name)
			patches_img = patchify(image, (patch_size, patch_size, 25), step=patch_size)  #Step=256 for 256 patches means no overlap


			for i in range(patches_img.shape[0]):
				for j in range(patches_img.shape[1]):

					single_patch_img = patches_img[i,j,:,:]
	                
					#Use minmaxscaler instead of just dividing by 255. 
					single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)

					#single_patch_img = (single_patch_img.astype('float32')) / 255. 
					single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
					image_dataset.append(single_patch_img)


make_image_patches(train_image_dir, train_image_dataset)
make_image_patches(test_image_dir, test_image_dataset)
train_image_dataset = np.array(train_image_dataset)
test_image_dataset = np.array(test_image_dataset)
print(train_image_dataset.shape)
print(test_image_dataset.shape)




# do the same for the masks
train_mask_dataset = []  
test_mask_dataset = []

def make_mask_patches(folder, mask_dataset):

	for path, subdirs, files in os.walk(folder):
		masks = os.listdir(path)[:100]

		for i, mask_name in enumerate(masks):

			# print(i, mask_name)

			with rasterio.open(folder + '/' + mask_name, 'r') as ds:
				arr = ds.read() 

			mask = np.moveaxis(arr, 0, -1)

			# mask = cv2.imread(path+'/'+mask_name, 1) # read image as RGB
			# mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
			# SIZE_X = (mask.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
			# SIZE_Y = (mask.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
			# mask = Image.fromarray(mask)
			# mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
			# #mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
			mask = np.array(mask)
			# print(i, mask.shape)


			# extract the patched for each image
			# print(i, "Now patchifying masks:", path+"/"+mask_name)
			patches_mask = patchify(mask, (patch_size, patch_size, 1), step=patch_size)  #Step=256 for 256 patches means no overlap


			for i in range(patches_mask.shape[0]):
				for j in range(patches_mask.shape[1]):

					single_patch_mask = patches_mask[i,j,:,:]
	                
					#single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
					single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.                               
					mask_dataset.append(single_patch_mask) 




make_mask_patches(train_mask_dir, train_mask_dataset)
make_mask_patches(test_mask_dir, test_mask_dataset)
train_mask_dataset = np.array(train_mask_dataset)
test_mask_dataset = np.array(test_mask_dataset)
print(train_mask_dataset.shape)
print(test_mask_dataset.shape)




# ###############################################

labels = []
for i in range(train_mask_dataset.shape[0]):
	labels.append(train_mask_dataset[i]) 

labels = np.array(labels)   
# # labels = np.expand_dims(labels, axis=3)

# print("Unique labels in label dataset are: ", np.unique(labels))

n_classes = len(np.unique(labels))
print(n_classes)
from tensorflow.keras.utils import to_categorical
labels_cat = to_categorical(labels, num_classes=n_classes)


print('labesl_categorical', labels_cat.shape)



# # np.save('img_arr', train_image_dataset)
# # np.save('mask_arr', train_mask_dataset)
# # np.save('labels_arr', labels_cat)

# # print('done saving')



# Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(train_image_dataset))
print(image_number)
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(np.reshape(train_image_dataset[image_number][:,:,:3], (patch_size, patch_size, 3)))
plt.subplot(122)
plt.imshow(np.reshape(train_mask_dataset[image_number][:,:,:1], (patch_size, patch_size, 1)))
plt.savefig('newRes/sanity3456.png', bbox_inches='tight')
plt.close()




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_image_dataset, labels_cat, test_size = 0.05, random_state = 42)


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


weights = [0.125, 0.125, 0.125,0.125, 0.125, 0.125, 0.125, 0.125]
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #


IMG_HEIGHT = train_image_dataset.shape[1]
IMG_WIDTH  = train_image_dataset.shape[2]
IMG_CHANNELS = train_image_dataset.shape[3]


print(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


from unet import multi_unet_model, jacard_coef 

metrics=['accuracy', jacard_coef]


# # # # functions to load different U-Net versions

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


# # # # strategy = tf.distribute.MirroredStrategy()
# # # # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


model = get_model()
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss=total_loss, metrics=metrics)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
model.summary()



checkpoint_filepath = 'newRes/my_best_model.hdf5'

my_callbacks = [tf.keras.callbacks.EarlyStopping(
    monitor="val_jacard_coef",
    min_delta=0,
    patience=20,
    verbose=1,
    mode="max",
    baseline=None,
    restore_best_weights=True,
),
tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_jacard_coef",
    verbose=1,
    save_best_only=True,
    mode="max",
)
]


history = model.fit(X_train, y_train, 
                    batch_size=8, 
                    verbose=1, 
                    epochs= 200, 
                    validation_data=(X_test, y_test), 
                    shuffle=False,
                    callbacks=[my_callbacks])




print('-----------------------------------')
print('Test results', model.evaluate(X_test, y_test))

print(' ')
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(12,6))
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('newRes/loss_test.png', bbox_inches='tight')
plt.close()

acc = history.history['jacard_coef']
val_acc = history.history['val_jacard_coef']

plt.figure(figsize=(12,6))
plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.savefig('newRes/IoU_test.png', bbox_inches='tight')
plt.close()


# # # # saved_model = keras.models.load_model('models/model2.h5', custom_objects={'loss': total_loss})
# # # # print('saved model with best weights', saved_model.evaluate(X_test,y_test))