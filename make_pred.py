"""
Prediction using smooth tiling as descibed here
"""

import cv2
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
import segmentation_models as sm

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from smooth_tiled_predictions import predict_img_with_smooth_windowing
from simpleUnet import jacard_coef  


img = cv2.imread("cropped/crop_img3.tif", 1)
original_mask = cv2.imread("cropped/crop_mask3.png", 1)
original_mask = cv2.cvtColor(original_mask,cv2.COLOR_BGR2RGB)


print(img.shape, original_mask.shape)


from tensorflow.keras.models import load_model

model = load_model("models/UNet_upsampling_100epochs_B_focal.hdf5", compile=False)
                  
# size of patches
patch_size = 128

# Number of classes 
n_classes = 3

         
# # #################################################################################
# # #Predict patch by patch with no smooth blending
# ###########################################

SIZE_X = (img.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
SIZE_Y = (img.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
large_img = Image.fromarray(img)
large_img = large_img.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
#image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
large_img = np.array(large_img)     

print(large_img.shape)


patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
patches_img = patches_img[:,:,0,:,:,:]

patched_prediction = []
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        
        single_patch_img = patches_img[i,j,:,:,:]
        
        #Use minmaxscaler instead of just dividing by 255. 
        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        single_patch_img = np.expand_dims(single_patch_img, axis=0)
        pred = model.predict(single_patch_img)
        pred = np.argmax(pred, axis=3)
        pred = pred[0, :,:]
                                 
        patched_prediction.append(pred)

patched_prediction = np.array(patched_prediction)
patched_prediction = np.reshape(patched_prediction, [patches_img.shape[0], patches_img.shape[1], 
                                            patches_img.shape[2], patches_img.shape[3]])

unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))



# # plt.imshow(unpatched_prediction)
# # plt.savefig('unpatched_prediction.png', bbox_inches='tight')
# # plt.close()
# # plt.axis('off')
# ###################################################################################
#Predict using smooth blending

input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

# Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
# Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
predictions_smooth = predict_img_with_smooth_windowing(
    input_img,
    window_size=patch_size,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    nb_classes=n_classes,
    pred_func=(
        lambda img_batch_subdiv: model.predict((img_batch_subdiv))
    )
)


final_prediction = np.argmax(predictions_smooth, axis=2)

print(final_prediction.shape)
print(unpatched_prediction.shape)

###################
#Convert labeled images back to original RGB colored masks. 

def label_to_rgb(predicted_image):
    
    # Building = '#3C1098'.lstrip('#')
    # Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
    
    # Land = '#8429F6'.lstrip('#')
    # Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
    # Road = '#6EC1E4'.lstrip('#') 
    # Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228
    
    # Vegetation =  'FEDD3A'.lstrip('#') 
    # Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58
    
    # Water = 'E2A929'.lstrip('#') 
    # Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41
    
    # Unlabeled = '#9B9B9B'.lstrip('#') 
    # Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155



    Water = np.array((0,0,255))

    Buildings = np.array((255,255,0))

    Forest = np.array((0,255,0))  
    
    
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))
    
    segmented_img[(predicted_image == 0)] = Water
    segmented_img[(predicted_image == 1)] = Buildings
    segmented_img[(predicted_image == 2)] = Forest
    # segmented_img[(predicted_image == 3)] = Vegetation
    # segmented_img[(predicted_image == 4)] = Water
    # segmented_img[(predicted_image == 5)] = Unlabeled
    
    segmented_img = segmented_img.astype(np.uint8)
    return(segmented_img)



########################
#Plot and save results
prediction_with_smooth_blending=label_to_rgb(final_prediction)
prediction_without_smooth_blending=label_to_rgb(unpatched_prediction)


prediction_with_smooth_blending = Image.fromarray(prediction_with_smooth_blending)
prediction_with_smooth_blending = prediction_with_smooth_blending.crop((0 ,0, SIZE_X, SIZE_Y))
img = Image.fromarray(img)
img = img.crop((0 ,0, SIZE_X, SIZE_Y))
original_mask = Image.fromarray(original_mask)
original_mask = original_mask.crop((0 ,0, SIZE_X, SIZE_Y))


plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Testing Image')
plt.imshow(img)
plt.subplot(222)
plt.title('Testing Label')
plt.imshow(original_mask)
plt.subplot(223)
plt.title('Prediction without smooth blending')
plt.imshow(prediction_without_smooth_blending)
plt.subplot(224)
plt.title('Prediction with smooth blending')
plt.imshow(prediction_with_smooth_blending)
plt.savefig('test_res/U_net_upsampling/U_net_final_test_pred_slide2.png', bbox_inches='tight')
plt.close()