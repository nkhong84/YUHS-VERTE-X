# YUHS-VERT-X
We propose a deep learning method for detecting prevalent vertebral factrures and osteoporosis on X-ray images: given an X-ray input image and clinical variables(Age, Sex, BMI), our model provides the risk scores of diseases. 

For more detail, please check our [**Paper**](#)!

## Image processing
Because of the intensity difference in individual images, histogram equalization was applied to all images, and Minmax scaling was selected as a normalization method to indicate the relative degree of intensity in the images. 

Considering that the X-ray image, 5% of the image size was cropped for areas not related to analysis, and resized according to the size (1024, 512). Since the width and height sizes may differ for each image, if the width was larger than the height, set the width to 512px, calculate the height according to the resolution ratio, and resize it. If the image was smaller than (1024,512), align the image to the center position and give Zero-padding to the rest of the area. These processed images were used as input for the model.

## Deep learning processing
Our deep learning models predicted the pVF-score and osteoVF-score, the risk scores of prevalent vertebral fracture and osteoporosis, by receiving spine X-ray images, respectively. It was built based on the Efficientnet-b4 model using Adam optimizer with the initial learning rate of 1e-4. To the prevent over-fitting, weight decay was applied to the models. Due to the size of spine X-ray images and memory limitations, the batch size was set to 30. The experiments were up to 100 epochs on 4 NVIDIA Tesla V100 Graphic processor units (GPU). 
