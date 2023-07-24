# Satellite-Image-Inpainting-based-on-EdgeLearning

A deep learning method suitable for inpainting satellite images, which is characterized by first extracting the landform texture information that is more important for satellite images, and then performing the inpainting learning.





### 1.Theory Introduction

For the inpainting of satellite images, more attention should be paid to the restoration of landform textures such as rivers, hills, and roads. Learning the edge information of landforms first, complementing the edges of the lost parts, and then inpainting the entire satellite image will make the restoration of landform information more accurate and real. The theoretical part of this method is referred from [EdgeConnect Generative Image Inpainting with Adversarial Edge Learning](https://arxiv.org/abs/1901.00212) which suggested that image inpainting is divided into two parts, the first part is the edge learning network, and the second part is the image inpainting network. For details, please refer to this article, and our method has made practical improvements to the algorithm and training process of the edge learning network, mainly including the following:

### 2.Our Improvement in Edge Learning Loss Function

Let $\boldsymbol{edge_-raw}$ be the edge map from the Ground truth Image, this method uses [Canny edge detector](https://ieeexplore.ieee.org/document/4767851), and image mask  $M$ as a pre-condition (1 for the missing region, 0 for background), then the $\boldsymbol{edgemiss}$ denotes the corrupted edge map from the Ground truth Image.
$$
\boldsymbol{edgemiss} =1-\boldsymbol{edgeraw}\odot(1-M)
$$

Specifically,  Let $G_1$ and $D_1$ be the generator and discriminator for the edge generator network which referred in [EdgeConnect](https://arxiv.org/abs/1901.00212), the generator predicts the edge map for the masked region can be calculated, and different from [EdgeConnect](https://arxiv.org/abs/1901.00212), our method remove the grayscale in the $G_1$.
$$
\boldsymbol{edgegen}=G_1(M,\boldsymbol{edgemiss})
$$

![](/Pic/1.png)



### 3.Dataset

We use the Landsat 7 satellite images are downloaded using the [Google Earth Engine (GEE)](https://doi.org/10.1016/j.rse.2017.06.031) and focus solely on the visible RGB bands of the Landsat 7 satellite (B3, B2,B1) with spatial resolutions of 30 meters. 60,000 images of various landforms were downloaded, each with a resolution of 256 × 256, corresponding to a land area of approximately 59 $km^2$. The masks are downloaded from [QD-IMD: Quick Draw Irregular Mask Dataset](https://github.com/karfly/qd-imd), this dataset is a manually drawn irregular mask that can simulate various image losses very well, For satellite images and mask datasets, 50,000 of these were for the training set and 10,000 for the test set.



### 4.Result

<img src="E:\Learning\Satellite_Image_Inpainting\MyProject\01.jpg" alt="image-20230723183210934" style="zoom:120%;" />



<img src="E:\Learning\Satellite_Image_Inpainting\MyProject\02.jpg" alt="image-20230723183232374" style="zoom:120%;" />

























