# Foreground-Object-Detection-in-Image-Sequence
Detection of Foreground Object using background Subtracton 
Image sequence (people running around in an airstrip) consisting of
RGB images and corresponding ground-truth images of foreground detection. 
following parameters for foreground extraction – v d = 9.0, m = 5, η v = 0.6 and t c = 0.01. 
Sensitivity (s) and False Alarm Rate (ρ) evaluated for λ = {0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}. 
the ROC Curve ploted using these (s, ρ) values.


Download Dataset from below link.
https://drive.google.com/drive/folders/1bU0gj3U8n6ykD58FgJJp5s5CU6BeN0o_?usp=sharing

1. Give path of Dataset

Image_files = glob.glob("Path_to_Image_folder/Image/*.bmp")

Background_files = glob.glob('Path_to_Background_Folder/BakSubGroundTruth/*.bmp')

2. Delete Duplicates from Image Folder

AirstripRunAroundFeb2006_1436(1).bmp, 1437(1), 1438(1), 1439(1) these are duplicates delete it first from Images Folder

3. Check if number of images in each folders Image and Background is 501

************************************************************************************************************************
Now Program can be run
************************************************************************************************************************

* For each lambda value, it takes 13 mins for model training; if 501 Images are used for model training.

* Program takes 105 mins on my system for Image Sequence of all 501 Images

* Program takes 46 mins on my system for Image Sequence of 200 Images (Default Value; Reasonably Sufficient for Training Model )

* Program takes 25 mins on my system for Image Sequence of 100 Images

* To change number of images in Image Sequence can be changed by variable "training_size" (default is 200)

* To change number of images (Randomly Chosen) for testing can be changed by variable "test_sample_size" (default is 50)


************************************************************************************************************************

Ouputed ROC Curve is attached:
ROC_Curve_100.png for Image Sequence of 100 Images
ROC_Curve_200.png for Image Sequence of 200 Images
ROC_Curve_500.png for Image Sequence of 500 Images
