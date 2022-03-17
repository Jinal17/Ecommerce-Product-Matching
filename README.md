# Project: Shopee - Ecommerce Product matching

This report details a deep learning approach that predicts if two Ecommerce products are same by their images and text. This can facilitate retailers with management of products and can improve consumer shopping experience. We are working on a Kaggle competition sponsored by Shopee. Shopee is a multinational ecommerce technology company. The goal of this project is to develop a model that will identify which products have been posted repeatedly at an ecommerce site Shopee. It will also address similar grouping of the images that are uploaded on a daily basis to the site by resellers and individual dealers. 

## Dataset
Dataset is provided by Shopee, which includes 34,251 images  in JPEG format, spread out between 11,014 classes. There is a csv file with details about the images that includes :  
	posting_id: the ID code for the posting
	image: the image id/md5sum
	image_phash: a perceptual hash of the image
	title: the product description for the posting
	label_group: It is the target label for all image postings that maps to the similar product. This will be used for training the model. Since this is the target label it is not provided in the test data set. 

![image](https://user-images.githubusercontent.com/29663370/158750294-dd96af1a-95c2-4aed-bdd4-66a1b090cb25.png)

#### Steps to train the nfnet model based on image data
* python3 shopee_training_nfnet.py

#### Steps to train the efficientnet model based on image data
* python3 shopee_training_efficientnet.py

#### Steps to train the resnet model based on image data
* python3 shopee_training_resnet.py

#### Steps to train the xlm-r-multilingual model based on text data
* python3 shopee_text_training.py

#### Steps to train the custom model based on image and text data
* python3 shopee_custom.py

#### Steps for running inference using ensemble model(nfnet,efficientnet,resnet) and xlm-r-multilingual model to create submission file
* python3 shopee_inference.py
