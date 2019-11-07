This Project trains a DNN for semantic segmentation. The label data(ground truth images) have only road pixels annonated.


Required python environment: * Need to update. *Anaconda


Usage of Functions:
---------->>Training a DNN.
           python main.py

Inputs:
	vgg16 model. To be kept in ./data. The model can be downloaded from https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip.
        * The model is not vanilla VGG16, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. 
        * More details here:https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf


	Dataset.(Kitti Road dataset: http://www.cvlibs.net/datasets/kitti/eval_road.php)
	Dataset can be download from here: https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/advanced_deep_learning/data_road.zip
        [Note about the dataset folder structure: Folder structure is ./data/data_road/training/gt_image_2 (ground truth data)
                                                                                               /image_2    (images)
                                                                                      /testing/image_2     (images)]
	[The testing data is not required for training]        
	[It is possible to use any training dataset provided the folder structure remains same]
Output:
      The trained model by the name of 'model'[model.ckpt.meta, model.ckpt.index, model.ckpt.data-00000-of-00001]

It will take a long time to train model. At the end "End" will be printed.


----------->>To apply semantic segmentation on a video. 
           Usage:
           python video_inference.py.
            Input: A raw video of name 'test_video.mp4'.
                   Trained 'model'[model.ckpt.meta, model.ckpt.index, model.ckpt.data-00000-of-00001]
            Output: It will create an output video: 'video_output.mp4'
           



	