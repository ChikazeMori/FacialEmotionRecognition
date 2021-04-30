========================================================
Real-world Affective Faces (RAF) Database
========================================================

***** Important *****

The dataset was provided to City, University of London for research purposes only.

The RAF database is available for non-commercial research purposes only.
All images of the RAF database are obtained from the Internet which are not property of PRIS, Beijing University of Posts and Telecommunications. The PRIS is not responsible for the content nor the meaning of these images.
You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
You agree not to further copy, publish or distribute any portion of the RAF database.
The PRIS reserves the right to terminate your access to the RAF database at any time.


For more information about the dataset, visit the project website:

  http://whdeng.cn/RAF/model1.html

If you use the dataset in a publication, please cite the paper below:

@inproceedings{li2017reliable,
  title={Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild},
  author={Li, Shan and Deng, Weihong and Du, JunPing},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  pages={2584--2593},
  year={2017},
  organization={IEEE}
}

Please note that we do not own the copyrights to these images. Their use is RESTRICTED to non-commercial research and educational purposes.


========================
File Information
========================

- Images
    - Training set (train)
       12271 aligned facial images with basic expression. See IMAGE section below for more information.
    - Testing set (test)
       3068 aligned facial images with basic expression. See IMAGE section below for more information.

- Emotion labels
    - Training set (labels/list_label_train.txt)
       Expression label of each image for training set. See EMOTION_LABEL section below for more information.
    - Testing set (labels/list_label_test.txt)
       Expression label of each image for testing set. See EMOTION_LABEL section below for more information.


=========================
IMAGE
=========================

1. Images are named in the format of "train_XXXXX_aligned.jpg" / "test_XXXX_aligned.jpg";
2. Images are first roughly aligned using similarity transformation according to the two eye locations and the center of mouth;
3. Images are then resized to 100*100.


=========================
EMOTION_LABEL
=========================

each row: <image_name> <emotion label>

Label explanation:
1: Surprise
2: Fear
3: Disgust
4: Happiness
5: Sadness
6: Anger
7: Neutral

