import cv2, time, os
import numpy as np
from skimage import io
from skimage.feature import hog
from sklearn.cluster import MiniBatchKMeans

def SIFT(label_list_train,label_list_test):
    os.chdir('..')
    print(os.getcwd())
    start_time = time.time()
    # train_dataset
    descriptors = []
    labels = []
    for i,row in label_list_train.iterrows():
        img_path = 'CW_Dataset/' + 'train' + '/'
        filename = row[0]
        label = row[1]
        img_path += str(filename.split('.')[0])+'_aligned.jpg'
        img = cv2.imread(img_path,0)
        # Create SIFT object. You can specify params here or later.
        sift = cv2.SIFT_create()
        # Find keypoints and descriptors directly
        kp, des = sift.detectAndCompute(img,None)
        if des is not None:
            descriptors.append(des)
            labels.append(label)
    # Convert to array for easier handling
    des_array = np.vstack(descriptors)
    # Number of centroids/codewords: good rule of thumb is 10*num_classes
    k = len(np.unique(labels)) * 10
    # Use MiniBatchKMeans for faster computation and lower memory usage
    batch_size = des_array.shape[0] // 4
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size).fit(des_array)
    # Convert descriptors into histograms of codewords for each image
    hist_list = []
    idx_list = []
    for des in descriptors:
        hist = np.zeros(k)
        idx = kmeans.predict(des)
        idx_list.append(idx)
        for j in idx:
            hist[j] = hist[j] + (1 / len(des))
        hist_list.append(hist)
    hist_array = np.vstack(hist_list)

    # test_dataset
    hist_list_test = []
    labels_test = []
    for i, row in label_list_test.iterrows():
        img_path = 'CW_Dataset/' + 'test' + '/'
        filename = row[0]
        label = row[1]
        labels_test.append(label)
        img_path += str(filename.split('.')[0]) + '_aligned.jpg'
        img = cv2.imread(img_path, 0)
        # Create SIFT object. You can specify params here or later.
        sift = cv2.SIFT_create()
        # Find keypoints and descriptors directly
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            hist = np.zeros(k)
            idx = kmeans.predict(des)
            for j in idx:
                hist[j] = hist[j] + (1 / len(des))
            # hist = scale.transform(hist.reshape(1, -1))
            hist_list_test.append(hist)
        else:
            hist_list_test.append(None)
    # Remove potential cases of images with no descriptors
    idx_not_empty = [i for i, x in enumerate(hist_list_test) if x is not None]
    hist_list_test = [hist_list_test[i] for i in idx_not_empty]
    labels_test = [labels_test[i] for i in idx_not_empty]
    hist_array_test = np.vstack(hist_list_test)
    
    os.chdir('Code')

    return hist_array,labels,hist_array_test,labels_test,time.time() - start_time

def HOG(label_list_train,label_list_test):
    os.chdir('..')
    print(os.getcwd())
    start_time = time.time()
    # train_dataset
    descriptors = []
    labels = []
    for i,row in label_list_train.iterrows():
        img_path = 'CW_Dataset/' + 'train' + '/'
        filename = row[0]
        label = row[1]
        img_path += str(filename.split('.')[0])+'_aligned.jpg'
        img = io.imread(img_path)
        # Create HOG object. You can specify params here or later.
        des, HOG_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                                 cells_per_block=(1, 1), visualize=True, multichannel=True)
        descriptors.append(des)
        labels.append(label)
    # Convert to array for easier handling
    des_array = np.vstack(descriptors)

    # test_dataset
    des_list_test = []
    labels_test = []
    for i, row in label_list_test.iterrows():
        img_path = 'CW_Dataset/' + 'test' + '/'
        filename = row[0]
        label = row[1]
        img_path += str(filename.split('.')[0]) + '_aligned.jpg'
        img = io.imread(img_path)
        # Create HOG object. You can specify params here or later.
        des, HOG_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                             cells_per_block=(1, 1), visualize=True, multichannel=True)
        des_list_test.append(des)
        labels_test.append(label)
    des_array_test = np.vstack(des_list_test)
    
    os.chdie('Code')

    return des_array,labels,des_array_test,labels_test,time.time() - start_time

def ORB(label_list_train,label_list_test):
    os.chdir('..')
    print(os.getcwd())
    start_time = time.time()
    # train_dataset
    descriptors = []
    labels = []
    for i,row in label_list_train.iterrows():
        img_path = 'CW_Dataset/' + 'train' + '/'
        filename = row[0]
        label = row[1]
        img_path += str(filename.split('.')[0])+'_aligned.jpg'
        img = cv2.imread(img_path,0)
        # Initiate STAR detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(img, None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        if des is not None:
            descriptors.append(des)
            labels.append(label)
    # Convert to array for easier handling
    des_array = np.vstack(descriptors)
    # Number of centroids/codewords: good rule of thumb is 10*num_classes
    k = len(np.unique(labels)) * 10
    # Use MiniBatchKMeans for faster computation and lower memory usage
    batch_size = des_array.shape[0] // 4
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size).fit(des_array)
    # Convert descriptors into histograms of codewords for each image
    hist_list = []
    idx_list = []
    for des in descriptors:
        hist = np.zeros(k)
        idx = kmeans.predict(des)
        idx_list.append(idx)
        for j in idx:
            hist[j] = hist[j] + (1 / len(des))
        hist_list.append(hist)
    hist_array = np.vstack(hist_list)

    # test_dataset
    hist_list_test = []
    labels_test = []
    for i, row in label_list_test.iterrows():
        img_path = 'CW_Dataset/' + 'test' + '/'
        filename = row[0]
        label = row[1]
        labels_test.append(label)
        img_path += str(filename.split('.')[0]) + '_aligned.jpg'
        img = cv2.imread(img_path, 0)
        # Initiate STAR detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(img, None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        if des is not None:
            hist = np.zeros(k)
            idx = kmeans.predict(des)
            for j in idx:
                hist[j] = hist[j] + (1 / len(des))
            # hist = scale.transform(hist.reshape(1, -1))
            hist_list_test.append(hist)
        else:
            hist_list_test.append(None)
    # Remove potential cases of images with no descriptors
    idx_not_empty = [i for i, x in enumerate(hist_list_test) if x is not None]
    hist_list_test = [hist_list_test[i] for i in idx_not_empty]
    labels_test = [labels_test[i] for i in idx_not_empty]
    hist_array_test = np.vstack(hist_list_test)
    
    os.chdir('Code')

    return hist_array,labels,hist_array_test,labels_test,time.time() - start_time