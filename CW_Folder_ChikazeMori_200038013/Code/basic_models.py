def train_test():
    import create_datasets as cd
    import feature_descriptors as fd
    import time, pickle, os
    from sklearn import svm,metrics
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV
    # datasets
    print(os.getcwd())
    train_label_list = cd.create_datasets(type='train')
    test_label_list = cd.create_datasets(type='test')
    
    # ORB
    ORB_X_train,ORB_y_train,ORB_X_test,ORB_y_test,ORB_time = fd.ORB(train_label_list,test_label_list)
    print('ORB speed: ',ORB_time)
    # SIFT
    SIFT_X_train,SIFT_y_train,SIFT_X_test,SIFT_y_test,SIFT_time = fd.SIFT(train_label_list,test_label_list)
    print('SIFT speed: ',SIFT_time)
    # HOG
    HOG_X_train,HOG_y_train,HOG_X_test,HOG_y_test,HOG_time = fd.HOG(train_label_list,test_label_list)
    print('HOG speed: ',HOG_time)
    # save the datasets
    ORB_data = {'X_train':ORB_X_train,'y_train':ORB_y_train,'X_test':ORB_X_test,'y_test':ORB_y_test,'speed':ORB_time}
    pickle.dump(ORB_data, open("ORB_data.p", "wb"))
    SIFT_data = {'X_train':SIFT_X_train,'y_train':SIFT_y_train,'X_test':SIFT_X_test,'y_test':SIFT_y_test,'speed':SIFT_time}
    pickle.dump(SIFT_data, open("SIFT_data.p", "wb"))
    HOG_data = {'X_train':HOG_X_train,'y_train':HOG_y_train,'X_test':HOG_X_test,'y_test':HOG_y_test,'speed':HOG_time}
    pickle.dump(HOG_data, open("HOG_data.p", "wb"))
    
    # hyperparameters
    SVM_parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    KNN_parameters = {'weights':('uniform', 'distance'), 'leaf_size':[15, 30]}
    MLP_parameters = {'alpha':[0.0001,0.001], 'learning_rate_init':[0.0001,0.001]}
    
    # train and test all the combinations of the feature descriptors and the classifiers
    for fd in ['ORB','SIFT','HOG']:
          if fd == 'ORB':
                X_train,y_train,X_test,y_test = ORB_X_train,ORB_y_train,ORB_X_test,ORB_y_test
          elif fd == 'SIFT':
                X_train,y_train,X_test,y_test = SIFT_X_train,SIFT_y_train,SIFT_X_test,SIFT_y_test
          else:
                X_train, y_train, X_test, y_test = HOG_X_train, HOG_y_train, HOG_X_test, HOG_y_test
          for c in ['SVM','KNN','MLP']:
                # train
                start_time = time.time()
                if c == 'SVM':
                      model = svm.SVC()
                      model = GridSearchCV(model, SVM_parameters)
                elif c == 'KNN':
                      model = KNeighborsClassifier(n_neighbors=7)
                      model = GridSearchCV(model, KNN_parameters)
                else:
                      model = MLPClassifier(max_iter=200)
                      model = GridSearchCV(model, MLP_parameters)
    
                model.fit(X_train,y_train)
                speed = time.time() - start_time
                print(fd+'_'+c+': ',speed)
                print(fd+'_'+c+' CV: ',model.cv_results_)
                # test
                predicted = model.predict(X_test).tolist()
                # result
                print(f"""Classification report for  {fd+'_'+c}:
                      {metrics.classification_report(y_test,predicted, zero_division=0)}\n""")
                # save the model and the results
                results = {'model':model,'speed':speed,'cv_result':model.cv_results_,'predicted':predicted, \
                                   'result':metrics.classification_report(y_test,predicted,zero_division=0)}
                pickle.dump(results, open(fd+'_'+c+'_result.p','wb'))