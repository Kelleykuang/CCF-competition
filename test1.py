#%%
import pandas as pd
import catboost as cbt
import gc
import numpy as np
import catboost as cb
from catboost import CatBoostRegressor as cbr
from sklearn.metrics import mean_absolute_error
import seaborn as sns

def trainRegreModel(trainData,trainLabels):
    '''
    Params:
        X_train: the parameters
        Y: the attributes
    Return: regression model
    '''
    model = cbr(iterations=6000,learning_rate=0.05,depth=4)
    model.fit(trainData, trainLabels)
    return model 

def trainClassifiModel(trainData,trainLabels):
    '''
    Params:
        trainData: the train data vector X
        trainLabels: the label of train data
    Usage:
        This function is used to train the classification model
    Return:
        the classification model
    '''
    cbt_model = cbt.CatBoostClassifier(iterations=2000,learning_rate=0.04,verbose=100,early_stopping_rounds=1000,task_type='CPU',loss_function='MultiClass')
    cbt_model.fit(trainData, trainLabels, eval_set=(trainData,trainLabels))
    gc.collect()
    return cbt_model

def predictAttr(testData, model, testAttr):
    predictAttr = model.predict(testData)
    np.save("predictAttrData",predictAttr)
    error = mean_absolute_error(testAttr, predictAttr)
    print("predict attr, error: ", error)

def predictLabel(testData, model, testLabel):
    ''' 
    Params: 
        classifyModel: the classification model 
        testData: the test data
    Return:
        return the predict list 
    '''
    predicLabel = model.predict_proba(testData)
    predicLabel = np.argmax(predicLabel, axis=-1)
    np.save("predictionData",predicLabel)
    gc.collect()
    result = predicLabel - testLabel
    zeroCount = np.count_nonzero(result==0)
    ac = zeroCount / predicLabel.shape[0]
    print("predictLabel ac: ", ac)

def main():
    '''
    Prepare data 
    '''
    train = pd.read_csv('data/first_round_training_data.csv')
    test = pd.read_csv('data/first_round_testing_data.csv')
    data = train.append(test).reset_index(drop=True)

    dit = {'Excellent':0,'Good':1,'Pass':2,'Fail':3}
    data['label'] = data['Quality_label'].map(dit)

    feature_name_attribute = ['Attribute{0}'.format(i) for i in range(1,4)]
    feature_name_parameter = ['Parameter{0}'.format(i) for i in range(1,11)]
    tr_index = ~data['label'].isnull() 
    # print(tr_index[0:5000])
    # print(tr_index[5001:6001])

    X_train = data[0:5000][feature_name_attribute].reset_index(drop=True)
    # print(X_train)
    
    y = data[0:5000]['label'].reset_index(drop=True).astype(int)
    X_test = data[5001:6000][feature_name_attribute].reset_index(drop=True)
    y_test = data[5001:6000]['label'].reset_index(drop=True).astype(int)
    y_test = np.asarray(y_test)

    trainDataParam = np.log1p(np.asarray(data[0:5000][feature_name_parameter].reset_index(drop=True)))
    trainDataAttr = np.log1p(np.asarray(data[0:5000][feature_name_attribute].reset_index(drop=True)))
    print(trainDataAttr)
    testDataParam = np.log1p(np.asarray(data[5001:6000][feature_name_parameter].reset_index(drop=True)))
    testDataAttr = np.log1p(np.asarray(data[5001:6000][feature_name_attribute].reset_index(drop=True)))
    # print(testDataAttr)

    # regreModelAttr0 = trainRegreModel(trainDataParam,trainDataAttr[:,0]).save_model("regreModelAttr0","cbm")

    # regreModelAttr1 = trainRegreModel(trainDataParam,trainDataAttr[:,1]).save_model("regreModelAttr1","cbm")
    regreModelAttr1 = cbr()
    regreModelAttr1.load_model("regreModelAttr1")
    predictAttr(testDataParam,regreModelAttr1, testDataAttr[:,1])

    # regreModelAttr2 = trainRegreModel(trainDataParam,trainDataAttr[:,2]).save_model("regreModelAttr2","cbm")

    # regreModelAttr3 = trainRegreModel(trainDataParam,trainDataAttr[:,3])

    # predictAttr = np.load("predictAttrData.npy")
    # print(predictAttr.shape)
    # error = mean_absolute_error(testDataAttr, predictAttr)
    # print("predict attr, error: ", error)


    # classificationModel = trainClassifiModel(X_train, y)

    # prediction = predict(X_test,classificationModel)
    # np.save("predictionData",prediction)
    # prediction = np.load("predictionData.npy")
    # print(y_test)


if __name__ == "__main__":
    main()
    
#%%
