import lightgbm as lgb #2.3.1
import numpy as np
import pandas as pd
import os
import gc
import time
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import random
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
np.random.seed(2019)
from src import Evaluation

# 单值特征，直接embedding:支持任何可以转换成字符串的数据类型，比如浮点数，会转换成字符串
single_features = ['periods_cont', 'aid', 'advertiser',
                   'good_id', 'good_type', 'ad_type_id', 'good_id_advertiser_count', 'good_id_aid_count', 'good_id_ad_size_count',
                   'good_id_ad_type_id_count', 'good_id_good_id_size', 'advertiser_good_id_count',
                   'advertiser_aid_count', 'advertiser_ad_size_count', 'advertiser_ad_type_id_count',
                   'advertiser_good_type_count','wday', 'request_day']

# 交叉特征，会使用分解机提取特征:支持任何可以转换成字符串的数据类型。比如浮点数，会转换成字符串
cross_features = ['aid', 'crowd_direction', 'delivery_periods', 'advertiser', 'good_id', 'good_type',
                  'ad_type_id', 'consuptionAbility', 'os', 'work', 'connectionType', 'ad_size']

# 多值特征，会使用分解机提取特征:支持字符串数据类型，用空格隔开
multi_features = ['aid_uids', 'age', 'area', 'status', 'behavior', 'good_id_advertisers',
                  'good_id_request_days', 'good_id_positions', 'good_id_period_ids', 'good_id_wdays',
                  'advertiser_good_ids', 'advertiser_request_days', 'advertiser_positions',
                  'advertiser_period_ids', 'advertiser_wdays']

# 稠密特征，直接放入MLP中:主要用于embedding特征，转化率等
dense_features = ['uid_w2v_embedding_aid_64_' + str(i) for i in range(64)] + [
    'uid_w2v_embedding_good_id_64_' + str(i) for i in range(64)] + ['uid_w2v_embedding_advertiser_64_' + str(i)
                                                                    for i in range(64)]
dense_features += ['uid_aid_aid_deepwalk_embedding_64_' + str(i) for i in range(64)] + [
    'uid_good_id_good_id_deepwalk_embedding_64_' + str(i) for i in range(64)]
dense_features += ['periods_on_' + str(i) for i in range(48)]

# key-values 特征，将稠密特征转换成向量: 浮点数类型，且数值在[0,1]之间
kv_features = ['history_aid_imp', 'history_aid_bid', 'history_aid_pctr', 'history_aid_quality_ecpm',
               'history_aid_totalEcpm', 'good_id_advertiser_count', 'good_id_aid_count',
               'good_id_ad_size_count', 'good_id_ad_type_id_count', 'good_id_good_id_size',
               'advertiser_good_id_count', 'advertiser_aid_count', 'advertiser_ad_size_count',
               'advertiser_ad_type_id_count', 'advertiser_good_type_count', 'advertiser_advertiser_size',
               'good_id_imp_median', 'good_id_imp_std', 'good_id_imp_min', 'good_id_imp_max',
               'advertiser_imp_mean', 'advertiser_imp_median', 'advertiser_imp_std', 'advertiser_imp_min',
               'advertiser_imp_max', 'create_timestamp']


periods_features = ['periods_on_' + str(i) for i in range(48)]

added_features = [
    'history_advertiser_imp','history_advertiser_bid','history_advertiser_pctr','history_advertiser_quality_ecpm','history_advertiser_totalEcpm',
    'history_good_id_imp','history_good_id_bid','history_good_id_pctr','history_good_id_quality_ecpm','history_good_id_totalEcpm',
    # 'history_good_type_imp','history_good_type_bid','history_good_type_pctr','history_good_type_quality_ecpm','history_good_type_totalEcpm',
    # 'history_ad_size_imp','history_ad_size_bid','history_ad_size_pctr','history_ad_size_quality_ecpm','history_ad_size_totalEcpm',
    # 'history_ad_type_id_imp','history_ad_type_id_bid','history_ad_type_id_pctr','history_ad_type_id_quality_ecpm','history_ad_type_id_totalEcpm',
    'history_wday_aid_imp','history_wday_aid_bid','history_wday_aid_pctr','history_wday_aid_quality_ecpm','history_wday_aid_totalEcpm',
    'history_wday_advertiser_imp','history_wday_advertiser_bid','history_wday_advertiser_pctr','history_wday_advertiser_quality_ecpm','history_wday_advertiser_totalEcpm',
    'history_wday_good_id_imp','history_wday_good_id_bid','history_wday_good_id_pctr','history_wday_good_id_quality_ecpm','history_wday_good_id_totalEcpm',
    # 'history_wday_good_type_imp','history_wday_good_type_bid','history_wday_good_type_pctr','history_wday_good_type_quality_ecpm','history_wday_good_type_totalEcpm',
    # 'history_wday_ad_size_imp','history_wday_ad_size_bid','history_wday_ad_size_pctr','history_wday_ad_size_quality_ecpm','history_wday_ad_size_totalEcpm',
    # 'history_wday_ad_type_id_imp','history_wday_ad_type_id_bid','history_wday_ad_type_id_pctr','history_wday_ad_type_id_quality_ecpm','history_wday_ad_type_id_totalEcpm'
    ]


####################################################################################

features = single_features + kv_features + multi_features + cross_features + periods_features + added_features
enc_features = multi_features + cross_features
cv_features=[]

#setting
lgb_model = lgb.LGBMRegressor(
    num_leaves=256, reg_alpha=0., reg_lambda=0.01, objective='mae', metric=False,
    max_depth=-1, learning_rate=0.03,min_child_samples=25,
    n_estimators=1000, subsample=0.7, colsample_bytree=0.45
)
def eval_f(y_true,y_pred):
    global best_score
    global test_cont
    y_pred=np.maximum(np.exp(y_pred*test_cont)-1,1)
    y_true=np.maximum(y_true,1)
    SMAPE=abs(y_true-y_pred)/((abs(y_true)+abs(y_pred))/2+1e-15)
    SMAPE=np.mean(SMAPE)
    if SMAPE<best_score:
        best_score=SMAPE
        return "score",SMAPE,False
    else:
        return "score",SMAPE,True


       
#load data
timestr = time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
print(timestr)

test=pd.read_pickle('data/test_added_NN.pkl')
dev=pd.read_pickle('data/dev_added_NN.pkl')
train=pd.read_pickle('data/train_added_NN_0.pkl')
train_dev=pd.read_pickle('data/train_dev_added_NN_0.pkl')

test['request_day'] = 17974
dev['request_day'] = 17973

dev=dev[dev['gold']==True]
for df in [test,dev,train,train_dev]:
    df['cont']=1
train['imp']=train[['imp','cont']].apply(lambda x:np.log(x[0]+1)/x[1],axis=1)
train_dev['imp']=train_dev[['imp','cont']].apply(lambda x:np.log(x[0]+1)/x[1],axis=1)

lb=LabelEncoder()
for train_df,test_df in [(train_dev,dev),(train,test)]:
    for f in enc_features:
        data=train_df[[f]].append(test_df[[f]])
        data=data.astype(str)
        lb.fit(data[f])
        train_df[f]=lb.transform(train_df[f].astype(str))
        test_df[f]=lb.transform(test_df[f].astype(str))

#dev training
train_x=train_dev[features]
test_x=dev[features]
train_x=train_x.astype(float)
test_x=test_x.astype(float)

print(train_x.shape,test_x.shape)
test_cont=dev['cont']
cv=CountVectorizer()
for feature in cv_features:
    data=train_dev[[feature]].append(dev[[feature]])
    cv.fit(data[feature])
    train_a = cv.transform(train_dev[feature])
    test_a = cv.transform(dev[feature])
    train_x = sparse.hstack((train_x, train_a)).tocsr()
    test_x = sparse.hstack((test_x, test_a)).tocsr()
    print(feature)


print(len(features))
best_score=9999
model=lgb_model.fit(train_x,train_dev['imp'], eval_set=[(test_x,dev['imp'])],
                    early_stopping_rounds=10000,
                    eval_metric=eval_f,verbose=100)

lgb_predictors = [i for i in train_dev[features].columns]
print(len(lgb_model.feature_importances_))
lgb_feat_imp = pd.Series(lgb_model.feature_importances_, lgb_predictors).sort_values(ascending=False)
lgb_feat_imp.to_csv('data/features/lgb_feat_imp_dev_{}.csv'.format(timestr))


dev_preds=np.zeros(len(dev))
dev_preds+=(np.exp(model.predict(test_x))-1)*test_cont
print(dev_preds.mean())
print("*"*80)




#test training
train_x=train[features]
test_x=test[features]
train_x=train_x.astype(float)
test_x=test_x.astype(float)



print(train_x.shape,test_x.shape)
test_preds=np.zeros(len(test))

for i in range(5):
    print("Fold",i)
    seed=random.randint(0,100000)
    print("Seed",seed)
    lgb_model = lgb.LGBMRegressor(
    num_leaves=256, reg_alpha=0., reg_lambda=0.01, objective='mae',
    max_depth=-1, learning_rate=0.03,min_child_samples=25,
    n_estimators=1000, subsample=0.7, colsample_bytree=0.45,random_state=seed)
    model=lgb_model.fit(train_x, train['imp'])

    test_preds+=model.predict(test_x)/5
    print(np.mean(np.exp(test_preds*5/(i+1)*test['cont'])-1))

    lgb_predictors = [i for i in train[features].columns]
    print(len(lgb_model.feature_importances_))
    lgb_feat_imp = pd.Series(lgb_model.feature_importances_, lgb_predictors).sort_values(ascending=False)
    lgb_feat_imp.to_csv('data/features/lgb_feat_imp_test_{}_{}.csv'.format(i, timestr))

test_preds=np.exp(test_preds*test['cont'])-1
print(dev_preds.mean())
print(test_preds.mean())
dev['lgb_preds'] = dev_preds
dev_fea=dev[['aid','bid','gold','imp','lgb_preds']]
test['lgb_preds'] = test_preds
test_fea=test[['aid','lgb_preds']]
test_fea.to_csv('stacking/lgb_pred_{}_test.csv'.format('lgb'),index=False)
dev_fea.to_csv('stacking/lgb_pred_{}_dev.csv'.format('lgb'),index=False)


#result
test['preds']=pd.read_csv('stacking/lgb_pred_{}_test.csv'.format('lgb'))['lgb_preds']
test['preds']=test['preds'].apply(lambda x: 0 if x<0  else x)
print(test['preds'].mean())

test[['id','preds']].to_csv('submission/lgb_pred_test.csv', sep='\t',index=False,header=False)

# Calculate score
score = Evaluation.calculate_score(test,"data/testdata/test_df_label.csv","data/testdata/test_df.csv","results/LGBScore.txt")
print(("#Your score is %.4f")%(score*100.0))
