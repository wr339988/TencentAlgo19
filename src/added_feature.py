import pandas as pd
import numpy as np
import random
import gc

np.random.seed(2019)
random.seed(2019)


def history(train_df, test_df, log, pivot, f):
    # 以pivot为主键，统计最近一次f的值
    print("history", pivot, f)
    nan = log[f].median()
    dic = {}
    for item in log[[pivot, 'request_day', f]].values:
        if (item[0], item[1]) not in dic:
            dic[(item[0], item[1])] = [item[2]]
        else:
            dic[(item[0], item[1])].append(item[2])
    for key in dic:
        dic[key] = np.mean(dic[key])
    # 统计训练集的特征
    items = []
    cont = 0
    day = log['request_day'].min()
    for item in train_df[[pivot, 'request_day']].values:
        flag = False
        for i in range(item[1] - 1, day - 1, -1):
            if (item[0], i) in dic:
                items.append(dic[(item[0], i)])
                flag = True
                cont += 1
                break
        if flag is False:
            items.append(nan)
    train_df['history_' + pivot + '_' + f] = items
    # 统计测试集的特征
    items = []
    cont = 0
    day_min = log['request_day'].min()
    day_max = log['request_day'].max()
    for item in test_df[pivot].values:
        flag = False
        for i in range(day_max, day_min - 1, -1):
            if (item, i) in dic:
                items.append(dic[(item, i)])
                flag = True
                cont += 1
                break
        if flag is False:
            items.append(nan)
    test_df['history_' + pivot + '_' + f] = items

    print(train_df['history_' + pivot + '_' + f].mean())
    print(test_df['history_' + pivot + '_' + f].mean())
    del items
    del dic
    gc.collect()
    return train_df, test_df

def history_wday(train_df, test_df, log, pivot, f, wday):
    # 以pivot为主键，统计最近一次同wday的f的值
    print("history", pivot, f)
    nan = log[f].median()
    dic = {}
    temp_log = log[[pivot, 'request_day', f, 'aid']].drop_duplicates(['aid', 'request_day'], keep='last')
    for item in log[[pivot, 'request_day','wday', f]].values:
        if (item[0], item[1]) not in dic:
            dic[(item[0], item[1], item[2])] = [item[3]]
        else:
            dic[(item[0], item[1], item[2])].append(item[3])
    for key in dic:
        dic[key] = np.mean(dic[key])
    # 统计训练集的特征
    items = []
    cont = 0
    day = log['request_day'].min()
    for item in train_df[[pivot, 'request_day', 'wday']].values:
        flag = False
        for i in range(item[1] - 1, day - 1, -1):
            if (item[0], i, item[2]) in dic:
                items.append(dic[(item[0], i, item[2])])
                flag = True
                cont += 1
                break
        if flag is False:
            items.append(nan)
    train_df['history_wday_' + pivot + '_' + f] = items

    # 统计测试集的特征
    items = []
    cont = 0
    day_min = log['request_day'].min()
    day_max = log['request_day'].max()
    for item in test_df[pivot].values:
        flag = False
        for i in range(day_max, day_min - 1, -1):
            if (item, i, wday) in dic:
                items.append(dic[(item, i,wday)])
                flag = True
                cont += 1
                break
        if flag is False:
            items.append(nan)
    test_df['history_wday_' + pivot + '_' + f] = items

    print(train_df['history_wday_' + pivot + '_' + f].mean())
    print(test_df['history_wday_' + pivot + '_' + f].mean())
    del items
    del dic
    gc.collect()
    return train_df, test_df

if __name__ == "__main__":
    for path1, path2, log_path, flag, wday, day in [
        ('data/train_dev.pkl', 'data/dev.pkl', 'data/user_log_dev.pkl', 'dev', 0, 17973),
        ('data/train.pkl', 'data/test.pkl', 'data/user_log_test.pkl', 'test', 1, 17974)]:
        train_df = pd.read_pickle(path1)
        test_df = pd.read_pickle(path2)
        log = pd.read_pickle(log_path)

        # for pivot in ['advertiser', 'good_id', 'good_type', 'ad_size', 'ad_type_id']:
        for pivot in ['advertiser']:
            for f in ['imp', 'bid', 'pctr', 'quality_ecpm', 'totalEcpm']:
                history(train_df, test_df, train_df, pivot, f)

        for pivot in ['aid']:
            for f in ['imp', 'bid', 'pctr', 'quality_ecpm', 'totalEcpm']:
                history_wday(train_df, test_df, log, pivot, f, wday)

        # for pivot in ['advertiser', 'good_id', 'good_type', 'ad_size', 'ad_type_id']:
        for pivot in ['advertiser']:
            for f in ['imp', 'bid', 'pctr', 'quality_ecpm', 'totalEcpm']:
                history_wday(train_df, test_df, train_df, pivot, f, wday)

        test_df['request_day'] = day
        train_aid = train_df[['aid']].drop_duplicates()
        test_df = pd.merge(test_df, train_aid, on='aid', how='left', indicator=True)
        test_df['isnew'] = test_df['_merge'] == 'left_only'
        del test_df['_merge']

        train_min = train_df.groupby(['aid'])[['request_day']].min().reset_index()
        train_df = pd.merge(train_df, train_min, on='aid')
        train_df['request_day_min'] = train_df['request_day_y']
        train_df['request_day'] = train_df['request_day_x']
        train_df['isnew'] = train_df['request_day'] == train_df['request_day_min']
        del train_df['request_day_y']
        del train_df['request_day_x']
        del train_df['request_day_min']

        # 保存数据
        print(train_df.shape, test_df.shape, log.shape)
        train_df.to_pickle(path1.replace('.pkl', '_added.pkl'))
        test_df.to_pickle(path2.replace('.pkl', '_added.pkl'))
        print(list(train_df))