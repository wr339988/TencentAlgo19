import sys
import numpy as np
import pandas as pd
np.random.seed(2019)
from src import Evaluation


def postprocess(model_name):

    if model_name == 'CIN':
        submission_result = 'submission/nn_pred_CIN_test.csv'
    else:
        submission_result = 'submission/lgb_pred_test.csv'

    predict_label = pd.read_csv(submission_result,sep='\t',names=['id','preds'])
    test=pd.read_pickle('data/test_NN.pkl')

    test = pd.merge(test,predict_label, left_index = True, right_index = True)
    test['rank'] = test[['aid', 'bid']].groupby('aid')['bid'].apply(
        lambda row: pd.Series(dict(zip(row.index, row.rank())))) - 1
    test['preds'] = test['preds'].apply(round)
    test['preds'] = test['preds'].apply(lambda x: 0 if x < 0 else x)
    test['preds'] = test['preds'] + test['rank'] * 0.0001

    test['preds'] = test['preds'].apply(lambda x: round(x, 4))

    predict_label = test[['aid','preds']].rename(columns={'aid': 'id'})

    predict_label.to_csv(submission_result.replace('.csv', '_postprocessed.csv'),sep='\t',index=False,header=False)

    score = Evaluation.calculate_score(predict_label,"data/testdata/test_df_label.csv","data/testdata/test_df.csv","results/{}Score_Post.txt".format(model_name))
    print(("#Your score is %.4f")%(score*100.0))


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Please provide model name [CIN, LGB] default is CIN.")
        model_name = 'CIN'
        postprocess(model_name)
    else:
        postprocess(sys.argv[1])