import dill
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
import os

import argparse
import logging
from scipy.stats import linregress
import pandas as pd

import sys
sys.path.append("..")
sys.path.append("../..")
from utils.util import multi_label_metric, create_log_id, logging_config


# Training settings
def get_args():
    # 读取训好的模型
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default='', help="User notes")
    parser.add_argument('--model_name', type=str, default='LR', help="model name")
    parser.add_argument('--dataset', type=str, default='mimic-iii', help='dataset')

    args = parser.parse_args()
    return args

def create_dataset(data, diag_voc, pro_voc, med_voc):
    i1_len = len(diag_voc.idx2word)
    i2_len = len(pro_voc.idx2word)
    output_len = len(med_voc.idx2word)
    input_len = i1_len + i2_len
    X = []
    y = []
    for patient in data:
        for visit in patient:
            i1 = visit[0]
            i2 = visit[1]
            o = visit[2]

            multi_hot_input = np.zeros(input_len)
            multi_hot_input[i1] = 1
            multi_hot_input[np.array(i2) + i1_len] = 1

            multi_hot_output = np.zeros(output_len)
            multi_hot_output[o] = 1

            X.append(multi_hot_input)
            y.append(multi_hot_output)

    return np.array(X), np.array(y)

def get_visit_weights(data):
    visit_weights = []
    for patient in data:
        for visit in patient:
            visit_weights.append(visit[3])
    return visit_weights

def get_grouped_metrics(test_y, y_pred, y_prob, data_test):
    visit_weights = get_visit_weights(data_test)
    ja = []
    for visit_id in range(len(visit_weights)):
        # 将一个一维的array扩展到二维 : arr 
        test_y_adm = test_y[visit_id].reshape(1, -1)
        y_pred_adm = y_pred[visit_id].reshape(1, -1)
        y_prob_adm = y_prob[visit_id].reshape(1, -1)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            test_y_adm, y_pred_adm, y_prob_adm)
        ja.append(adm_ja)
    weighted_jaccard = np.average(ja, weights=visit_weights)

    # create a dataframe with visit_weights and jaccard
    visit_weights_df = pd.DataFrame({'visit_weights': visit_weights, 'jaccard': ja})
    visit_weights_df.sort_values(by='visit_weights', inplace=True)
    visit_weights_df.reset_index(drop=True, inplace=True)

    sorted_jaccard = visit_weights_df['jaccard'].values

    K=int(len(sorted_jaccard)/5)+1
    grouped_mean_jac = [sorted_jaccard[i:i+K].mean() for i in range(0,int(len(sorted_jaccard)),K)]
    grouped_mean_jac = [round(i, 4) for i in grouped_mean_jac]
    # calculate the correlation between grouped_mean_jac and x
    corr = -np.corrcoef(grouped_mean_jac, np.arange(len(grouped_mean_jac)))[0, 1]
    slope_corr = -linregress(np.arange(len(grouped_mean_jac)), grouped_mean_jac)[0]

    logging.info(f'Weighted Jaccard: {weighted_jaccard:.4}, corr: {corr:.4}, slope_corr: {slope_corr:.4}')
    logging.info(f'grouped_mean_jac: {grouped_mean_jac}')
def bootstrap_test(
    classifier,
    data_test, diag_voc, pro_voc, med_voc,
    ddi_A, ddi_B,
    rounds=10, ratio=0.8, seed=0
):
    """
    对 test 集做 bootstrap：每轮抽取 ratio*N 个病人（有放回），共 rounds 次。
    统计: ja / ddi_rate / ddi_rate_positive / prauc / avg_f1 / avg_med 的 mean ± std。
    """
    import time
    rng = np.random.default_rng(seed)
    N = len(data_test)
    k = int(round(N * ratio))
    rows = []

    print('--------------------Begin Testing (bootstrap)--------------------')
    tic = time.time()
    for _ in range(rounds):
        idx = rng.choice(N, size=k, replace=True).tolist()
        subset = [data_test[i] for i in idx]


        X_sub, y_sub = create_dataset(subset, diag_voc, pro_voc, med_voc)

        # 复用已训练 classifier（不重新训练）
        y_pred = classifier.predict(X_sub)
        y_prob = classifier.predict_proba(X_sub)

        # 基本多标签指标
        ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_sub, y_pred, y_prob)

        # DDI 指标 + 平均用药数
        all_cnt = dd_cnt = dd_pos_cnt = med_cnt = visit_cnt = 0
        for adm in y_pred:
            med_code_set = np.where(adm == 1)[0]
            visit_cnt += 1
            med_cnt += len(med_code_set)
            L = len(med_code_set)
            for i in range(L):
                mi = med_code_set[i]
                for j in range(i + 1, L):
                    mj = med_code_set[j]
                    all_cnt += 1
                    if ddi_A[mi, mj] == 1 or ddi_A[mj, mi] == 1:
                        dd_cnt += 1
                    if ddi_B[mi, mj] == 1 or ddi_B[mj, mi] == 1:
                        dd_pos_cnt += 1

        ddi_rate = (dd_cnt / all_cnt) if all_cnt else 0.0
        ddi_rate_pos = (dd_pos_cnt / all_cnt) if all_cnt else 0.0
        avg_med = (med_cnt / visit_cnt) if visit_cnt else 0.0

        rows.append([ja, ddi_rate, ddi_rate_pos, prauc, avg_f1, avg_med])

    arr = np.array(rows, dtype=np.float64)
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    names = ['ja', 'ddi_rate', 'ddi_rate_positive', 'prauc', 'avg_f1', 'avg_med']

    out = '\n'.join(f'{n}:\t{m:.4f} ± {s:.4f}' for n, (m, s) in zip(names, zip(mean, std)))
    print(out); logging.info(out)
    print('avg test time/round:', (time.time() - tic) / rounds)


def main():
    # set logger
    log_directory_path = os.path.join('../log', args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log'+str(log_save_id)+'_'+args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    grid_search = False
    data_path = f'../../data/output/{args.dataset}' + '/records_final.pkl'
    voc_path = f'../../data/output/{args.dataset}' + '/voc_final.pkl'
    ddi_adj_path = f'../../data/output/{args.dataset}' + '/ddi_A_final.pkl'
    ddi_adj_path_positive = f'../../data/output/{args.dataset}' + '/ddi_B_final.pkl'
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point:split_point + eval_len]
    data_test = data[split_point+eval_len:]

    train_X, train_y = create_dataset(data_train, diag_voc, pro_voc, med_voc)
    test_X, test_y = create_dataset(data_test, diag_voc, pro_voc, med_voc)
    eval_X, eval_y = create_dataset(data_eval, diag_voc, pro_voc, med_voc)

    if grid_search:
        params = {
            'estimator__penalty': ['l2'],
            'estimator__C': np.linspace(0.00002, 1, 100)
        }

        model = LogisticRegression()
        classifier = OneVsRestClassifier(model)
        lr_gs = GridSearchCV(classifier, params, verbose=1).fit(train_X, train_y)

        print("Best Params", lr_gs.best_params_)
        print("Best Score", lr_gs.best_score_)

        return


    # sample_X, sample_y = create_dataset(sample_data, diag_voc, pro_voc, med_voc)

    model = LogisticRegression(C=0.90909)
    classifier = OneVsRestClassifier(model)
    classifier.fit(train_X, train_y)

    y_pred = classifier.predict(test_X)
    y_prob = classifier.predict_proba(test_X)

    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(test_y, y_pred, y_prob)

    get_grouped_metrics(test_y, y_pred, y_prob, data_test)

    # ddi rate
    ddi_A = dill.load(open(ddi_adj_path, 'rb'))
    ddi_B = dill.load(open(ddi_adj_path_positive, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    dd_pos_cnt=0
    med_cnt = 0
    visit_cnt = 0
    for adm in y_pred:
        med_code_set = np.where(adm==1)[0]
        visit_cnt += 1
        med_cnt += len(med_code_set)
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
                if ddi_B[med_i, med_j] ==1 or ddi_B[med_j,med_i] ==1:
                    dd_pos_cnt +=1
    ddi_rate = dd_cnt / all_cnt
    ddi_rate_positive = dd_pos_cnt/all_cnt
    logging.info(f'DDI Rate: {ddi_rate:.4f},DDI Rate Positive: {ddi_rate_positive:.4f},  Jaccard: {ja:.4f}, PRAUC: {prauc:.4f}, AVG_PRC: {avg_p:.4f}, AVG_RECALL: {avg_r:.4f}, AVG_F1: {avg_f1:.4f}, avg med: {med_cnt / visit_cnt:.4f}')

    dill.dump(classifier, open(os.path.join(save_dir, 'model.pkl'), 'wb'))
    ddi_A = dill.load(open(ddi_adj_path, 'rb'))
    ddi_B = dill.load(open(ddi_adj_path_positive, 'rb'))

    bootstrap_test(
        classifier,
        data_test, diag_voc, pro_voc, med_voc,
        ddi_A, ddi_B,
        rounds=10, ratio=0.8, seed=0
    )


if __name__ == '__main__':
    sys.path.append("..")
    np.random.seed(1203)

    args = get_args()
    main()