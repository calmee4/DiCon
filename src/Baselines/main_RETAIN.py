import argparse
from copy import deepcopy
import torch
import numpy as np
import dill
import logging
from torch.optim import Adam
import os
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

import sys
sys.path.append("..")
sys.path.append("../..")
from models.RETAIN import Retain
from utils.util import multi_label_metric, ddi_rate_score, get_n_params, create_log_id, logging_config, get_grouped_metrics, pop_metric, get_model_path, get_pretrained_model_path, sequence_output_process,ddi_rate_score_positive


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default='', help="User notes")
    parser.add_argument('--model_name', type=str, default='Retain', help="model name")
    parser.add_argument('--dataset', type=str, default='mimic-iii', help='dataset')
    parser.add_argument('-t', '--test', action='store_true', help="test mode")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default="log0", help='log dir prefix like "log0", for model test')
    parser.add_argument('--cuda', type=int, default=0, help='which cuda')
    parser.add_argument("--resume_path",type=str,default="./saved/mimic-iv/RETAIN/Epoch_41_JA_0.4028_DDI_0.07527.model")
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate') 
    parser.add_argument('--epoch', type=int, default=50, help='number of epoches')          # epoch增大容易过拟合
    args = parser.parse_args()
    return args

def eval(model, data_eval, voc_size, epoch, ddi_adj_path,ddi_adj_path_positive):
    # evaluate
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    visit_weights = []
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    for input in tqdm(data_eval, ncols=80, total=len(data_eval), desc="Evaluation"):
        if len(input) < 2: # visit > 2 !!! must, otherwise ja=nan
            continue



        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        visit_weights_patient = []
        for i in range(1, len(input)):
            y_pred_label_tmp = []
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[input[i][2]] = 1
            y_gt.append(y_gt_tmp)
            visit_weights_patient.append(input[i][3])

            target_output1 = model(input[:i])

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp >= 0.3] = 1
            y_pred_tmp[y_pred_tmp < 0.3] = 0
            y_pred.append(y_pred_tmp)
            for idx, value in enumerate(y_pred_tmp):
                if value == 1:
                    y_pred_label_tmp.append(idx)
            y_pred_label.append(y_pred_label_tmp)
            med_cnt += len(y_pred_label_tmp)
            visit_cnt += 1

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred),
                                                                                   np.array(y_pred_prob))
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        visit_weights.append(np.mean(visit_weights_patient))
    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, ddi_adj_path)
    ddi_rate_positive = ddi_rate_score_positive(smm_record, ddi_adj_path_positive)
    get_grouped_metrics(ja, visit_weights)

    logging.info(f'''Epoch {epoch:03d}, Jaccard: {np.mean(ja):.4}, DDI Rate: {ddi_rate:.4}, DDI Rate Positive : {ddi_rate_positive:.4},PRAUC: {np.mean(prauc):.4}, AVG_F1: {np.mean(avg_f1):.4}, AVG_MED: {med_cnt / visit_cnt:.4}''')
    return (
    ddi_rate,                      # DDI Rate
    ddi_rate_positive,             # Positive DDI Rate
    float(np.mean(ja)),            # Jaccard
    float(np.mean(prauc)),         # PRAUC
    float(np.mean(avg_f1)),        # AVG_F1
    float(med_cnt / visit_cnt),    # AVG_MED
)

@torch.no_grad()
def test_bootstrap(
    args, model, data_test, voc_size,
    *, ddi_adj_path, ddi_adj_path_positive,
    rounds=10, ratio=0.8, seed=0
):
    """
    对 test 集做 bootstrap：每轮抽取 ratio*N 个病人（有放回），共 rounds 次。
    汇总: ja / ddi_rate / ddi_rate_positive / prauc / avg_f1 / avg_med 的 mean ± std。
    """
    import numpy as np, time

    model = model.eval()
    print('--------------------Begin Testing (bootstrap)--------------------')

    rng = np.random.default_rng(seed)
    N = len(data_test)
    k = int(round(N * ratio))

    rows = []
    tic = time.time()
    for _ in range(rounds):
        idx = rng.choice(N, size=k, replace=True).tolist()
        subset = [data_test[i] for i in idx]

        ddi_rate, ddi_rate_pos, ja, prauc, avg_f1, avg_med = eval(
            model, subset, voc_size, epoch=0,
            ddi_adj_path=ddi_adj_path,
            ddi_adj_path_positive=ddi_adj_path_positive
        )
        rows.append([ja, ddi_rate, ddi_rate_pos, prauc, avg_f1, avg_med])

    arr = np.array(rows, dtype=np.float64)
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    names = ['ja', 'ddi_rate', 'ddi_rate_positive', 'prauc', 'avg_f1', 'avg_med']

    out_lines = [f'{n}:\t{m:.4f} ± {s:.4f}' for n, (m, s) in zip(names, zip(mean, std))]
    out = '\n'.join(out_lines)
    print(out); logging.info(out)
    print('avg test time/round:', (time.time() - tic) / rounds)
    print('parameters', get_n_params(model))


def main(args):
    # set logger
    if args.test:
        args.note = 'test of ' + args.log_dir_prefix
    log_directory_path = os.path.join('../log', args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log'+str(log_save_id)+'_'+args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)
    # load data
    data_path = f'../../data/output/{args.dataset}' + '/records_final.pkl'
    voc_path = f'../../data/output/{args.dataset}' + '/voc_final.pkl'
    ddi_adj_path = f'../../data/output/{args.dataset}' + '/ddi_A_final.pkl'
    ddi_adj_path_positive = f'../../data/output/{args.dataset}' + '/ddi_B_final.pkl'
    device = torch.device('cuda:{}'.format(args.cuda))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point:split_point + eval_len]
    data_test = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = Retain(voc_size, device=device)
    if args.test:
        model_path = args.resume_path
        model.load_state_dict(torch.load(open(model_path, 'rb')))
        model.to(device=device)
        logging.info("load model from %s", model_path)
        eval(model, data_test, voc_size, 0, ddi_adj_path,ddi_adj_path_positive)
        test_bootstrap(
            args, model, data_test, voc_size,
            ddi_adj_path=ddi_adj_path,
            ddi_adj_path_positive=ddi_adj_path_positive,
            rounds=10, ratio=0.8, seed=0
        )

        return

    model.to(device=device)
    print('parameters', get_n_params(model))
    logging.info(f'n_parameters:, {get_n_params(model)}')
    optimizer = Adam(model.parameters(), lr=args.lr)
    logging.info(f'Optimizer: {optimizer}')

    best_epoch, best_ja = 0, 0
    for epoch in range(args.epoch):
        epoch += 1
        model.train()
        for input in tqdm(data_train, ncols=80, total=len(data_train), desc="Training"):
            # loss = 0
            if len(input) < 2: # visit > 2 !!! must, otherwise ja=nan
                continue
            for i in range(1, len(input)):
                target = np.zeros((1, voc_size[2]))
                target[:, input[i][2]] = 1
                output_logits = model(input[:i])
                loss = F.binary_cross_entropy_with_logits(output_logits, torch.FloatTensor(target).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        ddi_rate, ddi_rate_positive, ja, prauc,  avg_f1 ,avg_med= eval(model, data_eval, voc_size, epoch, ddi_adj_path,ddi_adj_path_positive)
        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja
            best_model_state = deepcopy(model.state_dict()) 
        logging.info('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))
    logging.info('Train finished')
    torch.save(best_model_state, open(os.path.join(save_dir, \
                'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(best_epoch, best_ja, ddi_rate)), 'wb'))
    model.load_state_dict(best_model_state)
    print("begin testing")
    test_bootstrap(
        args, model, data_test, voc_size,
        ddi_adj_path=ddi_adj_path,
        ddi_adj_path_positive=ddi_adj_path_positive,
        rounds=10, ratio=0.8, seed=0
    )

if __name__ == '__main__':
    sys.path.append("..")
    torch.manual_seed(1203)
    args = get_args()
    main(args)