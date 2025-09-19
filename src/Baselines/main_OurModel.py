import os
import time
import dill
import torch
import logging
import argparse
import torch.optim as optim
import sys
from copy import deepcopy
import os
import math
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss
from torch.optim import AdamW


import dill
import networkx as nx
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
# 导入正常的库
from tqdm import tqdm
sys.path.append("..")
sys.path.append("../..")
import torch.nn.functional as F
from prettytable import PrettyTable
from collections import defaultdict
from rdkit import Chem
from utils.util import llprint, multi_label_metric, ddi_rate_score, set_seed,ddi_rate_score_positive
from utils.util import multi_label_metric, ddi_rate_score, get_n_params, create_log_id, logging_config, get_grouped_metrics, get_model_path, ddi_rate_score_positive
from utils.util import llprint, ddi_rate_score, get_n_params , Regularization,buildMPNN
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, f1_score
from models.gnn import graph_batch_from_smile

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CyclicLR, StepLR
import pandas as pd
from models.OurModel import OurModel
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import random
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.optim.swa_utils import AveragedModel, update_bn
from data_loader import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace
from torch.utils.data.dataloader import DataLoader
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def multi_label_metric(y_gt, y_pred, y_prob):
    def false_positives_and_negatives(y_gt, y_pred):
        false_positives_rate = []
        false_negatives_rate = []

        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]

            actual_negatives = np.where(y_gt[b] == 0)[0]

            fp = len(set(out_list) - set(target))
            fn = len(set(target) - set(out_list))
            tp = len(set(out_list) & set(target))
            tn = len(set(actual_negatives) - set(out_list))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            false_positives_rate.append(fpr)
            false_negatives_rate.append(fnr)

        return np.mean(false_positives_rate), np.mean(false_negatives_rate)

    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(
                    2 * average_prc[idx] * average_recall[idx] /
                    (average_prc[idx] + average_recall[idx])
                )
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(
                y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(
                y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # roc_auc
    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0
    # precision
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    # macro f1
    f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)
    # fp, fn
    fp, fn = false_positives_and_negatives(y_gt, y_pred)

    # jaccard
    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1), fp, fn
from collections import defaultdict

def eval_one_epoch(model, data_eval, voc_size, max_visits_to_report=7):
    model = model.eval()
    smm_record, ja, prauc, avg_p, avg_r, avg_f1, avg_fp, avg_fn, rate = [[] for _ in range(9)]

    visit_sums = defaultdict(lambda: {"ja": 0.0, "f1": 0.0, "prauc": 0.0, "n": 0})

    # 兼容 numpy / torch
    ddi_adj_positive_np = np.asarray(ddi_adj_positive.detach().cpu().numpy())
    rows, cols = np.where(np.triu(ddi_adj_positive_np, k=1) == 1)
    drugs = set(rows) | set(cols)

    med_cnt, visit_cnt = 0, 0
    gt_drugs_total = 0
    pred_drugs_correct = 0

    for step, input_seq in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        for adm_idx, adm in enumerate(input_seq):
            output, _, _ = model(input_seq[:adm_idx + 1])
            prob = torch.sigmoid(output).detach().cpu().numpy()[0]

            gt_vec = np.zeros(voc_size[2], dtype=np.int64)
            gt_vec[adm[2]] = 1
            pred_vec = (prob >= args.threshold).astype(int)

            y_gt.append(gt_vec)
            y_pred.append(pred_vec)
            y_pred_prob.append(prob)

            pred_idx = np.where(pred_vec == 1)[0]
            y_pred_label.append(sorted(pred_idx.tolist()))
            visit_cnt += 1
            med_cnt += len(pred_idx)

            # DDI 相关
            gt_set = set(np.where(gt_vec == 1)[0])
            pred_correct_set = gt_set & set(pred_idx)
            gt_drugs_total += len(gt_set & drugs)
            pred_drugs_correct += len(pred_correct_set & drugs)

            if 1 <= (adm_idx + 1) <= max_visits_to_report:
                ja_k, prauc_k, _, _, f1_k, _, _ = multi_label_metric(
                    np.array([gt_vec]), np.array([pred_vec]), np.array([prob])
                )
                k = adm_idx + 1
                visit_sums[k]["ja"]    += ja_k
                visit_sums[k]["f1"]    += f1_k
                visit_sums[k]["prauc"] += prauc_k
                visit_sums[k]["n"]     += 1

        smm_record.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1, fp, fn = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja); prauc.append(adm_prauc)
        avg_p.append(adm_avg_p); avg_r.append(adm_avg_r); avg_f1.append(adm_avg_f1)
        avg_fn.append(fn); avg_fp.append(fp)

        llprint('\rtest step: {} / {}'.format(step + 1, len(data_eval)))

    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path)
    ddi_rate_positive = ddi_rate_score(smm_record, path=ddi_adj_positive_path)

    output_str = '\nDDI Rate: {:.4f} , DDI Rate Positive: {:.4f}, Jaccard: {:.4f}, PRAUC: {:.4f}, ' + \
                 'AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}, ' + \
                 'AVG_FP: {:.4f}, AVG_FN: {:.4f}\n'
    llprint(output_str.format(
        ddi_rate, ddi_rate_positive, np.mean(ja), np.mean(prauc), np.mean(avg_p),
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt, np.mean(avg_fp), np.mean(avg_fn)
    ))
    logging.info(output_str.format(
        ddi_rate, ddi_rate_positive, np.mean(ja), np.mean(prauc), np.mean(avg_p),
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt, np.mean(avg_fp), np.mean(avg_fn)
    ))

    positive_ratio = pred_drugs_correct / gt_drugs_total if gt_drugs_total > 0 else 0
    print("Positive DDI 准确预测比例: {:.4f}".format(positive_ratio))

    return ddi_rate, ddi_rate_positive, np.mean(ja), np.mean(prauc), np.mean(avg_p), \
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt, np.mean(avg_fp), np.mean(avg_fn), positive_ratio


def buildPrjSmiles(molecule, med_voc, device="cpu:0"):
    average_index, smiles_all = [], []

    print(len(med_voc.items()))
    for index, ndc in med_voc.items():

        smilesList = list(molecule[ndc])

        """Create each data with the above defined functions."""
        counter = 0
        for smiles in smilesList:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles_all.append(smiles)
                counter += 1
            else:
                continue
        average_index.append(counter)

        """Transform the above each data of numpy
        to pytorch tensor on a device (i.e., CPU or GPU).
        """

    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        average_projection[i, col_counter: col_counter + item] = 1 / item
        col_counter += item


    binary_projection = np.where(average_projection != 0, 1, 0)

    return binary_projection, average_projection, smiles_all
def set_seed():
    seed = int(time.time())
    print("Using seed:", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
def parse_args():
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument('-n', '--note', type=str, default='', help="User notes")
    parser.add_argument("--debug", default=False,
                        help="debug mode, the number of samples, "
                             "the number of generations run are very small, "
                             "designed to run on cpu, the development of the use of")
    parser.add_argument("--Test", default=False, help="test mode")
    parser.add_argument('--model_name', type=str, default='OurModel', help="model name")
    # environment
    parser.add_argument('--dataset', default='mimic-iv', help='mimic-iii/mimic-iv')
    parser.add_argument('--resume_path', default="./saved/mimic-iii/Epoch_6_JA_0.5537_DDI_0.07087.model", type=str,
                        help='path of well trained model, only for evaluating the model, needs to be replaced manually')

    parser.add_argument('--cuda', type=int, default=0, help='which cuda')
    # parameters
    parser.add_argument('--dim', default=1796, type=int, help='model dimension')#1536比较好 可以试试768这种..
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate') #学习率至少1e-5
    parser.add_argument('--dp', default=0.15, type=float, help='dropout ratio')
    parser.add_argument("--regular", type=float, default=0.005, help="regularization parameter")
    parser.add_argument('--target_ddi', type=float, default=0.06, help='expected ddi for training')
    parser.add_argument('--target_ddi_positive', type=float, default=0.15, help='expected ddi for training')
    parser.add_argument('--coef', default=1, type=float, help='coefficient for DDI Loss Weight Annealing')
    parser.add_argument('--epochs', default=10, type=int, help='the epochs for training')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight_decay in training')
    parser.add_argument('--nhead', default=4, type=int, help='the number of head in attention')
    parser.add_argument('--early_stop',default=0,type=int,help = ' early stop after n epochs')
    parser.add_argument('--threshold',default=0.5,type=float,help='the threshold to distinguish 0/1 labels')
    parser.add_argument('--sample', type=float, default=1, help="positive DDI sampling ratio for training")
    args = parser.parse_args()
    if args.Test and args.resume_path is None:
        raise FileNotFoundError('Can\'t Load Model Weight From Empty Dir')

    return args
# def cal_right_positive_rate(model,device,data_test,voc_size): #计算正确的positive 的比例
#     model = model.to(device).eval()
#     print('--------------------Begin Testing--------------------')
#     logging.info("Begin calculate positive rate")

def Test(model, device, data_test, voc_size,data):
    model = model.to(device).eval()
    print('--------------------Begin Testing--------------------')
    logging.info("Begin Testing")
    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
    tic, result, sample_size = time.time(), [], round(len(data_test) * 0.8)
    np.random.seed(0)
    for _ in range(10):
        selected_indices = np.random.choice(len(data_test), size=round(len(data_test) * 0.8), replace=True)
        selected_indices_list = selected_indices.tolist()
        test_sample = [data_test[i] for i in selected_indices_list]
        ddi_rate, ddi_rate_positive, ja, prauc, avg_p, avg_r, avg_f1, avg_med, avg_fp, avg_fn,_ \
            = eval_one_epoch(model, test_sample, voc_size)
        result.append([ja, ddi_rate,ddi_rate_positive,avg_f1, prauc, avg_med, avg_fp, avg_fn])
    result = np.array(result)
    mean, std = result.mean(axis=0), result.std(axis=0)
    metric_list = ['ja', 'ddi_rate','ddi_rate_positive','avg_f1', 'prauc', 'med', 'avg_fp', 'avg_fn']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])
    print(outstring)
    logging.info(outstring)
    print('average test time: {}'.format((time.time() - tic) / 10))
    print('parameters', get_n_params(model))
def Train(model, device, data_train, data_eval, voc_size, args):
    logging.info(f'n_parameters:, {get_n_params(model)}')
    optimizer = AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    num_training_steps = len(data_train) * args.epochs
    num_warmup_steps = int(0)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    history = defaultdict(list)
    best = {"epoch": 0, "ja": 0, "ddi": 0, "prauc": 0, "f1": 0, "med": 0, 'model': model}
    total_train_time, ddi_losses, ddi_values = 0, [], []
    EPOCH = args.epochs

    if args.debug:
        EPOCH = 3
    for epoch in range(EPOCH):
        print(f'----------------Epoch {epoch + 1}------------------')
        logging.info(f'----------Epoch {epoch+1}-----------')
        model = model.train()
        tic, ddi_losses_epoch = time.time(), []
        for step, input_seq in enumerate(data_train):
            for adm_idx, adm in enumerate(input_seq):
                bce_target = torch.zeros((1, voc_size[2])).to(device)
                bce_target[:, adm[2]] = 1
                multi_target = -torch.ones((1, voc_size[2])).long()
                for idx, item in enumerate(adm[2]):
                    multi_target[0][idx] = item
                multi_target = multi_target.to(device)
                result, loss_ddi,loss_ddi_positive = model(input_seq[:adm_idx + 1])

                sigmoid_res = torch.sigmoid(result)

                loss_bce = binary_cross_entropy_with_logits(result, bce_target)
                # loss_bce = focal_loss_fn(result,bce_target)
                loss_multi = multilabel_margin_loss(sigmoid_res, multi_target)

                result = sigmoid_res.detach().cpu().numpy()[0]
                result[result >= args.threshold] = 1
                result[result < args.threshold] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score([[y_label]], path=ddi_adj_path)
                current_positive_ddi_rate = ddi_rate_score_positive([[y_label]],path=ddi_adj_positive_path)

                violation_neg = max(0.0, float(current_ddi_rate) - float(args.target_ddi))
                violation_pos = max(0.0, float(args.target_ddi_positive) - float(current_positive_ddi_rate))
                den = max(float(args.coef), 1e-12)  # 防 0
                alpha = 1.0 if (current_ddi_rate <= args.target_ddi and
                                current_positive_ddi_rate >= args.target_ddi_positive) \
                    else max(0.0, 1.0 - (violation_neg + violation_pos) / den)

                L_pred = 0.95 * loss_bce + 0.05 * loss_multi
                L_ddi = loss_ddi - loss_ddi_positive  # 惩负奖正；若想更强奖励可乘个 lambda_pos

                loss = alpha * L_pred + (1.0 - alpha) * L_ddi
                # if current_ddi_rate <= args.target_ddi:
                #     loss = 0.95 * loss_bce + 0.05 * loss_multi
                # else:
                #     beta = args.coef * (1 - (current_ddi_rate / args.target_ddi))
                #     beta = min(math.exp(beta), 1)
                #     loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * (loss_ddi  - loss_ddi_positive)

                ddi_losses_epoch.append(loss_ddi.detach().cpu().item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # 将 max_norm 设置为一个合理的阈值，比如 1.0
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))

        ddi_losses.append(sum(ddi_losses_epoch) / len(ddi_losses_epoch))
        print(f'\nddi_loss : {ddi_losses[-1]}\n')
        train_time, tic = time.time() - tic, time.time()
        total_train_time += train_time
        ddi_rate, ddi_rate_positive,ja, prauc, avg_p, avg_r, avg_f1, avg_med, avg_fp, avg_fn, _= eval_one_epoch(model, data_eval, voc_size)
        scheduler.step(1 - ja)
        print(f'training time: {train_time}, testing time: {time.time() - tic}')
        ddi_values.append(ddi_rate)
        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
            ))

        if epoch != 0:
            if best['ja'] < ja:
                best['epoch'] = epoch
                best['ja'] = ja
                best['model'] = deepcopy(model.state_dict())
                best['ddi'] = ddi_rate
                best['prauc'] = prauc
                best['f1'] = avg_f1
                best['med'] = avg_med
            print("best_epoch: {}, best_ja: {:.4f}".format(best['epoch'], best['ja']))
        if epoch - best['epoch'] > args.early_stop:  # n个epoch内，验证集性能不上升之后就停
            print("Due to no improvement after n epochs 、thus the model stops ")
            logging.info("Due to no improvement after n epochs 、thus the model stops ")
            break
        # graph_report(history)
    print(history)
    print('avg training time/epoch: {:.4f}'.format(total_train_time / EPOCH))
    # parameter_report(best, regular)
    file_name = 'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(best['epoch'], best['ja'], best['ddi'])
    save_path = os.path.join(save_dir, file_name)

    torch.save(best['model'], open(save_path, 'wb'))

    print(f"Model saved to {save_path}")
    return best['model']


if __name__ == '__main__':

    set_seed()
    args = parse_args()
    log_directory_path = os.path.join('../log', args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log' + str(log_save_id) + '_' + args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)
    device = torch.device('cuda:{}'.format(args.cuda))
    # 注意这里的data_path有修改
    data_path = f'../../data/output/{args.dataset}/records_final.pkl'
    voc_path = f'../../data/output/{args.dataset}/voc_final.pkl'
    ddi_adj_path = f'../../data/output/{args.dataset}/ddi_A_final.pkl'
    ddi_adj_positive_path = f'../../data/output/{args.dataset}' + '/ddi_B_final.pkl'
    ddi_mask_path = f'../../data/output/{args.dataset}/ddi_mask_H.pkl'
    molecule_path = f'../../data/input/{args.dataset}/atc3toSMILES.pkl'
    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = torch.from_numpy(dill.load(Fin)).to(device)
    with open(ddi_adj_positive_path, 'rb') as Fin:
        ddi_adj_positive = torch.from_numpy(dill.load(Fin)).to(device)
    with open(ddi_mask_path, 'rb') as Fin:
        ddi_mask_H = torch.from_numpy(dill.load(Fin)).to(device)
    # print(data[:10])
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
        data = [
            [visit[:3] if len(visit) == 4 else visit for visit in patient]
            for patient in data
        ]
        adm_id = 0
        for patient in data:
            for adm in patient:
                adm.append(adm_id)
                adm_id += 1
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)
    with open(molecule_path, 'rb') as Fin:
        molecule = dill.load(Fin)

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = [
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    ]
    MPNNSet, N_fingerprint, average_projection_MPNN = buildMPNN(molecule, med_voc.idx2word, 2, device)
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point:split_point + eval_len]
    data_test = data[split_point+eval_len:]

    binary_projection, average_projection, smiles_list = buildPrjSmiles(molecule, med_voc.idx2word)

    '''
--------    MoleRec-----------
    '''


    print("Loading all pre-trained embeddings...")

    # 定义权重文件路径
    pretrained_dir = f'../pretrained_weights/{args.dataset}'
    diag_emb_path = os.path.join(pretrained_dir, 'pretrained_diag_embeddings.pt')
    proc_emb_path = os.path.join(pretrained_dir, 'pretrained_proc_embeddings.pt')
    med_emb_path = os.path.join(pretrained_dir, 'pretrained_med_embeddings.pt')
    # 加载权重，如果文件存在的话
    pretrained_embeddings = {}
    if os.path.exists(diag_emb_path):
        pretrained_embeddings['diag'] = torch.load(diag_emb_path, map_location=device)
        print(f"Loaded 'diag' embeddings, shape: {pretrained_embeddings['diag'].shape}")
    else:
        print(f"Warning: 'diag' embedding file not found.")

    if os.path.exists(proc_emb_path):
        pretrained_embeddings['proc'] = torch.load(proc_emb_path, map_location=device)
        print(f"Loaded 'proc' embeddings, shape: {pretrained_embeddings['proc'].shape}")
    else:
        print(f"Warning: 'proc' embedding file not found.")

    if os.path.exists(med_emb_path):
        pretrained_embeddings['med'] = torch.load(med_emb_path, map_location=device)
        print(f"Loaded 'med' embeddings, shape: {pretrained_embeddings['med'].shape}")
    else:
        print(f"Warning: 'med' embedding file not found.")

    model = OurModel(
        args = args,
        tensor_ddi_adj=ddi_adj,
        tensor_ddi_adj_positive=ddi_adj_positive,
        dropout=args.dp,
        emb_dim=args.dim,
        voc_size=voc_size,
        device=device,
        sample=args.sample,
        pretrained_embeddings=pretrained_embeddings
    ).to(device)


    print("1.Training Phase")
    if args.Test:
        print("Test mode, skip training phase")
        with open(args.resume_path, 'rb') as Fin:
            model.load_state_dict(torch.load(Fin, map_location=device))
    else:
        best_model_state_dict = Train(model, device, data_train, data_eval, voc_size, args)
        print("Loading best trained weights for testing...")
        model.load_state_dict(best_model_state_dict)
    print("2.Testing Phase")
    logging.info("Testing Phase")
    # print(data[18])
    eval_one_epoch(model,data_test,voc_size)
    Test(model, device, data_test, voc_size,data)
