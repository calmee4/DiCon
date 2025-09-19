import os
import time
import dill
import torch
import logging
import argparse
import numpy as np
import torch.optim as optim
import sys
from copy import deepcopy

# 导入正常的库
from tqdm import tqdm
sys.path.append("..")
sys.path.append("../..")
import torch.nn.functional as F
from prettytable import PrettyTable
from collections import defaultdict
from utils.util import llprint, multi_label_metric, ddi_rate_score, set_seed,ddi_rate_score_positive
from utils.util import multi_label_metric, ddi_rate_score, get_n_params, create_log_id, logging_config, get_grouped_metrics, get_model_path, ddi_rate_score_positive
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CyclicLR, StepLR

from models.FastRx import FastRx
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

#导入配置
# /mnt/e/OurModel/OurModel_Final/src/log/mimic-iii/Fastrx/log
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default=' ', help="User notes")
    parser.add_argument('-t', '--test', action='store_true', default=False, help="test mode")
    parser.add_argument('-s', '--single', action='store_true', default=False, help="single visit")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default='None', help='log dir prefix like "log0"')
    parser.add_argument('--model_name', type=str, default='FastRx', help="model name")
    parser.add_argument('--dataset', type=str, default='mimic-iii', help='dataset')
    parser.add_argument('--early_stop', type=int, default=50,
                        help='early stop after this many epochs without improvement')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--target_ddi', type=float, default=0.05, help='target ddi')
    parser.add_argument('--T', type=float, default=0.5, help='T')  # T越小，使用DDI loss的概率越小
    parser.add_argument('--weight_decay', type=float, default=0.85, help="decay weight")
    # parser.add_argument('--weight_multi', type=float, default=0.01, help='weight of multilabel_margin_loss')
    parser.add_argument('--weight_multi', type=float, default=0.005, help='weight of multilabel_margin_loss')
    parser.add_argument('--embed_dim', type=int, default=512, help='dimension of node embedding(randomly initialize)')
    parser.add_argument('--cuda', type=int, default=0, help='which cuda')
    parser.add_argument('--resume_path', default="./saved/mimic-iv/FastRx/Epoch_49_JA_0.4494_DDI_0.07343.model", type=str,
                        help='path of well trained model, only for evaluating the model, needs to be replaced manually')
    args = parser.parse_args()
    return args
# ---- add: sanitize note for safe folder names ----
import re
def sanitize_note(s):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s or '').strip())

ddi_adj_positive_path = f'../../data/output/mimic-iv'+ '/ddi_B_final.pkl'
ddi_adj_path = f'../../data/output/mimic-iv'+ '/ddi_A_final.pkl'

ddi_adj_positive = dill.load(open(ddi_adj_positive_path, 'rb'))
def eval_one_epoch(model, data_eval, voc_size):
    model = model.eval()
    smm_record, ja, prauc, avg_p, avg_r, avg_f1, avg_fp, avg_fn = [[] for _ in range(8)]

    # DDI positive 药物全集
    ddi_adj_positive_np = ddi_adj_positive
    rows, cols = np.where(np.triu(ddi_adj_positive_np, k=1) == 1)
    drugs = set(rows) | set(cols)

    med_cnt, visit_cnt = 0, 0
    gt_drugs_total = 0       # ground truth 中属于 positive DDI 的药物数量
    pred_drugs_correct = 0   # 预测正确且属于 positive DDI 的药物数量

    for step, input_seq in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input_seq):
            output = model(input_seq[:adm_idx + 1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            output = torch.sigmoid(output).detach().cpu().numpy()[0]
            y_pred_prob.append(output)

            y_pred_tmp = output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

            # ==== 这里统计 Positive DDI 准确比例 ====
            gt_set = set(np.where(y_gt_tmp == 1)[0])
            pred_correct_set = gt_set & set(y_pred_label_tmp)

            gt_drugs_total += len(gt_set & drugs)
            pred_drugs_correct += len(pred_correct_set & drugs)

        smm_record.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

        llprint('\rtest step: {} / {}'.format(step + 1, len(data_eval)))
    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path)
    ddi_rate_positive = ddi_rate_score(smm_record, path=ddi_adj_positive_path)

    positive_ratio = pred_drugs_correct / gt_drugs_total if gt_drugs_total > 0 else 0

    output_str = '\nDDI Rate: {:.4f} , DDI Rate Positive: {:.4f}, Jaccard: {:.4f}, PRAUC: {:.4f}, ' + \
                 'AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}, ' + \
                 'Positive_Ratio: {:.4f}\n'
    llprint(output_str.format(
        ddi_rate, ddi_rate_positive, np.mean(ja), np.mean(prauc), np.mean(avg_p),
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt,
        positive_ratio
    ))
    logging.info(output_str.format(
        ddi_rate, ddi_rate_positive, np.mean(ja), np.mean(prauc), np.mean(avg_p),
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt,
        positive_ratio
    ))

    return ddi_rate, ddi_rate_positive, np.mean(ja), np.mean(prauc), np.mean(avg_p), \
           np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt


# 结果评估
@torch.no_grad()
def evaluator(
    args, model, data_val, voc_size, epoch,
    *,  # 强制关键字参数，防止再把路径传错位
    ddi_adj_path, ddi_adj_path_positive,
    rec_results_path=None, mode='Test'
):
    import numpy as np
    import torch
    import os, logging
    from tqdm import tqdm

    def _fp_fn_rates(y_gt, y_pred):
        # 计算每个病人的 FPR/FNR，然后取均值
        false_positives_rate, false_negatives_rate = [], []
        for b in range(y_gt.shape[0]):
            target = set(np.where(y_gt[b] == 1)[0])
            pred   = set(np.where(y_pred[b] == 1)[0])
            actual_neg = set(np.where(y_gt[b] == 0)[0])
            fp = len(pred - target)
            fn = len(target - pred)
            tp = len(pred & target)
            tn = len(actual_neg - pred)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            false_positives_rate.append(fpr)
            false_negatives_rate.append(fnr)
        return float(np.mean(false_positives_rate)), float(np.mean(false_negatives_rate))

    model.eval()

    # 访次数分组统计（按 1~5+ 组）
    ja_visit = [[] for _ in range(5)]

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    avg_fp, avg_fn = [], []
    visit_weights = []
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    rec_results, all_pred_prob = [], []

    for patient in tqdm(data_val, ncols=60, total=len(data_val), desc="Evaluation"):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        visit_weights_patient = []

        if mode == "Test":
            all_diseases, all_procedures, all_medications = [], [], []

        for adm_idx, adm in enumerate(patient):
            if mode == "Test":
                all_diseases.append(adm[0])
                all_procedures.append(adm[1])
                all_medications.append(adm[2])

            # 前 adm_idx+1 次就诊作为输入
            logits = model(patient[:adm_idx + 1])

            # 构造 GT（基于 voc_size[2]）
            y_gt_tmp = np.zeros(voc_size[2], dtype=np.float32)
            y_gt_tmp[adm[2]] = 1.0
            y_gt.append(y_gt_tmp)

            # 访次权重（没有就给 1）
            vw = adm[3] if len(adm) > 3 else 1
            visit_weights_patient.append(vw)

            # 概率与二值预测
            prob = torch.sigmoid(logits).detach().cpu().numpy()[0]
            y_pred_prob.append(prob)

            pred = (prob >= 0.5).astype(np.float32)
            y_pred.append(pred)
            y_pred_label.append(sorted(np.where(pred == 1)[0]))

            visit_cnt += 1
            med_cnt += int(pred.sum())

        smm_record.append(y_pred_label)

        # 逐病人指标
        ygt = np.array(y_gt)
        ypd = np.array(y_pred)
        ypb = np.array(y_pred_prob)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(ygt, ypd, ypb)
        fpr, fnr = _fp_fn_rates(ygt, ypd)

        if mode == "Test":
            # 按病人访次数分箱
            vlen = len(patient)
            if vlen < 5:
                ja_visit[vlen - 1].append(adm_ja)
            else:
                ja_visit[4].append(adm_ja)
            rec_results.append([all_diseases, all_procedures, all_medications, y_pred_label, visit_weights_patient, [adm_ja]])

        ja.append(adm_ja); prauc.append(adm_prauc)
        avg_p.append(adm_avg_p); avg_r.append(adm_avg_r); avg_f1.append(adm_avg_f1)
        avg_fp.append(fpr); avg_fn.append(fnr)
        visit_weights.append(np.max(visit_weights_patient))

    # 保存可视化/中间件（仅在 Test 模式且给了路径时）
    if mode == "Test" and rec_results_path:
        os.makedirs(rec_results_path, exist_ok=True)
        dill.dump(rec_results, open(os.path.join(rec_results_path, 'rec_results.pkl'), 'wb'))
        dill.dump(ja_visit, open(os.path.join(rec_results_path, 'ja_result.pkl'), 'wb'))

    # DDI
    ddi_rate          = ddi_rate_score(smm_record, path=ddi_adj_path)
    ddi_rate_positive = ddi_rate_score(smm_record, path=ddi_adj_path_positive)

    get_grouped_metrics(ja, visit_weights)  # 这行内部自带 logging

    logging.info(
        f"Epoch {epoch:03d}, Jaccard: {np.mean(ja):.4}, DDI Rate: {ddi_rate:.4}, Positive DDI Rate: {ddi_rate_positive:.4},"
        f"PRAUC: {np.mean(prauc):.4}, AVG_F1: {np.mean(avg_f1):.4},  AVG_PRC: {np.mean(avg_p):.4f}, "
        f"AVG_RECALL: {np.mean(avg_r):.4f}, AVG_MED: {med_cnt / visit_cnt:.4}"
    )

    # ==== 统一返回 10 个指标（对齐你旧版 eval_one_epoch 的顺序）====
    return (
        ddi_rate,
        ddi_rate_positive,
        float(np.mean(ja)),
        float(np.mean(prauc)),
        float(np.mean(avg_p)),
        float(np.mean(avg_r)),
        float(np.mean(avg_f1)),
        float(med_cnt / visit_cnt),
    )
@torch.no_grad()
def test_bootstrap(
    args, model, device, data_test, voc_size,
    *, ddi_adj_path, ddi_adj_path_positive,
    rounds=10, ratio=0.8, seed=0
):
    import numpy as np, time
    model = model.to(device).eval()
    print('--------------------Begin Testing (bootstrap)--------------------')

    rng = np.random.default_rng(seed)
    n = len(data_test)
    k = int(round(n * ratio))
    rows = []
    print(" calculate ddi positive rate ")
    # evaluator(
    #     args, model, data_test, voc_size, epoch=0,
    #     ddi_adj_path=ddi_adj_path,
    #     ddi_adj_path_positive=ddi_adj_path_positive,
    #     rec_results_path=None,  # bootstrap 不落盘
    #     mode='Eval'  # 用 Eval
    # )
    eval_one_epoch(model,data_test,voc_size)
    print("calculate over")
    tic = time.time()
    for _ in range(rounds):
        idx = rng.choice(n, size=k, replace=True).tolist()
        subset = [data_test[i] for i in idx]
        (ddi_rate, ddi_rate_pos, ja, prauc, avg_p, avg_r, f1, avg_med, avg_fp, avg_fn) = evaluator(
            args, model, subset, voc_size, epoch=0,
            ddi_adj_path=ddi_adj_path,
            ddi_adj_path_positive=ddi_adj_path_positive,
            rec_results_path=None,  # bootstrap 不落盘
            mode='Eval'             # 用 Eval
        )
        rows.append([ja, ddi_rate, ddi_rate_pos, f1, prauc, avg_med, avg_fp, avg_fn])

    arr = np.array(rows, dtype=np.float64)
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    metric_list = ['ja', 'ddi_rate', 'ddi_rate_positive', 'avg_f1', 'prauc', 'med', 'avg_fp', 'avg_fn']

    out = ''.join(f"{name}:\t{m:.4f} ± {s:.4f} & \n"
                  for name, m, s in zip(metric_list, mean, std))
    print(out)
    print('average test time:', (time.time() - tic) / rounds)
    print('parameters', get_n_params(model))

def transfer_procedure(data, diag_voc):
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j][1])):
                data[i][j][1][k] += len(diag_voc.idx2word) + 1
    return data

def main(args):
    # set logger
    if args.test:
        args.note = f'test of {args.log_dir_prefix}'
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
    ehr_adj_path = f'../../data/output/{args.dataset}' + '/ehr_adj_final.pkl'
    device = torch.device('cuda:{}'.format(args.cuda))
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = transfer_procedure(data, diag_voc)
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    val_len = int(len(data[split_point:]) / 2)
    data_val = data[split_point:split_point + val_len]
    data_test = data[split_point+val_len:]
    if args.single:
        data_train = [[visit] for patient in data_train for visit in patient]
        data_val = [[visit] for patient in data_val for visit in patient]
        data_test = [[visit] for patient in data_test for visit in patient]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    print(ehr_adj.shape)
    model = FastRx(args,voc_size,ehr_adj,ddi_adj) #这样模型就定义好了捏
    logging.info(model)
    # test
    if args.test:
        rec_results_path = os.path.join(save_dir, 'rec_results')

        with open(args.resume_path, 'rb') as Fin:
            model.load_state_dict(torch.load(Fin, map_location=device))
        model.to(device=device)

        evaluator(
            args, model, data_test, voc_size, epoch=0,
            ddi_adj_path=ddi_adj_path,  # ../../data/output/.../ddi_A_final.pkl
            ddi_adj_path_positive=ddi_adj_path_positive,  # ../../data/output/.../ddi_B_final.pkl
            rec_results_path=rec_results_path,
            mode='Test'
        )

        test_bootstrap(
            args, model, device, data_test, voc_size,
            ddi_adj_path=ddi_adj_path,
            ddi_adj_path_positive=ddi_adj_path_positive,
            rounds=10, ratio=0.8, seed=0
        )

        return
    else:
        writer = SummaryWriter(save_dir) # 自动生成log文件夹

    # train and validation
    model.to(device=device)
    logging.info(f'n_parameters:, {get_n_params(model)}')
    optimizer = Adam(model.parameters(), lr=args.lr)
    print('parameters', get_n_params(model))

    EPOCH = 50
    best_epoch, best_ja = 0, 0
    for epoch in range(EPOCH):
        epoch += 1
        print(
            f'\nepoch {epoch} --------------------------model_name={args.model_name}, dataset={args.dataset}, logger={log_save_id}')
        model.train()
        prediction_loss_cnt, neg_loss_cnt = 0, 0
        loss_train, loss_val = 0, 0
        for step, patient in tqdm(enumerate(data_train), ncols=60, desc="Training",
                                  total=len(data_train)):  # every patient
            for idx, adm in enumerate(patient):
                seq_input = patient[:idx + 1]
                result, loss_ddi = model(seq_input)  # result = target_output1
                prediction_loss, neg_loss, loss = loss_func(voc_size, adm, result, loss_ddi, ddi_adj_path, device,
                                                            args.weight_multi)
                prediction_loss_cnt += prediction_loss
                neg_loss_cnt += neg_loss
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                loss_train += loss.item() / len(data_train)  # 用于记录每个epoch的总loss

        with torch.no_grad():
            for step, patient in tqdm(enumerate(data_val), ncols=60, desc="Val loss",
                                      total=len(data_val)):  # every patient
                for idx, adm in enumerate(patient):  # every admission
                    seq_input = patient[:idx + 1]  # 前T次数据输入
                    result, loss_ddi = model(seq_input)
                    _, _, loss = loss_func(voc_size, adm, result, loss_ddi, ddi_adj_path, device, args.weight_multi)
                    loss_val += loss.item() / len(data_val)  # 用于记录每个epoch的总loss

        logging.info(f'loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}')
        # evaluation
        args.T *= args.weight_decay


        ddi_rate, ddi_rate_pos, ja, prauc, avg_p, avg_r, f1, avg_med, avg_fp, avg_fn = evaluator(
            args, model, data_val, voc_size, epoch,
            ddi_adj_path=ddi_adj_path,
            ddi_adj_path_positive=ddi_adj_path_positive,
            rec_results_path=None,  # 训练/验证不落盘
            mode='Eval'  # 训练期用 Eval，避免额外文件与日志
        )

        tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch,
                          loss_train, loss_val)

        # save best epoch
        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja, best_prauc, best_ddi_rate, best_avg_med = ja, prauc, ddi_rate, avg_med
            best_model_state = deepcopy(model.state_dict())
        logging.info('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))
        # print ('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))

        if epoch - best_epoch > args.early_stop:  # n个epoch内，验证集性能不上升之后就停
            break

    # save best model
    logging.info('Train finished')
    torch.save(best_model_state, open(os.path.join(save_dir, \
                                                   'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(best_epoch, best_ja,
                                                                                              ddi_rate)), 'wb'))
    model.load_state_dict(best_model_state)
    test_bootstrap(
        args, model, device, data_test, voc_size,
        ddi_adj_path=ddi_adj_path,
        ddi_adj_path_positive=ddi_adj_path_positive,
        rounds=10, ratio=0.8, seed=0
    )

def loss_func(voc_size, adm, result, loss_ddi, ddi_adj_path, device, weight_multi):
    loss_bce_target = np.zeros((1, voc_size[2]))
    loss_bce_target[:, adm[2]] = 1

    loss_multi_target = np.full((1, voc_size[2]), -1)
    for idx, item in enumerate(adm[2]):
        loss_multi_target[0][idx] = item

    prediction_loss_cnt, neg_loss_cnt = 0, 0
    loss_bce = F.binary_cross_entropy_with_logits(result, torch.FloatTensor(loss_bce_target).to(device))
    loss_multi = F.multilabel_margin_loss(F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device))
    result = F.sigmoid(result).detach().cpu().numpy()[0]
    result[result >= 0.5] = 1
    result[result < 0.5] = 0
    y_label = np.where(result == 1)[0]
    current_ddi_rate = ddi_rate_score([[y_label]], path=ddi_adj_path)
    # print("current_ddi_rate:{}".format(current_ddi_rate))

    if current_ddi_rate <= args.target_ddi:
        loss = (1 - args.weight_multi) * loss_bce + args.weight_multi * loss_multi
        prediction_loss_cnt += 1
        # print("current_ddi_rate<=args.target_ddi")
    else:
        rnd = np.exp((args.target_ddi - current_ddi_rate) / args.T)
        # print("rnd:{}".format(rnd))
        if np.random.rand(1) < rnd:
            # print("np.random.rand(1) < rnd:")
            loss = loss_ddi
            neg_loss_cnt += 1
        else:
            loss = (1 - args.weight_multi) * loss_bce + args.weight_multi * loss_multi
            prediction_loss_cnt += 1
    return prediction_loss_cnt, neg_loss_cnt, loss


def tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch,
                        loss_train=0, loss_val=0):
    if epoch > 0:
        writer.add_scalar('Loss/Train', loss_train, epoch)
        writer.add_scalar('Loss/Val', loss_val, epoch)

    writer.add_scalar('Metrics/Jaccard', ja, epoch)
    writer.add_scalar('Metrics/prauc', prauc, epoch)
    writer.add_scalar('Metrics/DDI', ddi_rate, epoch)
    writer.add_scalar('Metrics/Med_count', avg_med, epoch)

import random
if __name__ == '__main__':
    seed = int(time.time())

    print("Using seed:", seed)  # 可以打印出来方便调试

    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args = get_args()
    main(args)
    # torch.manual_seed(1203)
    # np.random.seed(2048)
    # args = get_args()
    # main(args)

