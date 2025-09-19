import os
import time
import dill
import torch
import logging
import argparse
import torch.optim as optim
import sys
from copy import deepcopy
from dowhy import CausalModel
import os
import math
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss
from torch.optim import AdamW,Adam
from torch.utils.data.dataloader import DataLoader
import dill
import networkx as nx
import pandas as pd
import statsmodels.api as sm
from cdt.causality.graph import GES
from dowhy import CausalModel
from tqdm import tqdm
# 导入正常的库
from tqdm import tqdm
sys.path.append("..")
sys.path.append("../..")
import torch.nn.functional as F
from prettytable import PrettyTable
from collections import defaultdict
from rdkit import Chem
from utils.util import llprint, multi_label_metric, ddi_rate_score, set_seed,ddi_rate_score_positive,llprint,  sequence_output_process, ddi_rate_score, get_n_params, output_flatten, print_result
from utils.util import multi_label_metric, ddi_rate_score, get_n_params, create_log_id, logging_config, get_grouped_metrics, get_model_path, ddi_rate_score_positive,sequence_metric
from utils.util import llprint, ddi_rate_score, get_n_params , Regularization,buildMPNN
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, f1_score
from models.gnn import graph_batch_from_smile
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CyclicLR, StepLR
import pandas as pd
from models.OurModel import OurModel
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from data_loader_new import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace, pad_batch_v2_val

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import random
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.optim.swa_utils import AveragedModel, update_bn
from models.VITA import VITA
torch.manual_seed(1203)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = 'VITA_mimic_iii'  # VITA_mimic_iv
past_name = 'past'
resume_path = 'saved/VITA_mimic_iii/08Epoch_35_JA_0.5223_DDI_0.07816_LOSS_1.2399385026098753.model'

"""
saved best epoch
# 'saved/VITA_mimic_iii_/08Epoch_35_JA_0.5223_DDI_0.07816_LOSS_1.2399385026098753.model'
# 'saved/VITA_mimic_iv/Epoch_52_JA_0.5207_DDI_0.09055_LOSS_1.3179462606297943.model'
"""
print(model_name)

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
# parser.add_argument('--Test', action='store_true', default=True, help="test mode")
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimension size')
parser.add_argument('--max_len', type=int, default=45, help='maximum prediction medication sequence')
parser.add_argument('--beam_size', type=int, default=4, help='max num of sentences in beam searching')
parser.add_argument('--dataset', type=str, default="mimic-iii", help='dataset')

parser.add_argument('--note', type=str, default='', help="note")

args = parser.parse_args()

def eval_recommend_batch(model, batch_data, device, TOKENS, args):
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS

    diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = batch_data
    # continue
    # Replace padding values according to the vocab
    diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
    procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
    dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
    stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
    dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
    stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
    medications = medications.to(device)
    m_mask_matrix = m_mask_matrix.to(device)
    d_mask_matrix = d_mask_matrix.to(device)
    p_mask_matrix = p_mask_matrix.to(device)
    dec_disease_mask = dec_disease_mask.to(device)
    stay_disease_mask = stay_disease_mask.to(device)
    dec_proc_mask = dec_proc_mask.to(device)
    stay_proc_mask = stay_proc_mask.to(device)

    batch_size = medications.size(0)
    max_visit_num = medications.size(1)
    input_disease_embdding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory, count, people = model.encode(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix,
        seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20)


    partial_input_medication = torch.full((batch_size, max_visit_num, 1), SOS_TOKEN).to(device)
    parital_logits = None
    cross_visit_scores_numpy = cross_visit_scores.cpu().detach().numpy

    for i in range(args.max_len):
        partial_input_med_num = partial_input_medication.size(2)
        partial_m_mask_matrix = torch.zeros((batch_size, max_visit_num, partial_input_med_num), device=device).float()

        parital_logits = model.decode(partial_input_medication, input_disease_embdding,  encoded_medication, last_seq_medication, cross_visit_scores,
            d_mask_matrix, p_mask_matrix, partial_m_mask_matrix, last_m_mask, drug_memory)
        _, next_medication = torch.topk(parital_logits[:, :, -1, :], 1, dim=-1)
        partial_input_medication = torch.cat([partial_input_medication, next_medication], dim=-1)

    return parital_logits, people, cross_visit_scores_numpy


def test(model, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, epoch, device, TOKENS, ddi_adj, args):
    model.eval()
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt_list = []
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    all_pred_list = []
    all_label_list = []

    ja_by_visit = [[] for _ in range(5)]
    auc_by_visit = [[] for _ in range(5)]
    pre_by_visit = [[] for _ in range(5)]
    recall_by_visit = [[] for _ in range(5)]
    f1_by_visit = [[] for _ in range(5)]
    smm_record_by_visit = [[] for _ in range(5)]

    for idx, data in enumerate(test_dataloader):
        diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
            d_mask_matrix, p_mask_matrix, m_mask_matrix, \
            dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
            dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data
        visit_cnt += seq_length.sum().item()

        output_logits, output_probs, gumbel_pick_index, cross_visit_scores_numpy = test_recommend_batch(model, data,
                                                                                                        device, TOKENS,
                                                                                                        ddi_adj, args)

        labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2],
                                             END_TOKEN, device, training=False, testing=True, max_len=args.max_len)
        _, probs = output_flatten(medications, output_probs, seq_length, m_length_matrix, voc_size[2], END_TOKEN,
                                  device, training=False, testing=True, max_len=args.max_len)
        y_gt = []
        y_pred = []
        y_pred_label = []
        y_pred_prob = []

        label_hisory = []
        label_hisory_list = []
        pred_list = []
        jaccard_list = []

        def cal_jaccard(set1, set2):
            if not set1 or not set2:
                return 0
            set1 = set(set1)
            set2 = set(set2)
            a, b = len(set1 & set2), len(set1 | set2)
            return a / b

        def cal_overlap_num(set1, set2):
            count = 0
            for d in set1:
                if d in set2:
                    count += 1
            return count

        # Predicted results for each admission.
        for label, prediction, prob_list in zip(labels, predictions, probs):
            try:
                label_hisory += label.tolist()  ### case study

                y_gt_tmp = np.zeros(voc_size[2])
                y_gt_tmp[label] = 1  # 0-1 sequence, representing the correct label
                y_gt.append(y_gt_tmp)

                out_list = []
                out_prob_list = []
                for med, prob in zip(prediction, prob_list):
                    if med in [voc_size[2], voc_size[2] + 1]:
                        break
                    out_list.append(med)
                    out_prob_list.append(prob[:-2])  # Remove the SOS and EOS symbols.

                ## case study
                if label_hisory:
                    jaccard_list.append(cal_jaccard(prediction, label_hisory))
                pred_list.append(out_list)
                label_hisory_list.append(label.tolist())

                # For drugs that haven't been predicted, take the average probability at each position; otherwise, directly take the corresponding probability.
                pred_out_prob_list = np.max(out_prob_list, axis=0)

                for i in range(131):
                    if i in out_list:
                        pred_out_prob_list[i] = out_prob_list[out_list.index(i)][i]

                y_pred_prob.append(pred_out_prob_list)
                y_pred_label.append(out_list)

                # prediction label
                y_pred_tmp = np.zeros(voc_size[2])
                y_pred_tmp[out_list] = 1
                y_pred.append(y_pred_tmp)
                med_cnt += len(prediction)
                med_cnt_list.append(len(prediction))
            except ValueError:
                pass

        smm_record.append(y_pred_label)
        for i in range(min(len(labels), 5)):
            try:
                single_ja, single_auc, single_p, single_r, single_f1 = sequence_metric(np.array([y_gt[i]]),
                                                                                       np.array([y_pred[i]]),
                                                                                       np.array([y_pred_prob[i]]),
                                                                                       np.array([y_pred_label[i]]))
                ja_by_visit[i].append(single_ja)
                auc_by_visit[i].append(single_auc)
                pre_by_visit[i].append(single_p)
                recall_by_visit[i].append(single_r)
                f1_by_visit[i].append(single_f1)
                smm_record_by_visit[i].append(y_pred_label[i:i + 1])
            except IndexError:
                pass

        # Store all prediction results.
        all_pred_list.append(pred_list)
        all_label_list.append(labels)
        try:
            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)
            llprint('\rtest step: {} / {}'.format(idx, len(test_dataloader)))
        except IndexError:
            pass

    print('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
    print('jaccard:', [np.mean(buf) for buf in ja])
    print('prauc:', [np.mean(buf) for buf in prauc])
    print('f1:', [np.mean(buf) for buf in avg_f1])
    print('DDI:', [ddi_rate_score(buf) for buf in smm_record_by_visit])
    pickle.dump(all_pred_list, open('out_list.pkl', 'wb'))
    pickle.dump(all_label_list, open('out_list_gt.pkl', 'wb'))

    return smm_record, ja, prauc, avg_p, avg_r, avg_f1, med_cnt_list, gumbel_pick_index, cross_visit_scores_numpy



def eval(model, eval_dataloader, voc_size, device, TOKENS, args):
    model.eval()
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    smm_record = []
    med_cnt, visit_cnt = 0, 0

    for idx, data in enumerate(eval_dataloader):
        diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
            d_mask_matrix, p_mask_matrix, m_mask_matrix, \
            dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
            dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data
        visit_cnt += seq_length.sum().item()

        output_logits, people, cross_visit_scores_numpy = eval_recommend_batch(model, data, device, TOKENS, args)

        # Predicted results for each medical condition.
        labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2],
                                             END_TOKEN, device, training=False, testing=False, max_len=args.max_len)

        y_gt = []  # Ground truth, representing the correct labels as 0-1 sequences
        y_pred = []  # Predicted results, 0-1 sequences
        y_pred_prob = []  # Average probability for each drug predicted, non-0-1 sequences
        y_pred_label = []  # Predicted results, non-0-1 sequences
        # Predicted results for each admission
        for label, prediction in zip(labels, predictions):
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[label] = 1  # 0-1 sequence, representing the correct labels
            y_gt.append(y_gt_tmp)

            # label: med set
            # prediction: [med_num, probability]
            out_list, sorted_predict = sequence_output_process(prediction, [voc_size[2], voc_size[2] + 1])
            y_pred_label.append(sorted(sorted_predict))
            y_pred_prob.append(np.mean(prediction[:, :-2], axis=0))

            # prediction label
            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            med_cnt += len(sorted_predict)

        smm_record.append(y_pred_label)

        try:
            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)
            llprint('\rtest step: {} / {}'.format(idx, len(eval_dataloader)))
        except ValueError:
            pass
            # ddi rate
    ddi_rate = ddi_rate_score(smm_record,
                              path=f'../../data/output/{args.dataset}/ddi_A_final.pkl')  # '../data/mimic-iv/ddi_A_final2.pkl'
    ddi_rate_positive = ddi_rate_score(smm_record,
                              path=f'../../data/output/{args.dataset}/ddi_B_final.pkl')  # '../data/mimic-iv/ddi_A_final2.pkl'

    llprint('\nDDI Rate: {}, Jaccard: {},  PRAUC: {}, AVG_PRC: {}, AVG_RECALL: {}, AVG_F1: {}, AVG_MED: {}\n'.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    return ddi_rate, ddi_rate_positive, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(
        avg_f1), med_cnt / visit_cnt, people, cross_visit_scores_numpy


def main(args):
    log_directory_path = os.path.join('../log', args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log' + str(log_save_id) + '_' + args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    # load data
    data_path = f'../../data/output/{args.dataset}/records_final.pkl'  # '../data/mimic-iv/records_final2.pkl'
    voc_path = f'../../data/output/{args.dataset}/voc_final.pkl'  # '../data/mimic-iv/voc_final2.pkl'
    ehr_adj_path = f'../../data/output/{args.dataset}/ehr_adj_final.pkl'  # '../data/mimic-iv/ehr_adj_final2.pkl'
    ddi_adj_path = f'../../data/output/{args.dataset}/ddi_A_final.pkl'  # '../data/mimic-iv/ddi_A_final2.pkl'
    ddi_mask_path = f'../../data/output/{args.dataset}/ddi_mask_H.pkl'  # '../data/mimic-iv/ddi_mask_H2.pkl'
    device = torch.device('cuda')
    print(device)

    data = dill.load(open(data_path, 'rb'))

    data = [x for x in data if len(x) >= 2]  ### Only admissions more than twice as input ####
    voc = dill.load(open(voc_path, 'rb'))
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    # frequency statistic
    med_count = defaultdict(int)
    for patient in data:
        for adm in patient:
            for med in adm[2]:
                med_count[med] += 1

    ## rare first
    for i in range(len(data)):
        for j in range(len(data[i])):
            cur_medications = sorted(data[i][j][2], key=lambda x: med_count[x])
            data[i][j][2] = cur_medications

    ## data split
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]

    model = VITA(voc_size, ehr_adj, ddi_adj, ddi_mask_H, emb_dim=args.emb_dim, device=device)
    model.to(device)  # <<< 必加
    logging.info(model)


    if args.Test:
        ddi_rate_list_t, ja_list_t, prauc_list_t, avg_p_list_t, avg_r_list_t, avg_f1_list_t, avg_med_list_t = [], [], [], [], [], [], []
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        result = []
        all_people = []
        all_score = []
        people_length = []
        predicted_med = []
        for step, input in enumerate(data_test):
            step_l = []
            for idx, adm in enumerate(input):
                if idx == 0:
                    pass
                else:
                    adm_list = []

                    seq_input = input[:idx + 1]  ## Added cumulatively
                    adm_list.append(seq_input)
                    test_dataset = mimic_data(adm_list)
                    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=True,
                                                 pin_memory=True)

                    smm_record, ja, prauc, precision, recall, f1, med_num, gumbel_pick_index, cross_visit_scores_numpy = test(
                        model, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, 0, device, TOKENS, ddi_adj, args)

                    data_num = len(ja)
                    final_length = int(data_num)
                    idx_list = list(range(data_num))
                    random.shuffle(idx_list)
                    idx_list = idx_list[:final_length]
                    avg_ja = np.mean([ja[i] for i in idx_list])
                    avg_prauc = np.mean([prauc[i] for i in idx_list])
                    avg_precision = np.mean([precision[i] for i in idx_list])
                    avg_recall = np.mean([recall[i] for i in idx_list])
                    avg_f1 = np.mean([f1[i] for i in idx_list])
                    avg_med = np.mean([med_num[i] for i in idx_list])
                    cur_smm_record = [smm_record[i] for i in idx_list]
                    ddi_rate = ddi_rate_score(cur_smm_record,
                                              path=f'../../data/output/{args.dataset}/ddi_A_final.pkl')  # '../data/mimic-iv/ddi_A_final2.pkl'
                    ddi_rate_positive = ddi_rate_score_positive(smm_record,
                                                                path=f"../../data/output/{args.dataset}/ddi_B_final.pkl")
                    isnan_list = [np.isnan(i) for i in
                                  [ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med]]
                    if True not in isnan_list:
                        result.append([ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med])
                        llprint(
                            '\nDDI Rate: {}, Jaccard: {}, PRAUC: {}, AVG_PRC: {}, AVG_RECALL: {}, AVG_F1: {}, AVG_MED: {}\n'.format(
                                ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med))
            predicted_med.append(smm_record)
            all_people.append(gumbel_pick_index)
            print(cross_visit_scores_numpy)
            print(gumbel_pick_index, len(seq_input))
            all_score.append(cross_visit_scores_numpy)
            people_length.append(len(input))
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)
        dill.dump(all_people, open(os.path.join('saved', model_name, 'gumbel_pick.pkl'),
                                   'wb'))  ## If there's no stored visit time chosen by the model, insert 0
        dill.dump(all_score, open(os.path.join('saved', model_name, 'all_score.pkl'),
                                  'wb'))  ## Save all attention scores for each visit
        dill.dump(people_length, open(os.path.join('saved', model_name, 'people_length.pkl'),
                                      'wb'))  ## Save the number of visits for each patient
        dill.dump(predicted_med,
                  open(os.path.join('saved', model_name, 'predicted_med.pkl'), 'wb'))  ## Save the predicted medications
        print(outstring)
        print('test time: {}'.format(time.time() - tic))

        return

    logging.info(f'n_parameters:, {get_n_params(model)}')
    optimizer = Adam(model.parameters(), lr=args.lr)
    logging.info(f'Optimizer: {optimizer}')

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 100  # 200
    temp_min = 0.5
    ANNEAL_RATE = 0.000003
    temp_max = 25
    ANNEAL_RATE2 = 0.000003

    for epoch in range(EPOCH):
        tic = time.time()
        print('\nepoch {} --------------------------'.format(epoch))
        loss_record = []
        gumble_list = []
        for step, input in enumerate(data_train):

            patient_list = []
            patient_list.append(len(input))
            for idx, adm in enumerate(input):
                if idx == 0:
                    pass
                else:
                    adm_list = []
                    seq_input = input[:idx + 1]
                    adm_list.append(seq_input)
                    train_dataset = mimic_data(adm_list)
                    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                                  collate_fn=pad_batch_v2_train, shuffle=True, pin_memory=True)

                    model.train()
                    for ind, data in enumerate(train_dataloader):
                        diseases, procedures, medications, seq_length, \
                            d_length_matrix, p_length_matrix, m_length_matrix, \
                            d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                            dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                            dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data
                        diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)

                        procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
                        dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)

                        stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
                        dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
                        stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
                        medications = medications.to(device)
                        m_mask_matrix = m_mask_matrix.to(device)
                        d_mask_matrix = d_mask_matrix.to(device)
                        p_mask_matrix = p_mask_matrix.to(device)
                        dec_disease_mask = dec_disease_mask.to(device)
                        stay_disease_mask = stay_disease_mask.to(device)
                        dec_proc_mask = dec_proc_mask.to(device)
                        stay_proc_mask = stay_proc_mask.to(device)
                        output_logits, count, gumbel_pick_index, cross_visit_scores_numpy = model(diseases, procedures,
                                                                                                  medications,
                                                                                                  d_mask_matrix,
                                                                                                  p_mask_matrix,
                                                                                                  m_mask_matrix,
                                                                                                  seq_length,
                                                                                                  dec_disease,
                                                                                                  stay_disease,
                                                                                                  dec_disease_mask,
                                                                                                  stay_disease_mask,
                                                                                                  dec_proc, stay_proc,
                                                                                                  dec_proc_mask,
                                                                                                  stay_proc_mask)
                        labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix,
                                                             voc_size[2] + 2, END_TOKEN, device, max_len=args.max_len)
                        patient_list.append(gumbel_pick_index)
                        loss = F.nll_loss(predictions, labels.long())
                        optimizer.zero_grad()
                        loss_record.append(loss.item())
                        loss.backward()
                        optimizer.step()
                        llprint('\rencoder_gumbel_training step: {} / {}'.format(step, len(data_train)))
            gumble_list.append(patient_list)

            # #### gumbel tau schedule ####
            if step % 100 == 0:
                # model.gumbel_tau = np.maximum(model.gumbel_tau * np.exp(-ANNEAL_RATE * step), temp_min)
                # model.att_tau = np.minimum(model.att_tau * np.exp(ANNEAL_RATE2 * step), temp_max)
                print(" New Gumbel Temperature: {}".format(model.gumbel_tau))
                print(" New Attention Temperature: {}".format(model.att_tau))

        # print("all_epoch_count: ",all_epoch_count)

        dill.dump(gumble_list,
                  open(os.path.join('saved', model_name, past_name, '{}epoch_train_gumbel_pick.pkl'.format(epoch)),
                       'wb'))  ## Save the indices of past visits similar to the current information selected by gumbel_softmax
        print()
        tic2 = time.time()
        ## Start of the eval function
        ddi_rate_list, ja_list, prauc_list, avg_p_list, avg_r_list, avg_f1_list, avg_med_list = [], [], [], [], [], [], []
        all_people = []
        for step, input in enumerate(data_eval):
            # step_l = []
            for idx, adm in enumerate(input):
                if idx == 0:
                    pass
                else:
                    adm_list = []
                    seq_input = input[:idx + 1]
                    adm_list.append(seq_input)

                    eval_dataset = mimic_data(adm_list)
                    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=pad_batch_v2_val, shuffle=True,
                                                 pin_memory=True)
                    ddi_rate, ddi_rate_positive, ja, prauc, avg_p, avg_r, avg_f1, avg_med, gumbel_pick_index, cross_visit_scores_numpy = eval(
                        model, eval_dataloader, voc_size, device, TOKENS, args)

                    print('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

                    history['ja'].append(ja)
                    history['ddi_rate'].append(ddi_rate)
                    history['avg_p'].append(avg_p)
                    history['avg_r'].append(avg_r)
                    history['avg_f1'].append(avg_f1)
                    history['prauc'].append(prauc)
                    history['med'].append(avg_med)
                    ddi_rate_list.append(ddi_rate)
                    ja_list.append(ja)
                    prauc_list.append(prauc)
                    avg_p_list.append(avg_p)
                    avg_r_list.append(avg_r)
                    avg_f1_list.append(avg_f1)
                    avg_med_list.append(avg_med)

        llprint('\n\rTrain--Epoch: %d, loss: %d' % (epoch, np.mean(loss_record)))

        """ Save the model for each epoch """
        torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, \
            'Epoch_{}_JA_{:.4}_DDI_{:.4}_LOSS_{}.model'.format(epoch, np.mean(ja_list), np.mean(ddi_rate_list), np.mean(loss_record))), 'wb'))
        logging.info(
            f'''Epoch {epoch:03d}, Jaccard: {np.mean(ja):.4}, DDI Rate: {ddi_rate:.4}, PRAUC: {np.mean(prauc):.4},DDI RATE POSITIVE :{ddi_rate_positive:.4},AVG_F1: {np.mean(avg_f1):.4}, AVG_PRC: {np.mean(avg_p):.4f}, AVG_RECALL: {np.mean(avg_r):.4f}, AVG_MED: {avg_med:.4}''')

        if best_ja < np.mean(ja_list):
            best_epoch = epoch
            best_ja = np.mean(ja_list)

        print('best_jaccard: {}'.format(best_ja))
        print('best_epoch: {}'.format(best_epoch))
        print('JA_{:.4}_DDI_{:.4}_PRAUC_{:.4}_F1_{:.4}'.format(np.mean(ja_list), np.mean(ddi_rate_list),
                                                               np.mean(prauc_list), np.mean(avg_f1_list)))

        dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))


if __name__ == '__main__':
    main(args)
    print(model_name)