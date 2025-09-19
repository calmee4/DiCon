#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import dill
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import BRICS
import matplotlib.pyplot as plt

##### process medications #####
# load med data
def med_process(med_file):
  if dataset == 'mimic-iii':
    med_pd = pd.read_csv(med_file, dtype={'NDC':'category'})

    # med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
    #                     'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
    #                     'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
    #                     'ROUTE','ENDDATE','DRUG'], axis=1, inplace=True)
    med_pd.drop(columns=['ROW_ID','DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
                        'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
                        'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
                        'ROUTE','ENDDATE'], axis=1, inplace=True)
    med_pd.drop(index = med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd
  if dataset == 'mimic-iv':
    med_pd = pd.read_csv(med_file, dtype={'ndc': 'category'})

    med_pd.drop(columns=['pharmacy_id', 'poe_id', 'poe_seq', 'order_provider_id',
                         'stoptime', 'formulary_drug_cd', 'gsn', 'drug_type',
                         'prod_strength', 'form_rx', 'dose_val_rx', 'dose_unit_rx',
                         'form_val_disp', 'form_unit_disp', 'doses_per_24_hrs', 'route'], axis=1, inplace=True)
    med_pd.drop(index=med_pd[med_pd['ndc'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['starttime'] = pd.to_datetime(med_pd['starttime'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['subject_id', 'hadm_id', 'starttime'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd

# ATC3-to-drugname
def ATC3toDrug(med_pd):
  if dataset =='mimic-iii':
    atc3toDrugDict = {}
    for atc3, drugname in med_pd[["ATC3", "DRUG"]].values:
        if atc3 in atc3toDrugDict:
            atc3toDrugDict[atc3].add(drugname)
        else:
            atc3toDrugDict[atc3] = set(drugname)

    return atc3toDrugDict
  if dataset =='mimic-iv':
      atc3toDrugDict = {}
      for atc3, drugname in med_pd[["ATC3", "drug"]].values:
          if atc3 in atc3toDrugDict:
              atc3toDrugDict[atc3].add(drugname)
          else:
              atc3toDrugDict[atc3] = set(drugname)

      return atc3toDrugDict
def get_atc3toSMILES(ATC3toDrugDict, druginfo):
    drug2smiles = {}
    atc3tosmiles = {}
    for drugname, smiles in druginfo[["name", "moldb_smiles"]].values:
        if type(smiles) == type("a"):
            drug2smiles[drugname] = smiles
    for atc3, drug in ATC3toDrugDict.items():
        temp = []
        for d in drug:
            try:
                temp.append(drug2smiles[d])
            except:
                pass
        if len(temp) > 0:
            atc3tosmiles[atc3] = temp[:3]

    return atc3tosmiles


# medication mapping
def codeMapping2atc4(med_pd):
  if dataset =='mimic-iii':
    with open(ndc2RXCUI_file, "r") as f:
        ndc2RXCUI = eval(f.read())
    med_pd["RXCUI"] = med_pd["NDC"].map(ndc2RXCUI)
    med_pd.dropna(inplace=True)

    RXCUI2atc4 = pd.read_csv(RXCUI2atc4_file)
    RXCUI2atc4 = RXCUI2atc4.drop(columns=["YEAR", "MONTH", "NDC"])
    RXCUI2atc4.drop_duplicates(subset=["RXCUI"], inplace=True)
    med_pd.drop(index=med_pd[med_pd["RXCUI"].isin([""])].index, axis=0, inplace=True)

    med_pd["RXCUI"] = med_pd["RXCUI"].astype("int64")
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(RXCUI2atc4, on=["RXCUI"])
    med_pd.drop(columns=["NDC", "RXCUI"], inplace=True)
    med_pd["ATC4"] = med_pd["ATC4"].map(lambda x: x[:4])
    med_pd = med_pd.rename(columns={"ATC4": "ATC3"})
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd
  if dataset =='mimic-iv':
      with open(ndc2RXCUI_file, "r") as f:
          ndc2RXCUI = eval(f.read())
      med_pd["RXCUI"] = med_pd["ndc"].map(ndc2RXCUI)
      med_pd.dropna(inplace=True)

      RXCUI2atc4 = pd.read_csv(RXCUI2atc4_file)
      RXCUI2atc4 = RXCUI2atc4.drop(columns=["YEAR", "MONTH", "NDC"])
      RXCUI2atc4.drop_duplicates(subset=["RXCUI"], inplace=True)
      med_pd.drop(index=med_pd[med_pd["RXCUI"].isin([""])].index, axis=0, inplace=True)

      med_pd["RXCUI"] = med_pd["RXCUI"].astype("int64")
      med_pd = med_pd.reset_index(drop=True)
      med_pd = med_pd.merge(RXCUI2atc4, on=["RXCUI"])
      med_pd.drop(columns=["ndc", "RXCUI"], inplace=True)
      med_pd["ATC4"] = med_pd["ATC4"].map(lambda x: x[:4])
      med_pd = med_pd.rename(columns={"ATC4": "ATC3"})
      med_pd = med_pd.drop_duplicates()
      med_pd = med_pd.reset_index(drop=True)
      return med_pd
# visit >= 2
def process_visit_lg2(med_pd):
  if dataset == 'mimic-iii':
    a = (
        med_pd[["SUBJECT_ID", "HADM_ID"]]
        .groupby(by="SUBJECT_ID")["HADM_ID"]
        .unique()
        .reset_index()
    )
    a["HADM_ID_Len"] = a["HADM_ID"].map(lambda x: len(x))
    a = a[a["HADM_ID_Len"] > 1]
    return a
  if dataset == 'mimic-iv':
    a = med_pd[['subject_id', 'hadm_id']].groupby(by='subject_id')['hadm_id'].unique().reset_index()
    a['hadm_id_Len'] = a['hadm_id'].map(lambda x: len(x))
    a = a[a['hadm_id_Len'] > 1]
    return a
# most common medications
def filter_300_most_med(med_pd):
    med_count = (
        med_pd.groupby(by=["ATC3"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
        .reset_index(drop=True)
    )
    med_pd = med_pd[med_pd["ATC3"].isin(med_count.loc[:299, "ATC3"])]

    return med_pd.reset_index(drop=True)


##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file, header=0,
                          names=[column.upper() for column in pd.read_csv(diag_file, nrows=1).columns])
    if "ICD9_CODE" in diag_pd.columns:
        diag_pd.rename(columns={'ICD9_CODE': 'ICD_CODE'}, inplace=True)
    diag_pd.dropna(inplace=True)
    diag_pd = diag_pd[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD_CODE']]
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=["SUBJECT_ID", "HADM_ID"], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    if dataset == "mimic-iv":
        # first, get the number of unique subject_id
        num_subject_id = len(diag_pd["SUBJECT_ID"].unique())
        # second, select the first 10% of the patients
        diag_pd = diag_pd[diag_pd["SUBJECT_ID"].isin(diag_pd["SUBJECT_ID"].unique()[: int(num_subject_id * 0.1)])]

    return diag_pd


##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={"ICD9_CODE": "category"}, header=0,
                         names=[column.upper() for column in pd.read_csv(procedure_file, nrows=1).columns])
    if "ICD9_CODE" in pro_pd.columns:
        pro_pd.rename(columns={'ICD9_CODE': 'ICD_CODE'}, inplace=True)
    pro_pd = pro_pd[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD_CODE']]
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], inplace=True)
    pro_pd.drop(columns=["SEQ_NUM"], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def filter_K_diag(diag_pd, K=20):
    # filter diagnosis with less than K occurrences
    # record length of diag_pd
    origin_len = len(diag_pd)

    diag_count = diag_pd.value_counts('ICD_CODE')
    diag_pd = diag_pd[diag_pd['ICD_CODE'].isin(diag_count[diag_count > K].index)]
    diag_pd = diag_pd.reset_index(drop=True)

    # record length of diag_pd
    new_len = len(diag_pd)
    print('filter diagnosis with less than {} occurrences: {} -> {}'.format(K, origin_len, new_len))
    filter_flag = (origin_len != new_len)
    return diag_pd, filter_flag


def filter_K_pro(pro_pd, K=20):
    # filter procedure with less than 20 occurrences
    # record length of pro_pd
    origin_len = len(pro_pd)

    pro_count = pro_pd.value_counts('ICD_CODE')
    pro_pd = pro_pd[pro_pd['ICD_CODE'].isin(pro_count[pro_count > K].index)]
    pro_pd = pro_pd.reset_index(drop=True)

    # record length of pro_pd
    new_len = len(pro_pd)
    print('filter procedure with less than {} occurrences: {} -> {}'.format(K, origin_len, new_len))
    filter_flag = (origin_len != new_len)
    return pro_pd, filter_flag


###### combine three tables #####
def combine_process(med_pd, diag_pd, pro_pd):
    # filter out the clinical codes with few occurrences, filter out patients with no clinical codes
  if dataset=='mimic-iii':
    filter_flag = True
    while filter_flag:
        med_pd_key = med_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
        diag_pd_key = diag_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
        pro_pd_key = pro_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()

        combined_key = med_pd_key.merge(
            diag_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner"
        )
        combined_key = combined_key.merge(
            pro_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner"
        )
        diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        diag_pd, filter_flag_diag = filter_K_diag(diag_pd)
        pro_pd, filter_flag_pro = filter_K_pro(pro_pd)
        filter_flag = filter_flag_diag or filter_flag_pro

    # flatten and merge
    diag_pd = (
        diag_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ICD_CODE"]
        .unique()
        .reset_index()
    )
    med_pd = med_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ATC3"].unique().reset_index()
    pro_pd = (
        pro_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ICD_CODE"]
        .unique()
        .reset_index()
        .rename(columns={"ICD_CODE": "PRO_CODE"})
    )
    med_pd["ATC3"] = med_pd["ATC3"].map(lambda x: list(x))
    pro_pd["PRO_CODE"] = pro_pd["PRO_CODE"].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    data = data.merge(pro_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    data["ATC3_num"] = data["ATC3"].map(lambda x: len(x))
    return data
  if dataset =='mimic-iv':
      print(med_pd.head())
      filter_flag = True
      med_pd.rename(columns={
          'subject_id': 'SUBJECT_ID',
          'hadm_id': 'HADM_ID'
      }, inplace=True)
      while filter_flag:
          med_pd_key = med_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
          diag_pd_key = diag_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
          pro_pd_key = pro_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
          combined_key = med_pd_key.merge(
              diag_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner"
          )
          combined_key = combined_key.merge(
              pro_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner"
          )
          diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
          med_pd = med_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how='inner')
          pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
          diag_pd, filter_flag_diag = filter_K_diag(diag_pd)
          pro_pd, filter_flag_pro = filter_K_pro(pro_pd)
          filter_flag = filter_flag_diag or filter_flag_pro

      # flatten and merge
      diag_pd = (
          diag_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ICD_CODE"]
          .unique()
          .reset_index()
      )
      med_pd = med_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ATC3"].unique().reset_index()
      pro_pd = (
          pro_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ICD_CODE"]
          .unique()
          .reset_index()
          .rename(columns={"ICD_CODE": "PRO_CODE"})
      )
      med_pd["ATC3"] = med_pd["ATC3"].map(lambda x: list(x))
      pro_pd["PRO_CODE"] = pro_pd["PRO_CODE"].map(lambda x: list(x))
      data = diag_pd.merge(med_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner")
      data = data.merge(pro_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner")
      data["ATC3_num"] = data["ATC3"].map(lambda x: len(x))
      return data
def statistics(data):
    print("#patients ", data["SUBJECT_ID"].unique().shape[0])
    print("#clinical events ", len(data))

    diag = data["ICD_CODE"].values
    med = data["ATC3"].values
    pro = data["PRO_CODE"].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])

    print("#diagnosis ", len(unique_diag))
    print("#med ", len(unique_med))
    print("#procedure", len(unique_pro))

    (
        avg_diag,
        avg_med,
        avg_pro,
        max_diag,
        max_med,
        max_pro,
        cnt,
        max_visit,
        avg_visit,
    ) = [0 for i in range(9)]

    for subject_id in data["SUBJECT_ID"].unique():
        item_data = data[data["SUBJECT_ID"] == subject_id]
        visit_cnt = 0
        for index, row in item_data.iterrows():
            x, y, z = [], [], []
            visit_cnt += 1
            cnt += 1
            x.extend(list(row["ICD_CODE"]))
            y.extend(list(row["ATC3"]))
            z.extend(list(row["PRO_CODE"]))
            avg_diag += len(x)
            avg_med += len(y)
            avg_pro += len(z)
            avg_visit += visit_cnt
            if len(x) > max_diag:
                max_diag = len(x)
            if len(y) > max_med:
                max_med = len(y)
            if len(z) > max_pro:
                max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print("#avg of diagnoses ", avg_diag / cnt)
    print("#avg of medicines ", avg_med / cnt)
    print("#avg of procedures ", avg_pro / cnt)
    print("#avg of visits ", avg_visit / len(data["SUBJECT_ID"].unique()))

    print("#max of diagnoses ", max_diag)
    print("#max of medicines ", max_med)
    print("#max of procedures ", max_pro)
    print("#max of visit ", max_visit)


##### indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


# create voc set
def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()

    for index, row in df.iterrows():
        diag_voc.add_sentence(row["ICD_CODE"])
        med_voc.add_sentence(row["ATC3"])
        pro_voc.add_sentence(row["PRO_CODE"])

    dill.dump(
        obj={"diag_voc": diag_voc, "med_voc": med_voc, "pro_voc": pro_voc},
        file=open(vocabulary_file, "wb"),
    )
    return diag_voc, med_voc, pro_voc


# 计算每个visit的权重，流行度越高，权重越低。
class PatientWeight():
    def __init__(self, data):
        # 统计ICD_CODE中每个元素出现的次数
        icd_counts_diag = data['ICD_CODE'].apply(pd.Series).stack().value_counts()
        weight_diag = 1 / icd_counts_diag

        # 计算归一化系数
        norm_effi = (icd_counts_diag * weight_diag).values.sum() / icd_counts_diag.values.sum()
        weight_diag = weight_diag / norm_effi

        # 创建一个新的数据框来存储统计结果
        self.weight_diag_df = pd.DataFrame({'Count': weight_diag.values}, index=weight_diag.index)

        # 统计PRO_CODE中每个元素出现的次数
        icd_counts_pro = data['PRO_CODE'].apply(pd.Series).stack().value_counts()
        weight_pro = 1 / icd_counts_pro

        # 计算归一化系数
        norm_effi = (icd_counts_pro * weight_pro).values.sum() / icd_counts_pro.values.sum()
        weight_pro = weight_pro / norm_effi

        # 创建一个新的数据框来存储统计结果
        self.weight_pro_df = pd.DataFrame({'Count': weight_pro.values}, index=weight_pro.index)

    def get(self, visit):
        weight_diag = self.weight_diag_df.loc[visit['ICD_CODE']].values
        # weight_pro = self.weight_pro_df.loc[visit['PRO_CODE']].values
        # get average weight
        # weight = np.concatenate((weight_diag, weight_pro), axis=0)
        # return np.mean(weight)
        return np.max(weight_diag)


# create final records
def create_patient_record(df, diag_voc, med_voc, pro_voc):
    get_weight = PatientWeight(data)
    records = []  # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    visit_weights = []
    for subject_id in df["SUBJECT_ID"].unique():
        item_df = df[df["SUBJECT_ID"] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row["ICD_CODE"]])
            admission.append([pro_voc.word2idx[i] for i in row["PRO_CODE"]])
            admission.append([med_voc.word2idx[i] for i in row["ATC3"]])
            visit_weight = get_weight.get(row)
            visit_weights.append(visit_weight)
            admission.append(visit_weight)
            patient.append(admission)
        records.append(patient)
    # 由于部分数据在构建因果图会卡死，因此在这里的records 提前去掉
    if dataset =='mimic-iv':
        print("len_records")
        print(len(records))
        print("after len_records")
        del(records[7040])
        del(records[1530])
        del(records[428])
        del(records[71])
        print(len(records))
    print("obtain ehr sequence data")

    dill.dump(obj=records, file=open(ehr_sequence_file, "wb"))
    return records, visit_weights


# get ddi matrix
def get_ddi_matrix(records, med_voc, ddi_file):
    TOPK = 40  # topk drug-drug interaction
    cid2atc_dic = defaultdict(set)
    med_voc_size = len(med_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)

    with open(cid2atc6_file, "r") as f:
        for line in f:
            line_ls = line[:-1].split(",")
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])

    # ddi load
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect
    ddi_most_pd = (
        ddi_df.groupby(by=["Polypharmacy Side Effect", "Side Effect Name"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
        .reset_index(drop=True)
    )
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:, :]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(
        ddi_most_pd[["Side Effect Name"]], how="inner", on=["Side Effect Name"]
    )
    ddi_df = (
        fliter_ddi_df[["STITCH 1", "STITCH 2"]].drop_duplicates().reset_index(drop=True)
    )

    # weighted ehr adj
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j <= i:
                        continue
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1
    dill.dump(ehr_adj, open(ehr_adjacency_file, "wb"))
    # ddi adj
    ddi_adj = np.zeros((med_voc_size, med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row["STITCH 1"]
        cid2 = row["STITCH 2"]

        # cid -> atc_level3
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:

                # atc_level3 -> atc_level4
                for i in atc3_atc4_dic[atc_i]:
                    for j in atc3_atc4_dic[atc_j]:
                        if med_voc.word2idx[i] != med_voc.word2idx[j]:
                            ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                            ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1
    dill.dump(ddi_adj, open(ddi_adjacency_file, "wb"))

    return ddi_adj


def cal_ddi_rate_score(records, ddi_adj):
    # ddi rate
    all_cnt = 0
    dd_cnt = 0
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_adj[med_i, med_j] == 1 or ddi_adj[med_j, med_i] == 1:
                        dd_cnt += 1
    ddi_rate_score = dd_cnt / all_cnt if all_cnt > 0 else 0
    print(f'{ddi_rate_score=}')
    return ddi_rate_score
import re

def get_positive_ddi(med_voc, ddi_adj):
    with open('descriptions.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    triplets = []
    efficacy_pattern = r"The therapeutic efficacy of (.+?) can be increased when used in combination with (.+?)\."
    for match in re.finditer(efficacy_pattern, text):
        drugA = match.group(1).strip()
        drugB = match.group(2).strip()
        sentence = match.group(0).strip()
        triplets.append((drugA, drugB, sentence))

    # 2. 提取降低副作用类三元组
    side_effect_pattern = r"The risk or severity of (.+?) can be decreased when (.+?) is combined with (.+?)\."
    for match in re.finditer(side_effect_pattern, text):
        drugA = match.group(2).strip()
        drugB = match.group(3).strip()
        sentence = match.group(0).strip()
        triplets.append((drugA, drugB, sentence))
    # 最终得到triplets的三元组
    # 通过当前的med_voc
    valid_atc3_set = set(med_voc.idx2word.values())  # 可用的 ATC3 编码集合

    # 构建 drug_name → set of ATC3
    drug_to_atc3 = defaultdict(set)
    for atc3, drugs in atc3toDrug.items():
        for drug in drugs:
            drug_to_atc3[drug.strip()].add(atc3)

    # 提取所有 ATC3 级别的相互作用对，且限定必须在 med_voc 允许的范围内
    atc3_interactions = set()

    for drug_a, drug_b, _ in triplets:
        atc3_a_set = drug_to_atc3.get(drug_a.strip())
        atc3_b_set = drug_to_atc3.get(drug_b.strip())

        if atc3_a_set and atc3_b_set:
            for atc3_a in atc3_a_set:
                for atc3_b in atc3_b_set:
                    # 新增的筛选条件：两个编码都必须出现在 med_voc
                    if atc3_a in valid_atc3_set and atc3_b in valid_atc3_set and atc3_a != atc3_b:
                        atc3_interactions.add(tuple(sorted([atc3_a, atc3_b])))
    print(f"共提取到 {len(atc3_interactions)} 对 ATC3 相互作用关系（限制在 med_voc 范围内）")

    # 1. 获取词汇表大小
    vocab_size = len(med_voc.word2idx)

    # 2. 初始化全零矩阵
    positive_ddi_adj = np.zeros((vocab_size, vocab_size))

    # 3. 遍历 ATC3 相互作用对，填充矩阵
    for atc3_a, atc3_b in atc3_interactions:
        idx_a = med_voc.word2idx[atc3_a]
        idx_b = med_voc.word2idx[atc3_b]
        positive_ddi_adj[idx_a, idx_b] = 1
        positive_ddi_adj[idx_b, idx_a] = 1  # 对称关系
    positive_ddi_adj[ddi_adj == 1] = 0
    return positive_ddi_adj


from collections import Counter
from itertools import combinations

def get_most_K_positive_ddi(ddi_adj_positive, med_voc, K=120):
    # 所有组合对的列表，保留重复项
    all_med_combinations = []

    for patient in records:
        for visit in patient:
            meds = visit[2]  # 第 2 项是用药列表（medication 索引）
            if len(meds) >= 2:
                # 获取该次就诊中的所有药物组合（两两）
                combs = combinations(meds, 2)
                all_med_combinations.extend(combs)

    # 统计每个药物组合对出现的次数
    combo_counter = Counter()

    for pair in all_med_combinations:
        # 为了确保 (a,b) 与 (b,a) 视为相同组合，先排序
        sorted_pair = tuple(sorted(pair))
        combo_counter[sorted_pair] += 1
    top_combos = combo_counter.most_common(len(combo_counter))  # 排个序
    # 找到ddi_adj_positive的作用对
    positive_pairs = []

    # 只遍历上三角，避免重复
    for i in range(ddi_adj_positive.shape[0]):
        for j in range(i + 1, ddi_adj_positive.shape[1]):
            if np.isclose(ddi_adj_positive[i][j], 1.0):
                positive_pairs.append(tuple(sorted((i, j))))
    top_combos_sorted = [(tuple(sorted(pair)), count) for pair, count in top_combos]
    positive_pair_set = set(positive_pairs)

    positive_top_combos = [(pair, count) for pair, count in top_combos_sorted if pair in positive_pair_set]
    top_k_pairs = [pair for pair, _ in positive_top_combos[K:]]
    vocab_size = len(med_voc.word2idx)
    # 初始化新矩阵
    positive_ddi_adj_K = np.zeros((vocab_size, vocab_size))
    for i, j in top_k_pairs:
        positive_ddi_adj_K[i][j] = 1
        positive_ddi_adj_K[j][i] = 1  # 保持对称

    return positive_ddi_adj_K
def get_ddi_mask(atc42SMLES, med_voc):
    # ATC3_List[22] = {0}
    # ATC3_List[25] = {0}
    # ATC3_List[27] = {0}
    fraction = []
    for k, v in med_voc.idx2word.items():
        tempF = set()
        for SMILES in atc42SMLES[v]:
            try:
                m = BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES))
                for frac in m:
                    tempF.add(frac)
            except:
                pass
        fraction.append(tempF)
    fracSet = []
    for i in fraction:
        fracSet += i
    fracSet = list(set(fracSet))  # set of all segments

    ddi_matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))
    for i, fracList in enumerate(fraction):
        for frac in fracList:
            ddi_matrix[i, fracSet.index(frac)] = 1
    return ddi_matrix, fracSet

for dataset in ['mimic-iii']:

    # for dataset in ['mimic-iv']:
    print("-" * 10, "processing dataset: ", dataset, "-" * 10)
    # files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
    # please change into your own MIMIC folder
    if dataset == 'mimic-iii':
        med_file = "./input/" + dataset + "/PRESCRIPTIONS.csv"
        diag_file = "./input/" + dataset + "/DIAGNOSES_ICD.csv"
        procedure_file = "./input/" + dataset + "/PROCEDURES_ICD.csv"
    elif dataset == 'mimic-iv':
        med_file = "./input/" + dataset + "/prescriptions.csv"
        diag_file = "./input/" + dataset + "/diagnoses_icd.csv"
        procedure_file = "./input/" + dataset + "/procedures_icd.csv"

    # input auxiliary files
    RXCUI2atc4_file = "./input/RXCUI2atc4.csv"
    cid2atc6_file = "./input/drug-atc.csv"
    ndc2RXCUI_file = "./input/ndc2RXCUI.txt"
    ddi_file = "./input/drug-DDI.csv"
    drugbankinfo = "./input/drugbank_drugs_info.csv"

    # output files
    output_dir = './output/' + dataset + '/'
    os.makedirs(output_dir, exist_ok=True)

    ddi_adjacency_file = output_dir + "/ddi_A_final.pkl"
    ddi_adjacency_file_positive = output_dir + "/ddi_B_final.pkl"
    ehr_adjacency_file = output_dir + "/ehr_adj_final.pkl"
    ehr_sequence_file = output_dir + "/records_final.pkl"
    vocabulary_file = output_dir + "/voc_final.pkl"
    ddi_mask_H_file = output_dir + "/ddi_mask_H.pkl"
    atc3toSMILES_file = output_dir + "/atc3toSMILES.pkl"
    substructure_smiles_file = output_dir + "/substructure_smiles.pkl"

    # for med
    med_pd = med_process(med_file)
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    if dataset == 'mimic-iii':
       med_pd = med_pd.merge(
        med_pd_lg2[["SUBJECT_ID"]], on="SUBJECT_ID", how="inner"
    ).reset_index(drop=True)
    if dataset == 'mimic-iv':
       med_pd = med_pd.merge(med_pd_lg2[['subject_id']], on='subject_id', how='inner').reset_index(drop=True)

    med_pd = codeMapping2atc4(med_pd)  # 损失大量药物编码
    med_pd = filter_300_most_med(med_pd)

    # med to SMILES mapping
    atc3toDrug = ATC3toDrug(med_pd)
    druginfo = pd.read_csv(drugbankinfo, dtype={"synthesis_patent_id": "category"})
    atc3toSMILES = get_atc3toSMILES(atc3toDrug, druginfo)
    dill.dump(atc3toSMILES, open(atc3toSMILES_file, "wb"))
    med_pd = med_pd[med_pd.ATC3.isin(atc3toSMILES.keys())]
    print("complete medication processing")

    # for diagnosis
    diag_pd = diag_process(diag_file)
    print("complete diagnosis processing")

    # for procedure
    pro_pd = procedure_process(procedure_file)
    print("complete procedure processing")

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd)
    statistics(data)
    print("complete combining")

    # create vocab
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    print("obtain voc")

    # create ehr sequence data
    records, visit_weights = create_patient_record(data, diag_voc, med_voc, pro_voc)

    # create ddi adj matrix
    ddi_adj = get_ddi_matrix(records, med_voc, ddi_file)
    print("obtain ddi adj matrix")
    ddi_adj_positive = get_positive_ddi(med_voc, ddi_adj)
    ddi_adj_positive = get_most_K_positive_ddi(ddi_adj_positive,med_voc)
    dill.dump(ddi_adj_positive, open(ddi_adjacency_file_positive, "wb"))
    print("obtain ddi adj positive")
    print(ddi_adj_positive.sum())
    # calculate ddi rate in EHR
    cal_ddi_rate_score(records, ddi_adj)
    cal_ddi_rate_score(records, ddi_adj_positive)
    # get ddi_mask_H
    ddi_mask_H, fracSet = get_ddi_mask(atc3toSMILES, med_voc)
    dill.dump(ddi_mask_H, open(ddi_mask_H_file, "wb"))
    dill.dump(fracSet, open(substructure_smiles_file, 'wb'))

    record = dill.load(open(f'{output_dir}/records_final.pkl', 'rb'))
    voc = dill.load(open(f'{output_dir}/voc_final.pkl', 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    )
    matrix1 = np.zeros([voc_size[0], voc_size[2]])
    matrix2 = np.zeros([voc_size[1], voc_size[2]])

    diag_count = np.zeros(voc_size[0])
    proc_count = np.zeros(voc_size[1])

    for patient in record:
        for adm in patient:
            for med in adm[2]:
                for diag in adm[0]:
                    matrix1[diag][med] += 1
                for proc in adm[1]:
                    matrix2[proc][med] += 1
            for diag in adm[0]:
                diag_count[diag] += 1
            for proc in adm[1]:
                proc_count[proc] += 1

    for i in range(matrix1.shape[0]):
        matrix1[i, :] /= diag_count[i]

    for i in range(matrix2.shape[0]):
        matrix2[i, :] /= proc_count[i]

    effect_df1 = pd.DataFrame(0.0, index=[f"Diag_{i}" for i in range(voc_size[0])],
                              columns=[f"Med_{j}" for j in range(voc_size[2])])
    effect_df1.iloc[:, :] = matrix1

    effect_df2 = pd.DataFrame(0.0, index=[f"Proc_{i}" for i in range(voc_size[1])],
                              columns=[f"Med_{j}" for j in range(voc_size[2])])
    effect_df2.iloc[:, :] = matrix2

    dill.dump(effect_df1, open(f'./output/{dataset}/graphs/Diag_Med_relevance.pkl', 'wb'))
    dill.dump(effect_df2, open(f'./output/{dataset}/graphs/Proc_Med_relevance.pkl', 'wb'))
    print(f"{dataset}'s grpahs over")


