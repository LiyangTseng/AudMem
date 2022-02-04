import os
import csv
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

VIGILANCE_THRESHOLD = 0.65

TRACK_LIST = ['normalize_5s_intro_thc1MtNagC8.wav', 'normalize_5s_intro_Wo2qUD1g7xM.wav', 'normalize_5s_intro_3ObVN3QQiZ8.wav', 'normalize_5s_intro_S-zQJFRX5Fg.wav', 'normalize_5s_intro_SyZOAgXiPMw.wav', 'normalize_5s_intro_GQT8ejgV2_A.wav', 'normalize_5s_intro_PQAIxeSIQU4.wav', 'normalize_5s_intro_E-8pyVBvCPQ.wav', 'normalize_5s_intro_Qr8eZSVaw10.wav', 'normalize_5s_intro_p7j-tz1Cn4o.wav', 'normalize_5s_intro_nISI4qF55F4.wav', 'normalize_5s_intro_RoeRU5zxkak.wav', 'normalize_5s_intro_EygNk739nnY.wav', 'normalize_5s_intro_w1G3rqVil1s.wav', 'normalize_5s_intro_KKc_RMln5UY.wav', 'normalize_5s_intro_Ng2JdroNfC0.wav', 'normalize_5s_intro_xc0sWhVhmkw.wav', 'normalize_5s_intro_VVRszjvg3_U.wav', 'normalize_5s_intro_C7u6rtswjCU.wav', 'normalize_5s_intro_HiPkwl5p1GY.wav', 'normalize_5s_intro_mYa_9d2Daas.wav', 'normalize_5s_intro_6MSYrN4YfKY.wav', 'normalize_5s_intro_O2q_9lBDM7I.wav', 'normalize_5s_intro_7E_a_VKjcl8.wav', 'normalize_5s_intro_a8cJLohQ_Jg.wav', 'normalize_5s_intro_7zz-nEVKZdc.wav', 'normalize_5s_intro_JeGhUESd_1o.wav', 'normalize_5s_intro_IN1f9k8qVDk.wav', 'normalize_5s_intro_RhBb77hG0iw.wav', 'normalize_5s_intro_qAiwzv8N7rM.wav', 'normalize_5s_intro_AoB8koE95C0.wav', 'normalize_5s_intro_j3DigipQ_hQ.wav', 'normalize_5s_intro_1X0SdKtnwo8.wav', 'normalize_5s_intro_RCJx5VW-fQI.wav', 'normalize_5s_intro_S_-qkv0NZ1g.wav', 'normalize_5s_intro_C90sY_Ht6Ig.wav', 'normalize_5s_intro_Z5gvqq3ChII.wav', 'normalize_5s_intro_zumMQrI_tMg.wav', 'normalize_5s_intro_gwsaElRJI2M.wav', 'normalize_5s_intro_ftjEcrrf7r0.wav', 'normalize_5s_intro_ZBBS4imv1qo.wav', 'normalize_5s_intro_DyQ_9p6y89c.wav', 'normalize_5s_intro_vgZv7Uu4YrA.wav', 'normalize_5s_intro_wcLXjQLwSBE.wav', 'normalize_5s_intro_7LuQQP-DAoc.wav', 'normalize_5s_intro_BEo0rqOZIng.wav', 'normalize_5s_intro_n4HTXYR-2AI.wav', 'normalize_5s_intro_72T4j04MS8o.wav', 'normalize_5s_intro_6TT_UgrRHq8.wav', 'normalize_5s_intro_uo8qDCDZhK0.wav', 'normalize_5s_intro_Et-YdmSo_3A.wav', 'normalize_5s_intro_oxKbrl4kyg8.wav', 'normalize_5s_intro_XgwqnGG-pbI.wav', 'normalize_5s_intro_1wpJkzCWHcI.wav', 'normalize_5s_intro_bwQ49N0jVvE.wav', 'normalize_5s_intro_OMR2W-7AyYU.wav', 'normalize_5s_intro_sjlkxcwhpwA.wav', 'normalize_5s_intro_4F1wvsJXXVY.wav', 'normalize_5s_intro_YEq-cvq_cK4.wav', 'normalize_5s_intro_42O51bcJyq0.wav', 'normalize_5s_intro_5FYAICvv-d0.wav', 'normalize_5s_intro_yBzk2xXE9yg.wav', 'normalize_5s_intro_zEWSSod0zTY.wav', 'normalize_5s_intro_dvf--10EYXw.wav', 'normalize_5s_intro_xQOXxmznGPg.wav', 'normalize_5s_intro_hMWoOunsMFM.wav', 'normalize_5s_intro_TnsOVDCq_b0.wav', 'normalize_5s_intro_Yh78Ll6-ODQ.wav', 'normalize_5s_intro_IYnu4-69fTA.wav', 'normalize_5s_intro_SubIr_Fyp4M.wav', 'normalize_5s_intro_WrRAZVJGImw.wav', 'normalize_5s_intro_gFnNr5vr5bQ.wav', 'normalize_5s_intro_j9KKh215HTs.wav', 'normalize_5s_intro_XBTT9tSVsh0.wav', 'normalize_5s_intro_u8BxVzRG9bE.wav', 'normalize_5s_intro_SQBuVfTX1ME.wav', 'normalize_5s_intro_-MqZKMbOYEA.wav', 'normalize_5s_intro_IpniN1Wq68Y.wav', 'normalize_5s_intro_1lunUbvf35M.wav', 'normalize_5s_intro_zk04E79riMQ.wav', 'normalize_5s_intro_uUfPwlxFFJM.wav', 'normalize_5s_intro_Ws-QlpSltr8.wav', 'normalize_5s_intro_xT1eOeXlTXg.wav', 'normalize_5s_intro_1Ngn3fZIK2E.wav', 'normalize_5s_intro_2JL_KcEzkqg.wav', 'normalize_5s_intro_4jvQFLlRQlo.wav', 'normalize_5s_intro_AjGkbFqi67c.wav', 'normalize_5s_intro_ahpmuikko3U.wav', 'normalize_5s_intro_sY5wXfgspQI.wav', 'normalize_5s_intro_HMvXE4Zs6ZA.wav', 'normalize_5s_intro_gv7BRXvZJbI.wav', 'normalize_5s_intro_4wgo8K28RNM.wav', 'normalize_5s_intro_2ySLmwsfP4Q.wav', 'normalize_5s_intro_MY4YJxn-9Og.wav', 'normalize_5s_intro_3gjfHYZ873o.wav', 'normalize_5s_intro_csHiDQXIggE.wav', 'normalize_5s_intro_C5VCGM2J5ls.wav', 'normalize_5s_intro_ey4Fc9DP5Rw.wav', 'normalize_5s_intro_bI7xde9-3BI.wav', 'normalize_5s_intro_EfZ-dVDySzc.wav', 'normalize_5s_intro_Zh3uBgwow8A.wav', 'normalize_5s_intro_JQTlG7NxJek.wav', 'normalize_5s_intro_1CrxzClzLvs.wav', 'normalize_5s_intro_0aC-jOKuBFE.wav', 'normalize_5s_intro_xePw8n4xu8o.wav', 'normalize_5s_intro_lEHM9HZf0IA.wav', 'normalize_5s_intro_xhmtXrtLkgo.wav', 'normalize_5s_intro_hHItMz0gfaU.wav', 'normalize_5s_intro_99f0oH45TVc.wav', 'normalize_5s_intro_co6WMzDOh1o.wav', 'normalize_5s_intro_xqzOxMdhmzU.wav', 'normalize_5s_intro_h-nnAeByB1A.wav', 'normalize_5s_intro_TFv9Kcym9dg.wav', 'normalize_5s_intro_tEW2eRQ-4DY.wav', 'normalize_5s_intro_VAc0xuVa7jI.wav', 'normalize_5s_intro_PALMMqZLAQk.wav', 'normalize_5s_intro_STpRa2JPFA0.wav', 'normalize_5s_intro_SgJMnEdtTXA.wav', 'normalize_5s_intro_NL2ZHPji3Z0.wav', 'normalize_5s_intro_EVSuxb6Ywcg.wav', 'normalize_5s_intro_wAJMhJpSCIc.wav', 'normalize_5s_intro_GphIn74Weu0.wav', 'normalize_5s_intro_gue_crpFdSE.wav', 'normalize_5s_intro_oQ0O2cd1T04.wav', 'normalize_5s_intro_vMcFA2x23FE.wav', 'normalize_5s_intro_FhvXg70ycrM.wav', 'normalize_5s_intro_lE_747E_Sdg.wav', 'normalize_5s_intro_i0MrGb1hT2U.wav', 'normalize_5s_intro_bI8-2blisUM.wav', 'normalize_5s_intro_aQ06TfyA1Ks.wav', 'normalize_5s_intro_ZvrysfBDzSs.wav', 'normalize_5s_intro_v2seHL0pwbg.wav', 'normalize_5s_intro_BrrWNfjgHGs.wav', 'normalize_5s_intro_j1c70vRHdhQ.wav', 'normalize_5s_intro_3DCHLwOqtJs.wav', 'normalize_5s_intro_g20t_K9dlhU.wav', 'normalize_5s_intro_EH1OEWJ9C5w.wav', 'normalize_5s_intro_SCBxmcwmX7U.wav', 'normalize_5s_intro_tXvpe2GbUec.wav', 'normalize_5s_intro_7ZgPGMfUVek.wav', 'normalize_5s_intro_aIJuCcGFJkc.wav', 'normalize_5s_intro_RLMl1umHgp0.wav', 'normalize_5s_intro_KT-m6qTJyN0.wav', 'normalize_5s_intro_WJs-_T8I74Y.wav', 'normalize_5s_intro_aIyqRdrHodE.wav', 'normalize_5s_intro_XJT-fM4nBJU.wav', 'normalize_5s_intro_7QQzDQceGgU.wav', 'normalize_5s_intro_fE2h3lGlOsk.wav', 'normalize_5s_intro_Oq1n8fUxQZc.wav', 'normalize_5s_intro_pssWSj42t8M.wav', 'normalize_5s_intro_GsPq9mzFNGY.wav', 'normalize_5s_intro_Jg9NbDizoPM.wav', 'normalize_5s_intro_Ib7m3Qh-4O4.wav', 'normalize_5s_intro_hn3wJ1_1Zsg.wav', 'normalize_5s_intro_hjZqVw3qI9E.wav', 'normalize_5s_intro_cUKD9tEeBp0.wav', 'normalize_5s_intro_q_4no3KCrY4.wav', 'normalize_5s_intro_VlWs8ey2nyg.wav', 'normalize_5s_intro_Srp0opA8V8o.wav', 'normalize_5s_intro_PYM9NUU9Roc.wav', 'normalize_5s_intro_v0UvOsCi8mc.wav', 'normalize_5s_intro_zaCbuB3w0kg.wav', 'normalize_5s_intro_PCp2iXA1uLE.wav', 'normalize_5s_intro_S2RnxiNJg0M.wav', 'normalize_5s_intro_Jtv4satRsP0.wav', 'normalize_5s_intro_ytq5pGcM77w.wav', 'normalize_5s_intro_9nWpMZFrbvI.wav', 'normalize_5s_intro_1kN-34GFMYM.wav', 'normalize_5s_intro_Yyvo9O8fN-A.wav', 'normalize_5s_intro_ulj-L3K_Gzs.wav', 'normalize_5s_intro_V-ar6MLjy5o.wav', 'normalize_5s_intro_dtOv6WvJ44w.wav', 'normalize_5s_intro_XkC8Uzl9pCY.wav', 'normalize_5s_intro_jII5qoCrzYE.wav', 'normalize_5s_intro_7pcZIsJNlAs.wav', 'normalize_5s_intro_0QN9KLFWn7I.wav', 'normalize_5s_intro_d6BzCEkGd3I.wav', 'normalize_5s_intro_lYxcW8jtFw0.wav', 'normalize_5s_intro_R1T_SrdQGH8.wav', 'normalize_5s_intro_YOKq1VmEbtc.wav', 'normalize_5s_intro_19Q9l85Feqw.wav', 'normalize_5s_intro_CXm7hPs_als.wav', 'normalize_5s_intro_nFOLhtsyvMA.wav', 'normalize_5s_intro_-8cFfkyk7vA.wav', 'normalize_5s_intro_ZIiQ1jMqhVM.wav', 'normalize_5s_intro_hejXc_FSYb8.wav', 'normalize_5s_intro_eXvBjCO19QY.wav', 'normalize_5s_intro_haCay85cpvo.wav', 'normalize_5s_intro_RpJz01guPMY.wav', 'normalize_5s_intro_sPlXrbVLdO8.wav', 'normalize_5s_intro_Mme9REVuidw.wav', 'normalize_5s_intro_UGTYqTKUl8w.wav', 'normalize_5s_intro_9DP0yMwvyWE.wav', 'normalize_5s_intro_WrDJMxSKlCA.wav', 'normalize_5s_intro_2F8Kr91wQ0U.wav', 'normalize_5s_intro_gyegm85BPPA.wav', 'normalize_5s_intro_Xhh3_-JRnDc.wav', 'normalize_5s_intro_WRSeV_27z6k.wav', 'normalize_5s_intro_HwcCBnfhsR4.wav', 'normalize_5s_intro_bd5m12UEHWI.wav', 'normalize_5s_intro_1juIFmPyG-Y.wav', 'normalize_5s_intro_DGsoqhIUgDQ.wav', 'normalize_5s_intro_2UL-1MOlSPw.wav', 'normalize_5s_intro_2AWE9tqnDPw.wav', 'normalize_5s_intro_68b_HImZAig.wav', 'normalize_5s_intro_GIulOhzXufc.wav', 'normalize_5s_intro_Stet_4bnclk.wav', 'normalize_5s_intro_RHGfkuJv0j0.wav', 'normalize_5s_intro_0uLI6BnVh6w.wav', 'normalize_5s_intro_uo6VU4euIbY.wav', 'normalize_5s_intro_6pARjpdqxYQ.wav', 'normalize_5s_intro_hjIhCG_nIPA.wav', 'normalize_5s_intro_hV-FwW1LgxU.wav', 'normalize_5s_intro_mWfWyhzC22U.wav', 'normalize_5s_intro_IISA6t-9zzc.wav', 'normalize_5s_intro_gDevCxVY_wA.wav', 'normalize_5s_intro_IrtcCSE2bVY.wav', 'normalize_5s_intro_feVUoKhP1mE.wav', 'normalize_5s_intro_Tfypj4UwvvA.wav', 'normalize_5s_intro_TeH7sCVCMJk.wav', 'normalize_5s_intro_0EVVKs6DQLo.wav', 'normalize_5s_intro_d7to9URtLZ4.wav', 'normalize_5s_intro_TzhhbYS9EO4.wav', 'normalize_5s_intro_nn5nypm7GG8.wav', 'normalize_5s_intro_hed6HkYNA7g.wav', 'normalize_5s_intro_rWznOAwxM1g.wav', 'normalize_5s_intro_zyQkFh-E4Ak.wav', 'normalize_5s_intro_agKkcRXN2iE.wav', 'normalize_5s_intro_SZaZU_qi6Xc.wav', 'normalize_5s_intro_ZpDQJnI4OhU.wav', 'normalize_5s_intro_D4nWzd63jV4.wav', 'normalize_5s_intro_9odM1BRqop4.wav', 'normalize_5s_intro_F64yFFnZfkI.wav', 'normalize_5s_intro_Js2JQH_kt0I.wav', 'normalize_5s_intro_Skt_NKI4d6U.wav']
TRACK_NUM = len(TRACK_LIST)
id_to_track = {idx: TRACK_LIST[idx] for idx in range(TRACK_NUM)}

# idx of audio
vigilance_idx_arr = [0, 2, 15, 31, 54, 71, 84, 101, 110, 125, 137, 150, 157, 170, 178, 184, 188, 198, 216, 221, 232]
# idx of audio
target_idx_arr = [1, 4, 5, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 29, 30, 32, 33, 35, 36, 
38, 40, 41, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 72, 74, 75, 76, 77, 
78, 80, 82, 83, 85, 86, 87, 89, 90, 91, 93, 94, 95, 98, 99, 100, 102, 104, 105, 107, 108, 111, 112, 113, 116, 118, 120, 
121, 122, 127, 128, 129, 131, 134, 135, 136, 138, 141, 142, 143, 144, 145, 147, 148, 149, 151, 152, 153, 154, 155, 156, 
158, 160, 161, 162, 164, 165, 166, 168, 171, 173, 174, 176, 177, 179, 180, 181, 182, 187, 190, 193, 195, 196, 200, 201, 
204, 207, 209, 211, 212, 213, 214, 215, 217, 218, 219, 220, 222, 223, 224, 225, 226, 227]

slotOrder = [0, 1, 2, 3, 4, 0, 5, 6, 2, 7, 8, 9, 1, 10, 11, 12, 13, 4, 14, 15, 16, 5, 17, 18, 19, 20, 
  15, 21, 7, 22, 23, 24, 25, 26, 27, 28, 29, 12, 30, 31, 32, 20, 33, 16, 34, 31, 14, 35, 36, 37, 27, 38, 39, 
  25, 40, 41, 42, 43, 44, 21, 45, 46, 32, 47, 48, 49, 30, 50, 51, 52, 53, 54, 35, 55, 56, 57, 58, 59, 60, 54, 
  61, 62, 63, 64, 65, 36, 66, 67, 68, 69, 70, 38, 71, 43, 72, 73, 74, 75, 40, 76, 71, 77, 53, 78, 72, 79, 51, 
  80, 81, 65, 82, 58, 47, 83, 84, 63, 85, 86, 87, 88, 84, 89, 90, 56, 91, 92, 93, 64, 94, 95, 68, 96, 97, 98, 
  74, 99, 6, 23, 100, 11, 101, 8, 102, 103, 104, 59, 101, 13, 22, 46, 100, 29, 105, 33, 48, 99, 106, 107, 45, 
  108, 109, 110, 105, 111, 61, 112, 102, 113, 114, 115, 110, 116, 117, 118, 119, 120, 121, 122, 86, 123, 124, 
  125, 108, 126, 127, 104, 128, 129, 125, 130, 121, 131, 132, 89, 107, 112, 93, 118, 133, 134, 75, 135, 136, 
  137, 138, 77, 139, 140, 82, 129, 141, 67, 142, 137, 135, 143, 144, 95, 145, 146, 147, 148, 149, 120, 150, 
  151, 152, 131, 153, 154, 134, 155, 150, 136, 156, 157, 158, 159, 160, 145, 138, 161, 141, 162, 157, 163, 
  164, 165, 166, 156, 167, 168, 169, 170, 160, 171, 164, 144, 172, 170, 153, 173, 161, 147, 174, 175, 151, 
  176, 149, 166, 116, 177, 178, 179, 113, 180, 17, 90, 178, 111, 181, 18, 127, 177, 76, 26, 128, 50, 182, 
  44, 142, 66, 179, 122, 41, 148, 183, 78, 83, 143, 184, 98, 185, 180, 162, 60, 184, 87, 186, 152, 187, 85, 
  188, 52, 165, 189, 182, 190, 188, 191, 154, 181, 192, 80, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 
  168, 203, 198, 204, 155, 205, 206, 193, 207, 208, 209, 210, 211, 212, 213, 196, 214, 215, 216, 62, 187, 158, 
  209, 217, 218, 216, 212, 190, 207, 219, 220, 221, 200, 204, 222, 223, 195, 221, 224, 225, 226, 227, 228, 
  229, 230, 214, 201, 215, 173, 211, 222, 171, 231, 224, 218, 232, 217, 174, 213, 223, 232, 219, 220, 233, 176, 225, 91, 226, 227, 234, 94]

vigilance_first_appearnce_idx = []
vigilance_last_appearnce_idx = []
for index in range(len(vigilance_idx_arr)):
    vigilance_idx = vigilance_idx_arr[index]
    # ref: https://stackoverflow.com/questions/522372/finding-first-and-last-index-of-some-value-in-a-list-in-python
    first_occ = slotOrder.index(vigilance_idx)
    last_occ = len(slotOrder)-1-slotOrder[::-1].index(vigilance_idx)
    vigilance_first_appearnce_idx.append(first_occ)
    vigilance_last_appearnce_idx.append(last_occ)

target_first_appearnce_idx = []
target_last_appearnce_idx = []
for index in range(len(target_idx_arr)):
    target_idx = target_idx_arr[index]
    # ref: https://stackoverflow.com/questions/522372/finding-first-and-last-index-of-some-value-in-a-list-in-python
    first_occ = slotOrder.index(target_idx)
    last_occ = len(slotOrder)-1-slotOrder[::-1].index(target_idx)
    target_first_appearnce_idx.append(first_occ)
    target_last_appearnce_idx.append(last_occ)
# sort both list according last appearance index
target_last_appearnce_idx, target_first_appearnce_idx = zip(*sorted(zip(
                                                    target_last_appearnce_idx, target_first_appearnce_idx)))


def filter_data(data_df):
    ''' prune repeated participations and incomplete fields '''
    
    # drop rows without completing experiment
    # TODO: should consider unfinished but qualified data
    data_df = data_df.drop(data_df[data_df['experimentFinished']==0].index)

    # TODO: should able to prune by vigilance on new data format
    # calculate vigilance score for all users, then filter unqualified
    for idx, row in data_df.iterrows():
        _, vigilance_score, _, _, _ = calculate_datum_stats(row, id_to_track)
        data_df.loc[idx,"vigilanceScore"] = vigilance_score
    data_df = data_df.drop(data_df[data_df['vigilanceScore']<VIGILANCE_THRESHOLD].index)
    
    # drop rows with NaN column
    data_df = data_df.dropna()
    # only keep the very first userEmail
    data_df = data_df.drop_duplicates(subset='userEmail', keep='first')

    return data_df

def calculate_datum_stats(singleUserData, id_to_track):
    ''' check if targets are memorized, return track_memorability_dict to update '''
    
    track_to_memo = {track: [] for idx, track in id_to_track.items()}
    
    # convert to list of int, each element representing track index
    audioOrder = list(map(int, (singleUserData.loc['audioOrder']).split(','))) 
    if not isinstance(singleUserData.loc['userResponse'], str): 
        # user did not response
        return None, 0, None, None, None
    userResponses = list(map(int, (singleUserData.loc['userResponse']).split(','))) 

    vigilance_progress_cnt, target_progress_cnt = 0, 0
    vigilance_performance, target_performance = [], []

    for response_idx in range(len(userResponses)):
        if vigilance_progress_cnt < len(vigilance_last_appearnce_idx) and \
            response_idx == vigilance_last_appearnce_idx[vigilance_progress_cnt]:
            
            # encounter one vigilance pair
            first_occ = vigilance_first_appearnce_idx[vigilance_progress_cnt]
            last_occ = vigilance_last_appearnce_idx[vigilance_progress_cnt]
            if userResponses[first_occ] == 0 and userResponses[last_occ] == 1:
                vigilance_performance.append(1) # vigilance memorized
            elif userResponses[first_occ] == 0 and userResponses[last_occ] == 0:
                vigilance_performance.append(0) # vigilance not memorized
            vigilance_progress_cnt += 1

        if response_idx == target_last_appearnce_idx[target_progress_cnt]:
            # encounter one target pair
            first_occ = target_first_appearnce_idx[target_progress_cnt]
            last_occ = target_last_appearnce_idx[target_progress_cnt]
            track_idx = audioOrder[slotOrder[first_occ]]
            track_name = id_to_track[track_idx]
            if userResponses[first_occ] == 0 and userResponses[last_occ] == 1:
                target_performance.append(1) # target memorized
                track_to_memo[track_name].append({last_occ-first_occ: 1})
            elif userResponses[first_occ] == 0 and userResponses[last_occ] == 0:
                target_performance.append(0) # target not memorized
                track_to_memo[track_name].append({last_occ-first_occ: 0})
            target_progress_cnt += 1

    vigilance_score = 0 if len(vigilance_performance)==0 \
                        else float(vigilance_performance.count(1))/len(vigilance_performance)
    vigilance_detail = "{}/{}".format(vigilance_performance.count(1), len(vigilance_performance))
    
    target_score = 0 if len(target_performance)==0 \
                        else float(target_performance.count(1))/len(target_performance)
    target_detail = "{}/{}".format(target_performance.count(1), len(target_performance))
    
    return track_to_memo, vigilance_score, vigilance_detail, target_score, target_detail

def get_experiment_stats(qualified_data_df):
    ''' returns:
        (1) stats by user (user_stats, <dataframe>) 
        (2) stats by track (track_stats, <dict>) 
    '''

    id_to_track = {idx: TRACK_LIST[idx] for idx in range(TRACK_NUM)}
    track_stats = {track: [] for idx, track in id_to_track.items()}

    user_stats = qualified_data_df[["userEmail"]].copy()
    user_stats["vigilance_score"] = 0
    user_stats["vigilance_stats"] = 0
    user_stats["target_score"] = 0
    user_stats["target_stats"] = 0
    for index, singleUserData in qualified_data_df.iterrows():
        track_to_memo, vigilance_score, vigilance_stats, target_score, target_stats = calculate_datum_stats(singleUserData, id_to_track)
        if vigilance_score < VIGILANCE_THRESHOLD:
            continue
        for track, memorability in track_to_memo.items():
            if not memorability == []:
                track_stats[track].extend(memorability)
        
        user_stats.loc[index, "vigilance_score"] = vigilance_score
        user_stats.loc[index, "target_score"] = target_score
        user_stats.loc[index, "vigilance_stats"] = vigilance_stats
        user_stats.loc[index, "target_stats"] = target_stats
    return user_stats, track_stats

def save_stats(args, user_stats, track_stats, track_order):
    ''' save statistics by participants and tracks ''' 

    # save user results (vigilance and target accuracy)
    os.makedirs(args.stats_dir, exist_ok=True)
    results_file = os.path.join(args.stats_dir, 'user_stats_beta.csv')
    user_stats.to_csv(results_file, index=False)
    print('user results saved at {}'.format(results_file))
    
    # save detailed track memorability results
    # the format of would be "track", "repeat_interval:memorized" 
    detailed_label_file = os.path.join(args.stats_dir, 'track_memorability_details_beta.csv')
    with open(detailed_label_file, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(['track', 'repeat_interval:memorized'])
        for track in track_order:
            writer.writerow([track, track_stats[track]])
    print('detailed labels saved at {}'.format(detailed_label_file))

    # save overall track memorability results
    label_score_file = os.path.join(args.labels_dir, 'track_memorability_scores_beta.csv')
    track_memo_scores = get_overall_track_memorability(track_stats, track_order)
    with open(label_score_file, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(['track', 'score'])
        for idx in range(len(track_order)):
            writer.writerow([track_order[idx], track_memo_scores[idx]])
    print('label scores saved at {}'.format(label_score_file))

    # save other experiment stats
    record_macro_stats(args, track_stats)

def get_overall_track_memorability(track_stats, track_order):
    ''' calcuate overall track memorability (regardless of interval length) 
        return list of memorability_score given track order
    '''
    track_memo_scores = []     
    for track in track_order:
        interval_memorized_list = track_stats[track]
        memorized_count = 0
        for interval_memorized_pair in interval_memorized_list:
            interval = list(interval_memorized_pair)[0]
            memorized = interval_memorized_pair.get(interval)
            memorized_count += memorized
        track_score = 0 if interval_memorized_list == [] else memorized_count/len(interval_memorized_list)
        track_memo_scores.append(track_score)
    return track_memo_scores

def record_macro_stats(args, track_stats):
    ''' save and plot the statistics of memorability against interval '''

    interval_memorability_pair_list = []
    for interval_memorability_pairs in track_stats.values():
        interval_memorability_pair_list.extend(interval_memorability_pairs)
    interval_memorability_dict = {}
    for interval_memorability_pair in interval_memorability_pair_list:
        interval = list(interval_memorability_pair.keys())[0]
        memorized = list(interval_memorability_pair.values())[0]
        if not interval in interval_memorability_dict.keys():
            interval_memorability_dict[interval] = [memorized]
        else:
            interval_memorability_dict[interval].append(memorized)

    interval_list = sorted(list(interval_memorability_dict.keys()))
    memorability_score_list = [interval_memorability_dict[interval].count(1)/len(interval_memorability_dict[interval]) for interval in interval_list]
    memorability_stat_list = ["{}/{}".format(interval_memorability_dict[interval].count(1), len(interval_memorability_dict[interval])) for interval in interval_list]
        
    stats_csv_file = os.path.join(args.stats_dir, 'interval_memorability.csv')
    with open(stats_csv_file, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(['interval', 'memorability_score', 'memorability_stats'])
        for idx in range(len(interval_list)):
            writer.writerow([interval_list[idx], memorability_score_list[idx], memorability_stat_list[idx]])
    print('total statistics csv saved at {}'.format(stats_csv_file))

    plt.figure()
    plt.title("memorability-interval plot")
    plt.xlabel("repeat interval")
    plt.ylabel("memorability score")
    plt.scatter(interval_list, memorability_score_list)
    stats_plot_file = os.path.join(args.stats_dir, 'macro_stats.png')
    plt.savefig(stats_plot_file)
    print('total statistics plot saved at {}'.format(stats_plot_file))

def calculate_consistency(args, qualified_data_df, track_order):
    ''' calculate memorability consistency by averaging a number of correlation '''
    

    correlations = [] # store correlations of different split
    out_fname = "correlations.csv"
    out_path = os.path.join(args.stats_dir, out_fname)
    with open(out_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["split_idx", "spearman's_correlation"])

        for idx in range(args.split_num):
            
            group_1_memo_scores, group_2_memo_scores = get_separated_groups_score(qualified_data_df)
            correlation, pval = stats.spearmanr(group_1_memo_scores, group_2_memo_scores)
            correlations.append(correlation)
            # write result to file
            writer.writerow([idx+1, correlation])

        writer.writerow(["average", sum(correlations)/len(correlations)])
    
    print("cosistency correlations saved at {}".format(out_path))
    
def get_separated_groups_score(qualified_data_df):
    ''' get memorability scores of two randomly split group
        returns
        (1) average memorability score of group 1
        (2) average memorability score of group 2
    '''
    
    # random split is done by shuffling dataframe, ref: https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    data_num = len(qualified_data_df)
    qualified_data_df = qualified_data_df.sample(frac=1)
    group1_data_df, group2_data_df = qualified_data_df.iloc[0:data_num//2], qualified_data_df.iloc[data_num//2:]
    
    # calculate memorability score in individual groups
    _, group1_track_stats = get_experiment_stats(group1_data_df)
    _, group2_track_stats = get_experiment_stats(group2_data_df)
    group1_track_memo_scores = get_overall_track_memorability(group1_track_stats, track_order)
    group2_track_memo_scores = get_overall_track_memorability(group2_track_stats, track_order)

    return group1_track_memo_scores, group2_track_memo_scores

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="experiment data config")
    parser.add_argument("--exp_data_path", help="path of experiment data", default="data/experimentData_beta.csv")
    parser.add_argument("--stats_dir", help="dir to store output", default="data/experiment_stats")
    parser.add_argument("--labels_dir", help="dir to store label", default="data/labels")
    parser.add_argument("--split_num", help="number of random split trials", default=25)
    args = parser.parse_args()

    raw_data = pd.read_csv(args.exp_data_path) # this file is exported from MySQL Database
    qualified_data = filter_data(raw_data)
    user_stats, track_stats = get_experiment_stats(qualified_data)
    track_order = list(track_stats.keys())
    save_stats(args, user_stats, track_stats, track_order)   
    calculate_consistency(args, qualified_data, track_order)
