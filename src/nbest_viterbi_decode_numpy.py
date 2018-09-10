# coding:utf-8
# @Author: paperplanet
# @Date  : 2018/9/8

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np


def init_variable():
    emission_score = np.array([[[6.68578529e+00, -3.20134735e+00, -3.24944526e-01,
                        4.88783658e-01, -4.98658276e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [7.42727852e+00, -3.41898274e+00, -3.59271020e-01,
                        5.71676314e-01, -5.48904276e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [7.90740728e+00, -3.51931071e+00, -4.09875840e-01,
                        6.59939051e-01, -5.82971621e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [8.43343544e+00, -3.59035063e+00, -4.95426089e-01,
                        8.42134893e-01, -6.27256584e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [8.48750973e+00, -3.52486658e+00, -4.94948953e-01,
                        9.78451014e-01, -6.44798851e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [8.51491833e+00, -3.42291021e+00, -4.94709462e-01,
                        1.12926447e+00, -6.61188412e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [7.86455250e+00, -3.05053878e+00, -4.68682379e-01,
                        1.28452647e+00, -6.37033653e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [8.33770275e+00, -3.18704128e+00, -4.37698454e-01,
                        1.39193630e+00, -6.85782719e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [8.09165096e+00, -2.98072743e+00, -5.55660486e-01,
                        1.68692219e+00, -6.87993240e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.26212215e+00, -3.36437774e+00, -5.44141650e-01,
                        1.66799164e+00, -7.75796938e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.68278217e+00, -3.48212934e+00, -7.82783628e-01,
                        1.98279285e+00, -8.12523651e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [6.39409351e+00, -2.05648351e+00, -8.43996227e-01,
                        2.66659927e+00, -6.50281763e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [-2.02719855e+00, 1.40851319e+00, 3.43898803e-01,
                        2.62654924e+00, -2.29798698e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [-3.10617709e+00, 1.34726059e+00, 8.54935467e-01,
                        2.51036835e+00, -2.08814931e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [-2.47049284e+00, 6.73485458e-01, 1.14099884e+00,
                        2.39840865e+00, -2.69990969e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [-8.03102732e-01, -3.44721317e-01, 1.33616400e+00,
                        2.22294331e+00, -3.80291629e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [8.52772999e+00, -3.68012142e+00, 8.02533925e-02,
                        1.88301671e+00, -8.32959270e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [1.02204857e+01, -3.89145184e+00, -3.57173949e-01,
                        1.79651713e+00, -8.88961124e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [1.08522902e+01, -3.92075944e+00, -5.12615442e-01,
                        1.81169105e+00, -9.14243031e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.64291668e+00, -3.30883694e+00, -6.42375112e-01,
                        2.13370490e+00, -8.50732517e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00]],

                      [[8.39771175e+00, -3.84396863e+00, -4.60517079e-01,
                        6.04534805e-01, -6.15453243e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [8.61073112e+00, -3.85449433e+00, -4.85974222e-01,
                        7.23589480e-01, -6.39145422e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [8.77016544e+00, -3.81910348e+00, -4.89924937e-01,
                        8.32199156e-01, -6.60681009e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [8.70501995e+00, -3.68190289e+00, -4.73491281e-01,
                        9.29595649e-01, -6.69948959e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [8.52464581e+00, -3.58832645e+00, -4.37417001e-01,
                        1.05047429e+00, -6.76847982e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.11687756e+00, -3.75543737e+00, -4.39233989e-01,
                        1.15986693e+00, -7.27008915e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.50767708e+00, -3.83756638e+00, -4.65802640e-01,
                        1.26141000e+00, -7.61667585e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.93514538e+00, -3.90708065e+00, -5.30415058e-01,
                        1.41912842e+00, -8.00839710e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.94030762e+00, -3.86018991e+00, -5.30434370e-01,
                        1.54605734e+00, -8.15033245e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.94352531e+00, -3.79223990e+00, -5.40814698e-01,
                        1.67290246e+00, -8.26585579e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.32656384e+00, -3.47475719e+00, -5.43340564e-01,
                        1.84424078e+00, -8.02706909e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.69780731e+00, -3.60773420e+00, -5.16409636e-01,
                        1.87792301e+00, -8.35943699e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.36379719e+00, -3.40826702e+00, -6.35896087e-01,
                        2.11203456e+00, -8.22353458e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [1.03167448e+01, -3.72574186e+00, -6.11401558e-01,
                        1.96223569e+00, -8.80881596e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [1.05330744e+01, -3.78194571e+00, -8.19724202e-01,
                        2.17305541e+00, -8.92132854e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [7.32272434e+00, -2.42377615e+00, -9.09445167e-01,
                        2.89774251e+00, -7.34459209e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [-1.03870046e+00, 9.90443766e-01, 2.70201236e-01,
                        3.01525354e+00, -3.31264377e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [1.22717813e-01, -2.15979218e-02, 5.38439095e-01,
                        3.09278321e+00, -4.42224693e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [-2.32051134e-01, -2.09532470e-01, 9.18213069e-01,
                        2.87649727e+00, -4.32271719e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [7.78469038e+00, -3.24421501e+00, -2.55472399e-02,
                        2.35747123e+00, -8.14153004e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.37604332e+00, -3.49119139e+00, -4.35785562e-01,
                        2.22020173e+00, -8.60559559e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.92206955e+00, -3.51610708e+00, -5.50446033e-01,
                        2.16868186e+00, -8.77168274e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.26817036e+00, -3.10642624e+00, -6.75005257e-01,
                        2.37851858e+00, -8.38644314e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [8.17101192e+00, -2.56197929e+00, -6.36741102e-01,
                        2.55563188e+00, -7.80730438e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [5.43278503e+00, -1.40049756e+00, -2.46703833e-01,
                        2.61345887e+00, -6.38186550e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [9.79092979e+00, -3.35115814e+00, -5.17103553e-01,
                        2.13563585e+00, -8.66261387e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [1.04898672e+01, -3.53461742e+00, -6.39010549e-01,
                        2.06201220e+00, -8.89705944e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [1.12619057e+01, -3.77836967e+00, -6.91998839e-01,
                        1.89470613e+00, -9.24510670e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [1.06390438e+01, -3.42988539e+00, -8.63238633e-01,
                        2.14935565e+00, -8.87445831e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [1.02402077e+01, -3.32775497e+00, -8.13070416e-01,
                        2.22191405e+00, -8.73875713e+00, -1.00000000e+03,
                        -1.00000000e+03],
                       [8.79168892e+00, -2.76306057e+00, -8.30264568e-01,
                        2.55729938e+00, -8.09503841e+00, -1.00000000e+03,
                        -1.00000000e+03]]])

    transition_score = np.array([[0.09870817, -0.11166503, -8.50794, -8.636973, -0.12820053,
                             -8.826562, 0.04264949],
                            [-7.582322, -7.6190705, -0.10145799, 0.08160423, -9.463068,
                             -7.803242, -9.325153],
                            [0.02235632, -0.11289039, -8.063921, -8.288115, -0.02873811,
                             -9.008408, -0.04201475],
                            [-7.4452763, -8.002927, 0.08172115, -0.10276299, -6.910625,
                             -7.9444375, -9.507603],
                            [-0.03417129, -0.04747989, -7.231127, -7.0696354, -0.01756917,
                             -9.096269, -0.04244959],
                            [0.05651978, -0.04987099, -8.29099, -8.556647, -0.05912597,
                             -7.361445, -8.161939],
                            [-8.471018, -9.404301, -8.423072, -9.044096, -8.549261,
                             -7.5359087, -7.6124372]], dtype=np.float32)

    mask = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1]])

    id_to_tag_all = {0: u'O', 1: u'B', 2: u'E', 3: u'M', 4: u'S', 5: u'_START', 6: u'_STOP'}

    return emission_score, mask, transition_score, id_to_tag_all


def viterbi_decode_nbest(emission_score, mask, transition_score, nbest=3):
    """
        input:
            emission_score: (batch, seq_len, tag_size)
            mask: (batch, seq_len)
            transition_score: (tag_size,tag_size)
        output:
            decode_idx: (batch, seq_len, nbest) decoded sequence
            path_score: (batch, nbest)
    """
    START_TAG = -2  # START_TAG的标签集id为倒数第2个
    STOP_TAG = -1  # STOP_TAG的标签集id为最后一个
    batch_size = emission_score.shape[0]
    seq_len = emission_score.shape[1]  # batch中最长的句子长度
    tag_size = emission_score.shape[2]  # 标签集大小，START_TAG,END_TAG
    length_mask = np.sum(mask, axis=1).reshape([batch_size, 1])  # (batch_size,1)的列向量
    ins_num = seq_len * batch_size

    mask = mask.transpose((1, 0))  # mask转置成(seq_len, batch_size)形状
    emission_score = emission_score.transpose([1, 0, 2]).reshape([ins_num, 1, tag_size])

    scores = emission_score + transition_score.reshape([1, tag_size, tag_size])
    scores = scores.reshape([seq_len, batch_size, tag_size, tag_size])

    seq_iter = enumerate(scores)

    back_points = list()
    nbest_scores_t_history = list()

    t, scores_t0 = next(seq_iter)  # scores_t0.shape = (batch_size,tag_size,tag_size)

    nbest_scores_t = scores_t0[:, START_TAG, :]  # (batch_size,tag_size)

    nbest_scores_t_history.append(np.tile(nbest_scores_t.reshape([batch_size, tag_size, 1]), [1, 1, nbest]))

    for t, scores_t in seq_iter:
        if t == 1:
            scores_t = scores_t.reshape([batch_size, tag_size, tag_size]) \
                         + nbest_scores_t.reshape([batch_size, tag_size, 1])
        else:
            scores_t = np.tile(scores_t.reshape([batch_size, tag_size, 1, tag_size]), [1, 1, nbest, 1]) \
                         + np.tile(nbest_scores_t.reshape([batch_size, tag_size, nbest, 1]), [1, 1, 1, tag_size])
            scores_t = scores_t.reshape([batch_size, tag_size * nbest, tag_size])

        cur_bp = np.argsort(scores_t, axis=1)[:, -nbest:][:, ::-1, :]
        # cur_bp为(batch_size,nbest,tag_size)形状的index，每个值的意义为，忽略第0维batch_size时，每一列前n个最大值的行号
        # argsort会做全排序，此处可以先argnbest_scores_t做局部排序，然后对前n个序列再做全排序，提高效率
        nbest_scores_t = scores_t[
            np.tile(np.arange(0, batch_size).reshape(batch_size, 1, 1),[1,nbest,tag_size]),
            cur_bp,
            np.tile(np.arange(0, tag_size).reshape([1, 1, tag_size]), [batch_size, nbest, 1])]
        # nbest_scores_t为在scores_t中每一行的topk取出的值(batch_size,nbest,tag_size)
        if t == 1:
            cur_bp = cur_bp * nbest
        nbest_scores_t = nbest_scores_t.transpose([0, 2, 1])
        # 形状为(batch_size,tag_size,nbest)，转置是因为在(tag_size,tag_size)的转移矩阵中，
        # 行号为上一时刻的tag编号，列号为当前时刻的tag编号，所以当前的n个最优序列需要转置后广播作为下一个时刻的行号，即出发tag
        cur_bp = cur_bp.transpose([0, 2, 1])  # 形状为(batch_size,tag_size,nbest)
        nbest_scores_t_history.append(nbest_scores_t)
        cur_bp = np.multiply(cur_bp, np.tile(mask[t].reshape([batch_size, 1, 1]), [1, tag_size, nbest]))
        back_points.append(cur_bp)
    nbest_scores_t_history = np.concatenate(nbest_scores_t_history, axis=0).reshape(
        [seq_len, batch_size, tag_size, nbest]).transpose([1, 0, 2, 3])
    ## (batch_size, seq_len, tag_size, nbest)
    last_position = np.tile(length_mask.reshape([batch_size, 1, 1, 1]), [1, 1, tag_size, nbest]) - 1
    last_nbest_scores = nbest_scores_t_history[
        np.tile(np.arange(batch_size).reshape([batch_size, 1, 1, 1]), [1, 1, tag_size, nbest]),
        last_position,
        np.tile(np.arange(tag_size).reshape([1, 1, tag_size, 1]), [batch_size, 1, 1, nbest]),
        np.tile(np.arange(nbest).reshape([1, 1, 1, nbest]), [batch_size, 1, tag_size, 1])]
    # 形状为(batch_size,1,tag_size,nbest)
    # 获取每个batch中example最后有效时间步的nbest_scores_t矩阵

    # 计算最后一个时间步转移到END_TAG的过程
    last_nbest_scores = last_nbest_scores.reshape([batch_size, tag_size, nbest, 1])
    last_values = np.tile(last_nbest_scores, [1, 1, 1, tag_size]) \
                  + np.tile(transition_score.reshape(1, tag_size, 1, tag_size), [batch_size, 1, nbest, 1])
    last_values = last_values.reshape([batch_size, tag_size * nbest, tag_size])

    end_bp = np.argsort(last_values, axis=1)[:, -nbest:][:, ::-1, :]
    end_nbest_scores = last_values[
        np.tile(np.arange(0, batch_size).reshape(batch_size, 1, 1), [1,nbest,tag_size])
        , end_bp,
        np.tile(np.arange(0, tag_size).reshape([1, 1, tag_size]), [batch_size, nbest, 1])]
    # end_nbest_scores为在last_values中每一列的topk取出的值(batch_size,nbest,tag_size)
    end_bp = end_bp.transpose([0, 2, 1])
    # 形状为(batch_size,tag_size,nbest)
    pad_zero = np.zeros([batch_size, tag_size, nbest], dtype=np.int32)
    back_points.append(pad_zero)
    back_points = np.concatenate(back_points, axis=0).reshape([seq_len, batch_size, tag_size, nbest])
    # (seq_len,batch_size,tag_size,nbest)
    last_pointer = end_bp[:, STOP_TAG, :]
    # (batch_size, nbest)
    insert_last = np.tile(last_pointer.reshape([batch_size, 1, 1, nbest]), [1, 1, tag_size, 1])
    # (batch_size,1,tag_size,nbest)
    back_points = back_points.transpose([1, 0, 2, 3])
    # (batch_size,seq_len,tag_size,nbest)

    back_points[np.tile(np.arange(0, batch_size).reshape([batch_size, 1, 1, 1]), [1, 1, tag_size, nbest]),
                last_position,
                np.tile(np.arange(0, tag_size).reshape([1, 1, tag_size, 1]), [batch_size, 1, 1, nbest]),
                np.tile(np.arange(0, nbest).reshape([1, 1, 1, nbest]), [batch_size, 1, tag_size, 1])] = insert_last
    # 把back_points中每个example对应的最后一个时间步的(tag_size,nbest)的全零矩阵改成insert_last中对应example的矩阵
    back_points = back_points.transpose([1, 0, 2, 3])
    # (seq_len, batch_size, tag_size, nbest)
    decode_idx = np.zeros([seq_len, batch_size, nbest], dtype=np.int32)
    decode_idx[-1] = (last_pointer / nbest).astype(np.int32)
    mask_1 = np.array([([1] * n + [0] * (seq_len-n))  for n in length_mask.reshape([batch_size])-1]).transpose([1,0])
    # (seq_len,batch_size)
    pointer = last_pointer.copy()
    for t in range(len(back_points) - 2, -1, -1):
        new_pointer = back_points[t].reshape([batch_size, tag_size * nbest])[
            np.tile(np.arange(0, batch_size).reshape(batch_size, 1), [1, nbest]),
            pointer.reshape([batch_size, nbest]).astype(np.int32)]
        new_pointer = new_pointer * mask_1[t].reshape([batch_size, 1]) \
                      + last_pointer.reshape([batch_size, nbest]) * (1-mask_1[t]).reshape([batch_size, 1])
        # 每个example最后一个tag直接使用last_pointer的值，往前再开始顺藤摸瓜
        decode_idx[t] = (new_pointer / nbest).astype(np.int32)
        # # use new pointer to remember the last end nbest ids for non longest
        pointer = new_pointer
    decode_idx = decode_idx.transpose([1, 0, 2])
    # (batch_size,seq_len,nbest)
    scores = end_nbest_scores[:, :, STOP_TAG]

    def softmax(x, axis=1):
        max_x = np.max(x, axis=axis, keepdims=True)
        minus_x = x - max_x
        return np.exp(minus_x) / np.sum(np.exp(minus_x), axis=axis, keepdims=True)

    path_score = softmax(scores)

    return path_score, decode_idx


if __name__ == "__main__":
    emission_score, mask, transition_score, id_to_tag_all = init_variable()
    path_score, decode_idx = viterbi_decode_nbest(emission_score, mask, transition_score)
    # print("path_score:\n",path_score)
    # print("decode_idx:\n",decode_idx)
    seq_length = np.sum(mask, axis=1).tolist()
    decode_idx_list = decode_idx.transpose([0, 2, 1]).tolist()
    for example_id in range(len(decode_idx_list)):
        for nbest_id in range(len(decode_idx_list[example_id])):
            print("example {}, {}st best seq:".format(example_id, nbest_id + 1),
                  "".join([id_to_tag_all[x] for idx, x in enumerate(decode_idx_list[example_id][nbest_id]) if
                           idx < seq_length[example_id]]),
                  "\tscore: {}".format(path_score[example_id][nbest_id]))
