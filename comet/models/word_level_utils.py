# import torch
# from more_itertools import collapse
import numpy as np
# import math

def convert_word_tags(wt_list):
        word_tags = []
        d = {'BAD': '<BAD>', 'OK': '<OK>'}
        for l in wt_list:
            word_tags.append(' '.join([d[x] for x in l.strip().split(' ')]))
        return word_tags

# def convert_word_tags_feats(wt_list):
#         word_tags = []
#         d = {'BAD': 1, 'OK': 0}
#         for l in wt_list:
#             word_tags.append([d[x] for x in l.split(' ')])
#         return word_tags


# def flatten(self, list_of_lists):
#     for list in list_of_lists:
#         for item in list:
#             yield item

# def confusion_matrix(hat_y_all, y_all, n_classes=None):
#     cnfm = np.zeros((n_classes, n_classes))

#     for hat_y, y in zip(hat_y_all, y_all):
#         hat_y = hat_y.view(-1,2).cpu()
#         y = y.view(-1).cpu()
#         for j in range(y.size(0)):
#             if y[j]>-1:
#                 hat_yj = np.argmax(hat_y[j])
#                 cnfm[int(y[j]), hat_yj] += 1
#     return cnfm

# def matthews_correlation_coefficient(hat_y, y):
#     """Compute Matthews Correlation Coefficient.
#     Arguments:
#         hat_y: list of np array of predicted binary labels.
#         y: list of np array of true binary labels.
#     Return:
#         the Matthews correlation coefficient of hat_y and y.
#     """

#     cnfm = confusion_matrix(hat_y, y, 2)
#     tp = cnfm[0][0]
#     tn = cnfm[1][1]
#     fp = cnfm[1][0]
#     fn = cnfm[0][1]
#     class_p = tp + fn
#     class_n = tn + fp
#     pred_p = tp + fp
#     pred_n = tn + fn
    
#     normalizer = class_p * class_n * pred_p * pred_n
  
#     if normalizer:
#         return ((tp * tn) - (fp * fn)) / math.sqrt(normalizer)
#     else:
#         return 0