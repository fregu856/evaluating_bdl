# code-checked
# server-checked

import numpy as np

# function for colorizing a label image:
def label_img_2_color(img):
    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    img_color[img == 0] = np.array([128, 64, 128])
    img_color[img == 1] = np.array([244, 35,232])
    img_color[img == 2] = np.array([70, 70, 70])
    img_color[img == 3] = np.array([102,102,156])
    img_color[img == 4] = np.array([190,153,153])
    img_color[img == 5] = np.array([153,153,153])
    img_color[img == 6] = np.array([250,170, 30])
    img_color[img == 7] = np.array([220,220, 0])
    img_color[img == 8] = np.array([107,142, 35])
    img_color[img == 9] = np.array([152,251,152])
    img_color[img == 10] = np.array([ 70,130,180])
    img_color[img == 11] = np.array([220, 20, 60])
    img_color[img == 12] = np.array([255, 0, 0])
    img_color[img == 13] = np.array([0, 0, 142])
    img_color[img == 14] = np.array([0, 0, 70])
    img_color[img == 15] = np.array([0, 60,100])
    img_color[img == 16] = np.array([0, 80,100])
    img_color[img == 17] = np.array([0, 0,230])
    img_color[img == 18] = np.array([119, 11, 32])

    img_color[img == 255] = np.array([0, 0, 0])

    return img_color

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix
