import numpy as np


def make_train_feed_dict(model, batch):
    """make train feed dict for training"""
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: .5}
    return feed_dict


def make_test_feed_dict(model, batch):
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: 1.0}
    return feed_dict


def run_train_step(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
        'train_op': model.train_op,
        'loss': model.loss,
        'global_step': model.global_step,
    }
    return sess.run(to_return, feed_dict)


def run_eval_step_recall(prediction,true_label,sentence,api,c):
    recall_sum = 0
    recall_cnt = 0

    for i, label in enumerate(true_label):
        sentence_list=list(true_label.index)
        #print(sentence_list)
        if label == c:
            recall_sum+=1
            if prediction[i] ==c:
                recall_cnt+=1
            else:
                print("NOT recall :: " + api[sentence_list[i]] + "  ::  " + str(sentence[sentence_list[i]]))
    return recall_cnt/recall_sum



def run_eval_step_precision(prediction,true_label,sentence,api,c):
    precision_sum = 0
    precision_cnt = 0

    for i, label in enumerate(prediction):
        sentence_list = list(true_label.index)
        if label == c:
            precision_sum+=1
            if true_label[sentence_list[i]] ==c:
                precision_cnt+=1
            else:
                print("NOT precision :: " + api[sentence_list[i]] + "  ::  " + sentence[sentence_list[i]])
    #acc = np.sum(np.equal(prediction, batch[1])) / len(prediction)
    return precision_cnt/precision_sum



def get_attn_weight(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    return sess.run(model.alpha, feed_dict)
