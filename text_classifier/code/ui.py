from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tools.prepare_data import *
from tools.model_helper import *
import tensorflow.keras as k
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tools.prepare_data import *
from tools.model_helper import *
import tensorflow.keras as k
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time

class ABLSTM(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        #self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]






    def scaled_dot_product_attention(self, q, k, v, mask):
        """计算注意力权重。
        q, k, v 必须具有匹配的前置维度。
        k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
        虽然 mask 根据其类型（填充或前瞻）有不同的形状，
        但是 mask 必须能进行广播转换以便求和。

        参数:
        q: 请求的形状 == (..., seq_len_q, depth)
        k: 主键的形状 == (..., seq_len_k, depth)
        v: 数值的形状 == (..., seq_len_v, depth_v)
        mask: Float 张量，其形状能转换成
              (..., seq_len_q, seq_len_k)。默认为None。

        返回值:
        输出，注意力权重
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # 缩放 matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # 将 mask 加入到缩放的张量上。
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            scaled_attention_logits += (mask * -1e9)

        # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
        # 相加等于1。
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights


    def build_graph(self):
        print("building graph")

        input_sentense = k.Input((None, 300))

        x = k.layers.Bidirectional(k.layers.LSTM(self.hidden_size, return_sequences=True))(input_sentense)

        x, _ = self.scaled_dot_product_attention(x, x, x, None)


        x = k.layers.Bidirectional(k.layers.LSTM(self.hidden_size, return_sequences=True))(x)

        x, attention_map = self.scaled_dot_product_attention(x, x, x, None)


        x = k.layers.Dropout(0.26)(x)

        x = k.layers.GlobalAveragePooling1D()(x)

        x = k.layers.Dense(self.n_class)(x)

        x = k.layers.Activation('softmax')(x)

        print("graph built successfully!")

        return k.models.Model(input_sentense, x), k.models.Model(input_sentense, attention_map)

def rp(y_test, yp,sentence,api):
    rst = np.argmax(yp, axis=-1)
    precision = run_eval_step_precision(rst,y_test[:len(rst)],sentence,api,1)
    recall = run_eval_step_recall(rst,y_test[:len(rst)],sentence,api,1)
    print("the precison is "+ str(precision))
    print("the recall  is " + str(recall))

    y_test_label = np.int32(y_test)[:len(rst)]
    print('rp', np.sum(y_test_label * rst) / np.sum(y_test_label), np.sum(y_test_label * rst) / np.sum(rst))
    print('acc', np.sum(y_test_label == rst) / len(rst))


def train():
    # load data

    x_train, y_train, api_train = load_data("../data/UI/train.csv", sample_ratio=1, one_hot=False)
    x_test, y_test, api_test = load_data("../data/UI/test.csv", one_hot=False)
    sentence, _, api = load_data("../data/UI/test.csv", one_hot=False)
    # data preprocessing
    x_train, x_test = data_preprocessing_v3(x_train, x_test, max_len=110)

    print("train size: ", len(x_train))

    # split dataset to test and dev
    x_test_part, x_dev, y_test_part, y_dev, dev_size, test_size = \
        split_dataset(x_test, y_test, 0.2)
    print("Validation Size: ", dev_size)

    config = {
        "max_len": 110,
        "hidden_size": 150,
        "embedding_size": 300,
        "n_class": 2,
        "learning_rate": 3e-4,
        "batch_size": 8,
        "train_epoch": 10
    }

    print(np.mean(y_train), np.mean(y_dev), np.mean(y_test))

    classifier = ABLSTM(config)
    model, att_model = classifier.build_graph()
    optimizer = k.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])
    print(model.summary())

    dev_batch = (x_dev, y_dev)

    saver = k.callbacks.ModelCheckpoint('lstm.hdf5', monitor='val_loss', save_best_only=True)

    model.fit(fill_feed_dict(x_train, y_train, config["batch_size"]), epochs=config['train_epoch'],
              steps_per_epoch=len(x_train) // config['batch_size'], validation_data=dev_batch, callbacks=[saver])

    print("Start evaluating:  \n")
    model.load_weights('lstm.hdf5')
    print("test size " + str(len(x_test)))
    print("label is 1 " + str(np.sum(y_test)))
    rst = model.predict(fill_feed_dict_once(x_test, y_test, config["batch_size"]), verbose=1)
    weights = att_model.predict(fill_feed_dict_once(x_test, y_test, config["batch_size"]), verbose=1)
    rp(y_test, rst, sentence, api)

    model.save("./self_atten_model/UI_model")
    att_model.save("./self_atten_model/UI_atten_weights")
    print("\n\n\n\n\n\n\n")

def testall(test,tofile):
    #"data/UI/linux_mainline/total_ui_linux.csv"
    sentence, _, api = load_data(test, one_hot=False)
    x_test, y_test, api_test = load_data(test, one_hot=False)
    x_test_index = x_test.index

    x_train, x_test = data_preprocessing_v4(x_test, max_len=130)
    print("second evaluating:  \n")
    AV_model = k.models.load_model("../models/UI_model_best")
    rst1 = AV_model.predict(fill_feed_dict_once(x_test, y_test, 8), verbose=1)
    results = np.argmax(rst1, axis=-1)

    api_list = []
    comment = []
    count=0
    for index, r in enumerate(results):
        if r == 1:
            count += 1
            series_index = x_test_index[index]
            api_list.append(str(api[series_index]))
            comment.append(str(sentence[series_index]))

            line = str(api[series_index]) + " :: " + str(sentence[series_index])
            print(line)

    data = {}
    data["api_list"] = api_list
    data["comment"] = comment
    df = pd.DataFrame(data)
    df.to_excel(tofile)
    print(count)

def test():
    x_train, y_train, api_train = load_data("../data/UI/train.csv", sample_ratio=1, one_hot=False)
    x_test, y_test, api_test = load_data("../data/UI/test.csv", one_hot=False)
    sentence, _, api = load_data("../data/UI/test.csv", one_hot=False)
    # data preprocessing
    x_train, x_test = data_preprocessing_v3(x_train, x_test, max_len=130)

    print("second evaluating:  \n")
    AV_model = k.models.load_model("./matix_model/UI_model_best")
    rst1 = AV_model.predict(fill_feed_dict_once(x_test, y_test, 1), verbose=1)
    rp(y_test, rst1, sentence, api)

# 注意力权重制图函数
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention[0:len(sentence)-1,0:len(sentence)-1], cmap='viridis')

    fontdict = {'fontsize': 6}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def plotAtten():
    config = {
        "max_len": 110,
        "hidden_size": 150,
        "embedding_size": 300,
        "n_class": 2,
        "learning_rate": 3e-4,
        "batch_size": 8,
        "train_epoch": 10
    }
    x_train, y_train, api_train = load_data("../data/UI/train.csv", sample_ratio=1, one_hot=False)
    x_test, y_test, api_test = load_data("../data/UI/test.csv", one_hot=False)
    sentence, _, api = load_data("../data/UI/test.csv", one_hot=False)
    # data preprocessing
    x_train, x_test = data_preprocessing_v3(x_train, x_test, max_len=110)

    print("train size: ", len(x_train))

    model= k.models.load_model("./self_atten_model/UI_model")
    att_model= k.models.load_model("./self_atten_model/UI_atten_weights")

    rst = model.predict(fill_feed_dict_once(x_test, y_test, config["batch_size"]), verbose=1)
    rst1 = np.argmax(rst, axis=-1)
    choice = None
    weights = att_model.predict(fill_feed_dict_once(x_test, y_test, config["batch_size"]), verbose=1)
    for index, r in enumerate(rst1):
        if r == 1:
            choice=index
            sentence_list = list(y_test.index)
            t1=sentence[sentence_list[choice]].split(" ")
            plot_attention(weights[choice], t1, t1)



if __name__ == '__main__':
    #train()
    #plotAtten()
    #test()
    testall()
