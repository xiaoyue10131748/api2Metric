from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tools.prepare_data import *
from tools.model_helper import *
import tensorflow.keras as k
import pickle

class ABLSTM(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        #self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]
        self.perms_words=["gid uid suid security permission right capability capabilities capable capabable perm privilege permission securityfs"]
        self.UI_words=["user userspace mount user-namespace unmounting user-mode tty user context"]
        self.AC_words = ["must check verify validate determine acquire request revalidate require lock security capable access NetLabel gid sgid rights semaphore flag futex credential encryption"]
        self.AV_words =["bluetooth wireless tcp socket ip packet ack acks protocol netdevice rtnetlink inet sctp skbs network nfc console tty usb net device usbnet physical dvb gadget dev local tty console urb phy"]
        #self.keywords = keyword_preprocessing_v2(self.perms_words,max_len=50)
        self.keywords = keyword_preprocessing_v3(self.AV_words,self.max_len)




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

        input_sentense = k.Input((None,300))

        #x = k.layers.Embedding(self.vocab_size, self.embedding_size,weights=[self.embedding_matrix],trainable=False)(input_sentense)

        x = k.layers.Bidirectional(k.layers.LSTM(self.hidden_size, return_sequences=True))(input_sentense)

        x, _ = self.scaled_dot_product_attention(x, x, x, None)


        x = k.layers.Bidirectional(k.layers.LSTM(self.hidden_size, return_sequences=True))(x)

        x, _ = self.scaled_dot_product_attention(x, x, x, None)

        x, attention_map= self.scaled_dot_product_attention(self.keywords, x,  x, None)
        #tf.print(attention_map)

        x = k.layers.Dropout(0.20)(x)

        x = k.layers.GlobalAveragePooling1D()(x)

        x = k.layers.Dense(self.n_class)(x)

        x = k.layers.Activation('softmax')(x)

        print("graph built successfully!")

        return k.models.Model(input_sentense, x), k.models.Model(input_sentense, attention_map)

def rp(y_test, yp,sentence,api):
    rst = np.argmax(yp, axis=-1)
    precision_n = run_eval_step_precision(rst,y_test[:len(rst)],sentence,api,1)
    recall_n = run_eval_step_recall(rst,y_test[:len(rst)],sentence,api,1)
    print("the N precison is "+ str(precision_n))
    print("the N recall  is " + str(recall_n))
    print("\n")
    precision_a = run_eval_step_precision(rst,y_test[:len(rst)],sentence,api,2)
    recall_a = run_eval_step_recall(rst,y_test[:len(rst)],sentence,api,2)
    print("the A precison is "+ str(precision_a))
    print("the A recall  is " + str(recall_a))
    print("\n")
    precision_p = run_eval_step_precision(rst,y_test[:len(rst)],sentence,api,3)
    recall_p = run_eval_step_recall(rst,y_test[:len(rst)],sentence,api,3)
    print("the P precison is "+ str(precision_p))
    print("the P recall  is " + str(recall_p))
    print("\n")


    y_test_label = np.int32(y_test)[:len(rst)]
    #print('rp', np.sum(y_test_label * rst) / np.sum(y_test_label), np.sum(y_test_label * rst) / np.sum(rst))
    #print('acc', np.sum(y_test_label == rst) / len(rst))

def train():
    # load data

    x_train, y_train, api_train = load_data("../data/AV/train.csv", sample_ratio=1, one_hot=False)
    x_test, y_test, api_test = load_data("../data//AV/test.csv", one_hot=False)
    sentence, _, api = load_data("../data/AV/test.csv", one_hot=False)
    # data preprocessing
    x_train, x_test = data_preprocessing_v3(x_train, x_test, max_len=150)


    # split dataset to test and dev
    x_test_part, x_dev, y_test_part, y_dev, dev_size, test_size = \
        split_dataset(x_test, y_test, 0.2)
    print("Validation Size: ", dev_size)

    config = {
        "max_len": 150,
        "hidden_size": 150,
        "embedding_size": 300,
        "n_class": 4,
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

    model.save("./matix_model/AV_model")
    print("\n\n\n\n\n\n\n")


def test():

    x_train, y_train, api_train = load_data("../data/AV/linux/train.csv", sample_ratio=1, one_hot=False)
    x_test, y_test, api_test = load_data("../data/AV/linux/test.csv", one_hot=False)
    sentence, _, api = load_data("../data/AV/linux/test.csv", one_hot=False)
    # data preprocessing
    x_train, x_test = data_preprocessing_v3(x_train, x_test, max_len=150)


    print("second evaluating:  \n")
    AV_model = k.models.load_model("../models/AV_model_best")
    rst1 = AV_model.predict(fill_feed_dict_once(x_test, y_test, 1), verbose=1)
    rp(y_test, rst1, sentence, api)

def testall(test,tofile):
    sentence, _, api = load_data(test, one_hot=False)
    x_test, y_test, api_test = load_data(test, one_hot=False)
    x_test_index = x_test.index

    x_test = data_preprocessing_v4(x_test, max_len=150)
    print("second evaluating:  \n")
    AV_model = k.models.load_model("../models/AV_model_best")
    rst1 = AV_model.predict(fill_feed_dict_once(x_test, y_test, 8), verbose=1)
    results = np.argmax(rst1, axis=-1)

    api_list = []
    comment = []
    label_list=[]
    count=0
    for index, r in enumerate(results):
        if r == 1:
            count += 1
            series_index = x_test_index[index]
            api_list.append(str(api[series_index]))
            comment.append(str(sentence[series_index]))
            label_list.append("N")
            line = str(api[series_index]) + " :: " + str(sentence[series_index])
            print(line)
        if r == 2:
            count += 1
            series_index = x_test_index[index]
            api_list.append(str(api[series_index]))
            comment.append(str(sentence[series_index]))
            label_list.append("A")
            line = str(api[series_index]) + " :: " + str(sentence[series_index])
            print(line)
        if r == 3:
            count += 1
            series_index = x_test_index[index]
            api_list.append(str(api[series_index]))
            comment.append(str(sentence[series_index]))
            label_list.append("P")
            line = str(api[series_index]) + " :: " + str(sentence[series_index])
            print(line)

    data = {}
    data["api_list"] = api_list
    data["comment"] = comment
    data["label_list"] = label_list
    df = pd.DataFrame(data)
    df.to_excel(tofile)
    print(count)

if __name__ == '__main__':
    #train()
    test()
    #testall()