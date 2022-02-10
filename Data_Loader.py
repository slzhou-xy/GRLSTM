import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import gcn_emb


class MyData(Dataset):
    def __init__(self, data, label, double_neighbor_label):
        self.data = data
        self.label = label
        self.double_neighbor_label = double_neighbor_label
        self.idx = list(range(len(data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tuple_ = (self.data[idx], self.label[idx], self.idx[idx], self.double_neighbor_label[idx])
        return tuple_


def load_weight(emb_file):
    weights_l = np.loadtxt(emb_file, skiprows=1)

    emb_w = np.zeros((weights_l.shape[0] + 1, weights_l.shape[1] - 1))
    for weight in weights_l:
        idx = int(weight[0])
        emb_w[idx, :] = weight[1:]
        if idx <= 10:
            print(idx)
    return emb_w


def load_traindata(train_file):
    data = np.load(train_file, allow_pickle=True)
    x = data['train_list']
    y_idx = data['train_y']

    double_neighbor_label = data['train_y'][data['train_y']]

    return x, y_idx, double_neighbor_label


def load_valdata(val_file):
    data = np.load(val_file, allow_pickle=True)
    x = data['val_list']
    y_idx = data['train_y']

    return x, y_idx


def load_testdata(val_file):
    data = np.load(val_file, allow_pickle=True)
    x = data['test_list']
    y_idx = data['test_y']

    return x, y_idx


'''
TODO: ADD 
'''
def load_poi(poi_file):
    data = np.load(poi_file, allow_pickle=True)
    neighbor = data['poi_neighbors']
    poi = data['poi']
    return neighbor, poi


"""----------------------------------------------No GCN Data Loader----------------------------------------------"""


# load train data for training
def TrainValueDataLoader(train_file, emb_file, poi_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        label = [sq[1] for sq in data_tuple]
        idx_list = [sq[2] for sq in data_tuple]

        # 二邻居
        double_neighbor_label = [sq[3] for sq in data_tuple]

        data_neg = []
        # 对每个轨迹采样一个不是最近的轨迹
        for idx, d in enumerate(data):
            neg = np.random.randint(emb_weights.shape[0])
            while neg == idx_list[idx] or neg == label[idx]:
                neg = np.random.randint(emb_weights.shape[0])
            data_neg.append(torch.LongTensor(train_x[neg]))

        double_neighbor_neg = []
        for idx, d in enumerate(data):
            neg = np.random.randint(emb_weights.shape[0])
            while neg == idx_list[idx] or neg == double_neighbor_label[idx]:
                neg = np.random.randint(emb_weights.shape[0])
            double_neighbor_neg.append(torch.LongTensor(train_x[neg]))

        data_label = []
        for d in label:
            data_label.append(torch.LongTensor(train_x[d]))

        double_neighbor_data_label = []
        for d in double_neighbor_label:
            double_neighbor_data_label.append(torch.LongTensor(train_x[d]))

        data_length = [len(sq) for sq in data]
        neg_length = [len(sq) for sq in data_neg]
        label_length = [len(sq) for sq in data_label]

        double_neighbor_neg_length = [len(sq) for sq in double_neighbor_neg]
        double_neighbor_label_length = [len(sq) for sq in double_neighbor_data_label]

        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=emb_weights.shape[0] - 1)  # 用零补充，使长度对齐
        data_label = rnn_utils.pad_sequence(data_label, batch_first=True,
                                            padding_value=emb_weights.shape[0] - 1)  # 用零补充，使长度对齐
        data_neg = rnn_utils.pad_sequence(data_neg, batch_first=True,
                                          padding_value=emb_weights.shape[0] - 1)  # 这行代码只是为了把列表变为tensor

        double_neighbor_neg = rnn_utils.pad_sequence(double_neighbor_neg, batch_first=True,
                                                     padding_value=emb_weights.shape[0] - 1)
        double_neighbor_data_label = rnn_utils.pad_sequence(double_neighbor_data_label, batch_first=True,
                                                            padding_value=emb_weights.shape[0] - 1)

        return data, data_neg, data_label, data_length, neg_length, label_length, \
               double_neighbor_neg, double_neighbor_data_label, double_neighbor_neg_length, double_neighbor_label_length

    train_x, train_y, double_neighbor_label = load_traindata(train_file)
    emb_weights = load_weight(emb_file)
    data_ = MyData(train_x, train_y, double_neighbor_label)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    '''
    TODO: ADD 
    '''
    neighbor, poi = load_poi(poi_file)
    # 任意选一个负样本
    poi_neg = []
    for idx, d in enumerate(poi):
        neg = np.random.randint(28342)
        while neg == poi[idx] or neg in neighbor[idx]:
            neg = np.random.randint(28342)
        poi_neg.append(neg)
    poi_neg = torch.LongTensor(poi_neg)

    # 任意选一个正样本
    poi_label = []
    for n in neighbor:
        rand = np.random.randint(n.shape[0])
        poi_label.append(n[rand])
    poi_label = torch.LongTensor(poi_label)

    return dataset, emb_weights, poi_label, poi_neg


"""----------------------------------------------GCN Data Loader----------------------------------------------"""


# load train data for training gcn
def GcnTrainValueDataLoader(train_file, emb_file, poi_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)

        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        label = [sq[1] for sq in data_tuple]
        idx_list = [sq[2] for sq in data_tuple]

        # 二邻居
        double_neighbor_label = [sq[3] for sq in data_tuple]

        data_neg = []
        for idx, d in enumerate(data):
            neg = np.random.randint(emb_weights.shape[0])
            while neg == idx_list[idx] or neg == label[idx]:
                neg = np.random.randint(emb_weights.shape[0])
            data_neg.append(torch.LongTensor(train_x[neg]))

        double_neighbor_neg = []
        for idx, d in enumerate(data):
            neg = np.random.randint(emb_weights.shape[0])
            while neg == idx_list[idx] or neg == double_neighbor_label[idx]:
                neg = np.random.randint(emb_weights.shape[0])
            double_neighbor_neg.append(torch.LongTensor(train_x[neg]))

        data_label = []
        for d in label:
            data_label.append(torch.LongTensor(train_x[d]))

        double_neighbor_data_label = []
        for d in double_neighbor_label:
            double_neighbor_data_label.append(torch.LongTensor(train_x[d]))

        data_length = [len(sq) for sq in data]
        neg_length = [len(sq) for sq in data_neg]
        label_length = [len(sq) for sq in data_label]

        double_neighbor_neg_length = [len(sq) for sq in double_neighbor_neg]
        double_neighbor_label_length = [len(sq) for sq in double_neighbor_data_label]

        data = rnn_utils.pad_sequence(data, batch_first=True,
                                      padding_value=emb_weights.shape[0] - 1)
        data_label = rnn_utils.pad_sequence(data_label, batch_first=True,
                                            padding_value=emb_weights.shape[0] - 1)
        data_neg = rnn_utils.pad_sequence(data_neg, batch_first=True,
                                          padding_value=emb_weights.shape[0] - 1)

        double_neighbor_neg = rnn_utils.pad_sequence(double_neighbor_neg, batch_first=True,
                                                     padding_value=emb_weights.shape[0] - 1)
        double_neighbor_data_label = rnn_utils.pad_sequence(double_neighbor_data_label, batch_first=True,
                                                            padding_value=emb_weights.shape[0] - 1)

        return data, data_neg, data_label, data_length, neg_length, label_length,\
               double_neighbor_neg, double_neighbor_data_label, double_neighbor_neg_length, double_neighbor_label_length

    train_x, train_y, double_neighbor_label = load_traindata(train_file)

    emb_weights = gcn_emb.gcn_emb()
    data_ = MyData(train_x, train_y, double_neighbor_label)
    # 将数据按照batch_size的批次读入
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    '''
    TODO: ADD 
    '''
    neighbor, poi = load_poi(poi_file)
    # 任意选一个负样本
    poi_neg = []
    for idx, d in enumerate(poi):
        neg = np.random.randint(28343)
        while neg == poi[idx] or neg in neighbor[idx]:
            neg = np.random.randint(28343)
        poi_neg.append(neg)
    poi_neg = torch.LongTensor(poi_neg)

    # 任意选一个正样本
    poi_label = []
    for n in neighbor:
        rand = np.random.randint(n.shape[0])
        poi_label.append(n[rand])
    poi_label = torch.LongTensor(poi_label)

    return dataset, emb_weights, poi_label, poi_neg


"""----------------------------------------------Val Data Loader----------------------------------------------"""


# load train data for validation
def TrainDataValLoader(train_file, emb_weights, poi_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        label = [torch.LongTensor([sq[1]]) for sq in data_tuple]
        idx_list = [sq[2] for sq in data_tuple]

        data_length = [len(sq) for sq in data]
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=emb_weights.shape[0] - 1)  # 用零补充，使长度对齐
        label = rnn_utils.pad_sequence(label, batch_first=True,
                                       padding_value=0.0)  # 用零补充，使长度对齐
        return data, label, data_length, idx_list

    val_x, val_y, _ = load_traindata(train_file)
    data_ = MyData(val_x, val_y, val_y)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    '''
    TODO: ADD 
    '''
    neighbor, poi = load_poi(poi_file)
    # 任意选一个负样本
    poi_neg = []
    for idx, d in enumerate(poi):
        neg = np.random.randint(28343)
        while neg == poi[idx] or neg in neighbor[idx]:
            neg = np.random.randint(28343)
        poi_neg.append(neg)
    poi_neg = torch.LongTensor(poi_neg)

    # 任意选一个正样本
    poi_label = []
    for n in neighbor:
        rand = np.random.randint(n.shape[0])
        poi_label.append(n[rand])
    poi_label = torch.LongTensor(poi_label)

    return dataset, poi_label, poi_neg


def ValValueDataLoader(val_file, emb_weights, poi_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        label = [sq[1] for sq in data_tuple]
        idx_list = [sq[2] for sq in data_tuple]

        data_length = [len(sq) for sq in data]
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=emb_weights.shape[0] - 1)  # 用零补充，使长度对齐
        return data, label, data_length, idx_list

    val_x, val_y = load_valdata(val_file)
    data_ = MyData(val_x, val_y, val_y)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    '''
    TODO: ADD 
    '''
    neighbor, poi = load_poi(poi_file)
    # 任意选一个负样本
    poi_neg = []
    for idx, d in enumerate(poi):
        neg = np.random.randint(28343)
        while neg == poi[idx] or neg in neighbor[idx]:
            neg = np.random.randint(28343)
        poi_neg.append(neg)
    poi_neg = torch.LongTensor(poi_neg)

    # 任意选一个正样本
    poi_label = []
    for n in neighbor:
        rand = np.random.randint(n.shape[0])
        poi_label.append(n[rand])
    poi_label = torch.LongTensor(poi_label)

    return dataset, poi_label, poi_neg


"""----------------------------------------------Test Data Loader----------------------------------------------"""


def TestValueDataLoader(val_file, emb_weights, poi_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        label = [sq[1] for sq in data_tuple]
        idx_list = [sq[2] for sq in data_tuple]

        data_length = [len(sq) for sq in data]
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=emb_weights.shape[0] - 1)  # 用零补充，使长度对齐
        return data, label, data_length, idx_list

    val_x, val_y = load_testdata(val_file)
    data_ = MyData(val_x, val_y, val_y)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    '''
    TODO: ADD 
    '''
    neighbor, poi = load_poi(poi_file)
    # 任意选一个负样本
    poi_neg = []
    for idx, d in enumerate(poi):
        neg = np.random.randint(28343)
        while neg == poi[idx] or neg in neighbor[idx]:
            neg = np.random.randint(28343)
        poi_neg.append(neg)
    poi_neg = torch.LongTensor(poi_neg)

    # 任意选一个正样本
    poi_label = []
    for n in neighbor:
        rand = np.random.randint(n.shape[0])
        poi_label.append(n[rand])
    poi_label = torch.LongTensor(poi_label)

    return dataset, poi_label, poi_neg
