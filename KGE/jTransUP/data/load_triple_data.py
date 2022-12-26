import os
import numpy as np
from jTransUP.utils.data import MakeTrainIterator, MakeEvalIterator


def loadTriples(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        triple_total = 0
        triple_list = []
        triple_head_dict = {}
        triple_tail_dict = {}
        for line in fin:
            line_split = line.strip().split('\t')
            line_split = line_split[0].split()
            if len(line_split) != 3:
                continue
            h_id = int(line_split[0])
            t_id = int(line_split[1])
            r_id = int(line_split[2])

            triple_list.append((h_id, t_id, r_id))

            tmp_heads = triple_head_dict.get((t_id, r_id), set())  # key是由(tail, relation)组成的二元组,value是head的集合
            tmp_heads.add(h_id)
            triple_head_dict[(t_id, r_id)] = tmp_heads  # {(t, r): {h1,h2,...}}

            tmp_tails = triple_tail_dict.get((h_id, r_id), set())
            tmp_tails.add(t_id)
            triple_tail_dict[(h_id, r_id)] = tmp_tails

            triple_total += 1

    return triple_total, triple_list, triple_head_dict, triple_tail_dict


# org-->id
def loadVocab(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        vocab = {}
        for line in fin:
            # line_split = line.strip().split('\t')
            line_split = line.strip().split(' ')
            if len(line_split) != 2:
                continue
            # e_id = int(line_split[0])
            # e_uri = line_split[1]
            e_uri = line_split[0]  # original entity
            e_id = int(line_split[1])  # mapped entity_id or relation_id

            vocab[e_uri] = e_id

    return vocab


def load_data(kg_path, eval_filenames, batch_size, negtive_samples=1, logger=None):
    # each dataset has the /kg/ dictionary

    train_file = os.path.join(kg_path, "train.txt")
    eval_files = []
    for file_name in eval_filenames:
        eval_files.append(os.path.join(kg_path, file_name))

    e_map_file = os.path.join(kg_path, "e_map.txt")
    r_map_file = os.path.join(kg_path, "r_map.txt")
    # 三元组数量   元组列表     字典,key:(t, r), value:集合{h1,h2,...}
    train_total, train_list, train_head_dict, train_tail_dict = loadTriples(train_file)

    eval_dataset = []
    for eval_file in eval_files:
        eval_dataset.append(loadTriples(eval_file))

    if logger is not None:
        eval_totals = [str(eval_data[0]) for eval_data in eval_dataset]  # 统计验证集中三元组的个数
        logger.info("Totally {} train triples, {} eval triples in files: {}!".format(train_total, ",".join(eval_totals),
                                                                                     ";".join(eval_files)))

    # get entity total
    e_map = loadVocab(e_map_file)  # 字典 org_id: mapped_id
    # get relation total
    r_map = loadVocab(r_map_file)

    if logger is not None:
        logger.info("successfully load {} entities and {} relations!".format(len(e_map), len(r_map)))

    train_iter = MakeTrainIterator(train_list, batch_size, negtive_samples=negtive_samples)  # 返回列表[[h t r], ...]

    # train_total, train_list, train_head_dict, train_tail_dict
    new_eval_datasets = []
    dt = np.dtype('int,int')
    for eval_data in eval_dataset:  # train_total, train_list, train_head_dict, train_tail_dict
        tmp_head_iter = MakeEvalIterator(list(eval_data[2].keys()), dt, batch_size)
        tmp_tail_iter = MakeEvalIterator(list(eval_data[3].keys()), dt, batch_size)
        new_eval_datasets.append([tmp_head_iter, tmp_tail_iter, eval_data[0], eval_data[1], eval_data[2], eval_data[3]])

    train_dataset = (train_iter, train_total, train_list, train_head_dict, train_tail_dict)

    return train_dataset, new_eval_datasets, e_map, r_map


if __name__ == "__main__":
    # Demo:
    data_path = "/Users/caoyixin/Github/joint-kg-recommender/datasets/ml1m/"

    # i2kg_file = os.path.join(data_path, 'i2kg_map.tsv')
    # i2kg_pairs = loadR2KgMap(i2kg_file)
    # e_set = set([p[1] for p in i2kg_pairs])

    rel_file = os.path.join(data_path + 'kg/', 'relation_filter.dat')
    rel_vocab = set()
    with open(rel_file, 'r') as fin:
        for line in fin:
            rel_vocab.add(line.strip())

    _, triple_datasets = load_data(data_path, rel_vocab=rel_vocab)

    trainList, testList, validList, e_map, r_map, entity_total, relation_total = triple_datasets
    print("entity:{}, relation:{}!".format(entity_total, relation_total))
    print("totally triples for {} train, {} valid, and {} test!".format(len(trainList), len(validList), len(testList)))
