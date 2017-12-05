# -*- coding: utf-8 -*-

import collections
import os
import sys
import numpy as np

# 将数据转换为字词向量
def data_process(file_name):
    datas = []
    # 提取每行诗的标题和内容，去除长度小于5和大于79的数据，并按诗的字数排序
    with open(file_name, "r") as f:
        for line in f.readlines():
            try:
                line = line.decode('UTF-8')
                line = line.strip(u'\n')
                title, content = line.strip(u' ').split(u':')  
                content = content.replace(u' ',u'')  
                if u'_' in content or u'(' in content or u'（' in content or u'《' in content or u'[' in content: 
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = u'[' + content + u']' 
                datas.append(content)
            except ValueError as e:
                pass
    # 按字数从小到多排序
    datas = sorted(datas, key=lambda l: len(line))
    all_words = []
    for data in datas:
      # 将所有字拆分到数组里
      all_words += [word for word in data]
    # 统计每个字出现的次数。counter输出格式 {'不': 13507, '人': 12003, '山': 10162, '风': 9675}
    counter = collections.Counter(all_words)
    # 对数据做格式转换，变为［('不', 13507), ('人', 12003), ('山', 10162), ('风', 9675)］
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    # 每个字映射为一个数字ID , {'不': 4, '人': 5, '山': 6, '风': 7}
    word_num_map = dict(zip(words, range(len(words)))) 
    # 把原始数据诗词转换成词向量的形式
    to_num = lambda word: word_num_map.get(word, len(words))
    datas_vector = [ list(map(to_num, data)) for data in datas] 
    return datas_vector, word_num_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = poems_vec[start_index:end_index]
        length = max(map(len, batches))
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            x_data[row, :len(batches[row])] = batches[row]
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


if __name__ == '__main__':
  data_vector, word_num_map, words = data_process('../data/poetry.txt')
  x, y = generate_batch(64, data_vector, word_num_map)
  print(len(x))