'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-27 20:13:34
@LastEditTime: 2019-09-01 16:27:03
@LastEditors: Please set LastEditors
'''
import random


class Bucket:
    def __init__(self, src_len, tgt_len, batch_size):
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.batch_size = batch_size
        self.instances = []

    def append(self, ins):
        new_ins = self._pad_ins(ins)
        self.instances.append(new_ins)

    def replicate(self):
        if len(self.instances) != 0:
            remainder = len(self.instances) % self.batch_size
            if remainder == 0:
                supple_len = 0
            else:
                supple_len = self.batch_size - remainder
            replicates = [random.choice(self.instances) for idx in range(supple_len)]
            self.instances.extend(replicates)

    def shuffle(self):
        random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)
