'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-13 22:16:00
@LastEditTime: 2019-08-29 15:39:31
@LastEditors: Please set LastEditors
'''


BOS_SYMBOL = "<s>"
EOS_SYMBOL = "</s>"
UNK_SYMBOL = "<unk>"
PAD_SYMBOL = "<pad>"
PAD_ID = 0
END_ID = 3
TOKEN_SEPARATOR = " "
VOCAB_SYMBOLS = [PAD_SYMBOL, UNK_SYMBOL, BOS_SYMBOL, EOS_SYMBOL]
VOCAB_ENCODING = "utf-8"

LOG_NAME = "log.txt"
ARG_SEPARATOR = ":"

AMR_EDGE_TYPE = ["d", "r", "s", "g"]
DIRECTED_EDGE_ID = 1
REVERSE_EDGE_ID = 2
SELF_EDGE_ID = 3
GLOGAL_EDGE_ID = 4

PRINT_SPACE = 20
MINIMUM_VALUE = -1e15
