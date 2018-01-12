import os

VOCAB_FILE = "./corpus/vocab.txt"
filename = "./corpus/english_context.txt"

bos = "bos_"
eos = "_eos"

v_set = set()
with open(filename,'r') as f:
    for line in f:
        ts = set(line.lower().split())
        v_set |= ts
print("v list length = ", len(v_set))

if os.path.exists(VOCAB_FILE):
    os.remove(VOCAB_FILE)

with open(VOCAB_FILE, 'a', encoding="utf-8") as f:
    # i=0
    # L = len(v_set)
    for v in v_set:
        f.write(v+"\n")
        # i+=1
        # if i<L:
        #     f.write("\n")
    f.write(bos+"\n")
    f.write(eos)