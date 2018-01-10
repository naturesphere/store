
reddit_file = './corpus/english_context.txt'
vocab_list = []
with open(reddit_file, 'r',encoding="utf8") as f2:
    for line in f2:
        ln = line.strip()
        if not ln:
            continue
        tokens = ln.strip().split(' ')
        for token in tokens:
            if len(token) and token != ' ':
                t = token.lower()
                if t not in vocab_list:
                    vocab_list.append(t)
print(len(vocab_list))
print(sorted(vocab_list))