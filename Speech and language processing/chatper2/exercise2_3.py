import re, string

patterns = [
    (r"\b(i'm|i am)\b", "YOU ARE"),
    (r"\b(i|me)\b","YOU"),
    (r"\b(my)\b", "YOUR"),
    (r"\b(well,?) ", ""),
    (r".* YOU ARE (depressed|sad) .*", r"I AM SORRY TO HEAR YOU ARE \1"),
    (r".* YOU ARE (depressed|sad) .*", r"WHY DO YOU THINK YOU ARE \1"),
    (r".* all .*", "IN WHAT WAY"),
    (r".* always .*", "CAN YOU THINK OF A SPECIFIC EXAMPLE"),
    (r"[%s]"%re.escape(string.punctuation),""),
]

while True:
    comment = input("You: ")
    response = comment.lower()
    for pat, sub in patterns:
        response = re.sub(pat,sub,response)
    print("Bot:"+response)