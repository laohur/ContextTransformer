# from torchnlp.metrics import get_moses_multi_bleu


hypotheses = ["The brown fox jumps over the dog 笑"]
references = ["The quick brown fox jumps over the lazy dog 笑"]

# Compute BLEU score with the official BLEU perl script
# bleu=get_moses_multi_bleu(hypotheses, references, lowercase=True)  # RETURNS: 47.9
# print(bleu)


s = "The quick brown fox jumps over the lazy dog 笑 o o"
l=s.split(" ")
l.remove("o")
print(l)
