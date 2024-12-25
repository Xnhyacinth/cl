

from datasets import load_metric

sari = load_metric("src/llamafactory/train/cl/sari")

# from evaluate import load
# sari = load("sari")
sources=["About 95 species are currently accepted."]
predictions=["About 95 you now get in."]
references=[["About 95 species are currently known.","About 95 species are now accepted.","95 species are now accepted."]]
sari_score = sari.compute(sources=sources, predictions=predictions, references=references)
print(sari_score)