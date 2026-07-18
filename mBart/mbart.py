from transformers import pipeline


pipeline = pipeline(
    task="text-generation",
    model="facebook/mbart-large-50-many-to-many-mmt",
    src_lang="en_XX",
    tgt_lang="fr_XX",
    device=0,
)
print(pipeline("UN Chief Says There Is No Military Solution in Syria"))