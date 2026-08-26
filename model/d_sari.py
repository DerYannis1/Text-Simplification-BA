"""
Quelle:
    Sun, R., Jin, H., & Wan, X. (2021). Document-Level Text Simplification:
    Dataset, Metric, and Model. EMNLP 2021.
    D_SARI.py aus:  https://github.com/RLSNLP/Document-level-text-simplification/blob/main/D_SARI.py

    
Änderungen:
    - Satztrennung angepasst, da NLTK im Deutschen nicht wie im Original funktioniert. Hier mit einfachen Regex.
    - Division durch Null bei SLP abgefangen, falls Vorhersage oder Referenz aus null Saetzen bestehen.
"""

from collections import Counter
import math
import re

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text):
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p for p in parts if p]


def _d_sari_ngram(sgrams, cgrams, rgramslist, numref):
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)

    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter({g: c * numref for g, c in sgramcounter.items()})
    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter({g: c * numref for g, c in cgramcounter.items()})

    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keeptmpscore1 = sum(keepgramcountergood_rep[g] / keepgramcounter_rep[g] for g in keepgramcountergood_rep)
    keeptmpscore2 = sum(keepgramcountergood_rep[g] / keepgramcounterall_rep[g] for g in keepgramcountergood_rep)

    keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep) if keepgramcounter_rep else 0
    keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep) if keepgramcounterall_rep else 0
    keepscore = 0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)

    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    delgramcounterall_rep = sgramcounter_rep - rgramcounter
    deltmpscore1 = sum(delgramcountergood_rep[g] / delgramcounter_rep[g] for g in delgramcountergood_rep)
    delscore_precision = deltmpscore1 / len(delgramcounter_rep) if delgramcounter_rep else 0
    # HINWEIS: die Original-Implementierung berechnet delscore (F1) UND
    # delscore_recall, gibt am Ende aber nur delscore_precision zurueck --
    # das wird hier bewusst 1:1 uebernommen (nicht "korrigiert"), damit die
    # Werte mit anderen Arbeiten, die dieselbe Referenzimplementierung
    # nutzen, vergleichbar bleiben.

    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = addgramcounter & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)
    addtmpscore = len(addgramcountergood)
    addscore_precision = addtmpscore / len(addgramcounter) if addgramcounter else 0
    addscore_recall = addtmpscore / len(addgramcounterall) if addgramcounterall else 0
    addscore = 0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)

    return keepscore, delscore_precision, addscore


def _make_ngrams(words, n):
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)] if len(words) >= n else []


def d_sari_sent(source, prediction, references):
    numref = len(references)
    s1 = source.lower().split(" ")
    c1 = prediction.lower().split(" ")
    r1_list = [r.lower().split(" ") for r in references]

    keep_scores, del_scores, add_scores = [], [], []
    for n in range(1, 5):
        sgrams = _make_ngrams(s1, n)
        cgrams = _make_ngrams(c1, n)
        rgramslist = [_make_ngrams(r, n) for r in r1_list]
        k, d, a = _d_sari_ngram(sgrams, cgrams, rgramslist, numref)
        keep_scores.append(k)
        del_scores.append(d)
        add_scores.append(a)

    avg_keep = sum(keep_scores) / 4
    avg_del = sum(del_scores) / 4
    avg_add = sum(add_scores) / 4

    input_length = len(s1)
    output_length = len(c1)
    reference_length = int(sum(len(r) for r in r1_list) / numref)

    output_sent_n = len(_split_sentences(prediction))
    reference_sent_n = int(sum(len(_split_sentences(r)) for r in references) / numref)

    LP_1 = 1 if output_length >= reference_length else math.exp(
        (output_length - reference_length) / output_length)
    LP_2 = 1 if output_length <= reference_length else math.exp(
        (reference_length - output_length) / max(input_length - reference_length, 1))
    SLP = math.exp(-abs(reference_sent_n - output_sent_n) / max(reference_sent_n, output_sent_n, 1))

    avg_keep = avg_keep * LP_2 * SLP
    avg_add = avg_add * LP_1
    avg_del = avg_del * LP_2

    final_score = (avg_keep + avg_del + avg_add) / 3
    return final_score, avg_keep, avg_del, avg_add