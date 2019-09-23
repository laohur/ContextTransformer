# https://codeload.github.com/vikasnar/Bleu/

import sys
import codecs
import os
import math
import operator
import json
from functools import reduce


def fetch_data(cand, ref):
    """ Store each reference and candidate sentences as a list """
    references = []
    if '.txt' in ref:
        reference_file = codecs.open(ref, 'r', 'utf-8')
        references.append(reference_file.readlines())
    else:
        for root, dirs, files in os.walk(ref):
            for f in files:
                reference_file = codecs.open(os.path.join(root, f), 'r', 'utf-8')
                references.append(reference_file.readlines())
    candidate_file = codecs.open(cand, 'r', 'utf-8')
    candidate = candidate_file.readlines()
    return candidate, references


def count_dict(words, n):
    assert isinstance(words, list)
    ngram_d = {}
    limits = len(words) - n + 1
    # loop through the sentance consider the ngram length
    for i in range(limits):
        ngram = ' '.join(words[i:i + n])
        if ngram in ngram_d.keys():
            ngram_d[ngram] += 1
        else:
            ngram_d[ngram] = 1
    return ngram_d


def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for ref_words in references:
            # ref_sentence = ref_words[si]
            # ref_sentence = ref_words
            ngram_d = count_dict(ref_words, n)
            # ngram_d = {}
            # if (isinstance(ref_sentence, str)):
            #     words = ref_sentence.strip().split()
            # else:
            #     words = ref_sentence
            # if (len(words) == 0):
            #     print("len(ref_sentence)==0  ", ref_sentence)
            #     return 0;
            ref_lengths.append(len(ref_words))
            # limits = len(words) - n + 1
            # # loop through the sentance consider the ngram length
            # for i in range(limits):
            #     ngram = ' '.join(words[i:i + n])
            #     if ngram in ngram_d.keys():
            #         ngram_d[ngram] += 1
            #     else:
            #         ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # candidate
        cand_words = candidate[si]
        cand_dict = count_dict(cand_words, n)
        # cand_dict = {}
        # words = cand_sentence.strip().split()
        # if (len(words) == 0):
        #     print("len(cand_sentence)==0  ", cand_sentence)
        #     return 0
        limits = len(cand_words) - n + 1
        # for i in range(0, limits):
        #     ngram = ' '.join(words[i:i + n]).lower()
        #     if ngram in cand_dict:
        #         cand_dict[ngram] += 1
        #     else:
        #         cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(cand_words))
        c += len(cand_words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l - ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l - ref) < least_diff:
            least_diff = abs(cand_l - ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1 - (float(r) / c))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def bleu(candidate, references):
    if (candidate == None or references == None or len(candidate) == 0 or len(references) == 0):
        print(candidate, references)
        return 0
    precisions = []
    for i in range(4):
        pr, bp = count_ngram(candidate, references, i + 1)
        precisions.append(pr)
    score = geometric_mean(precisions) * bp
    return score


if __name__ == "__main__":
    # candidate, references = fetch_data(sys.argv[1], sys.argv[2])
    # hypotheses, references = fetch_data("candidate.txt", "testSet")
    # hypotheses = ["It is a guide to action which ensures that the military always obeys the commands of the party."]
    # references = ["It is a guide to action that ensures that the military will forever heed Party commands.","It is the guiding principle which guarantees the military forces always being under the command of the Party.","It is the practical guide for the army always to heed the directions of the party."]
    hypotheses = "The brown fox jumps over the dog 笑"
    references = "The quick brown fox jumps over the lazy dog 笑"
    # hypotheses = []
    # references = ["& ;"]
    # hypotheses = ["The brown fox jumps over the dog 笑", "The brown fox jumps over the dog 2 笑"]
    # references = ["The quick brown fox jumps over the lazy dog 笑", "The quick brown fox jumps over the lazy dog 笑"]
    bleu_score = bleu([hypotheses.strip().split()], [references.strip().split()])
    print(bleu_score)
    out = open('bleu_score.txt', 'w')
    out.write(str(bleu))
    out.close()
