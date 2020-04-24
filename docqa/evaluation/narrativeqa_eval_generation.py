import json
import sys
from sys import argv
import numpy as np

from docqa.evaluation.pycocoevalcap.meteor.meteor import Meteor
from docqa.evaluation.pycocoevalcap.cider.cider import Cider
from docqa.evaluation.pycocoevalcap.rouge.rouge import Rouge
from docqa.evaluation.pycocoevalcap.bleu.bleu import Bleu

meteor_obj = Meteor()
rouge_obj = Rouge()
cider_obj = Cider()
bleu_obj = Bleu(4)

system = sys.argv[1]
system_out = system + ".scores.json"

# load predictions and references
word_target_dict = {}
word_response_dict = {}

with open(system, mode="r") as f_in:
    for i, line in enumerate(f_in):
        item = json.loads(line.strip())

        word_response_dict[i] = [item["best_span_str"]]
        word_target_dict[i] = [item["meta"]["answer_texts"][0], item["meta"]["answer_texts"][1]]


bleu_score, bleu_scores = bleu_obj.compute_score(
        word_target_dict, word_response_dict)
bleu1_score, _, _, bleu4_score = bleu_score
bleu1_scores, _, _, bleu4_scores = bleu_scores

meteor_score, meteor_scores = meteor_obj.compute_score(
        word_target_dict, word_response_dict) 

rouge_score, rouge_scores = rouge_obj.compute_score(
        word_target_dict, word_response_dict) 

cider_score, cider_scores = cider_obj.compute_score(
        word_target_dict, word_response_dict) 

print("ROUGE-L: ", rouge_score)
print("BLEU-1: ", bleu1_score)
print("BLEU-4: ", bleu4_score)
print("METEOR: ", meteor_score)
print("CiDER: ", cider_score)
