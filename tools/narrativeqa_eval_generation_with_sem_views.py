import itertools
import os

import time

import argparse
import json
import sys
from sys import argv
import numpy as np

from allennlp.tools.squad_eval import normalize_answer
from tools.pycocoevalcap.meteor.meteor import Meteor
from tools.pycocoevalcap.cider.cider import Cider
from tools.pycocoevalcap.rouge.rouge import Rouge
from tools.pycocoevalcap.bleu.bleu import Bleu
from tools.pycocoevalcap.squad.squad_em import SquadEM
from tools.pycocoevalcap.squad.squad_f1 import SquadF1
from tools.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from tools.qa_annotations import get_features_from_annotaiton

METEOR_NAME = "METEOR"
BLEU_NAME = "BLEU"
ROUGE_L_NAME = "ROUGE-L"
CIDER_NAME = "CiDER"
EM_NAME = "SQ_EM"
F1_NAME = "SQ_F1"

wh_words = ["what", "when", "where", "which", "who", "whom", "whose", "why", "how far", "how long", "how many", "how much", "how old", "how"]
question_words_dict = {}
use_defined_q_words = True
if use_defined_q_words:
    question_words_dict = {x: 0 for x in wh_words}
    question_words_dict["other"] = 0


def get_wh_type(question):
    quest_lower = question.lower()
    question_type = None
    for qw in wh_words:
        if qw in quest_lower:
            question_type = qw

            break

    if question_type is None:
        question_type = "other"

    return question_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate items with Cider, Rouge and Bleu')

    parser.add_argument('-i', '--input_file', dest="input_file", metavar='Input file', type=str,
                        help='Input predictions file')

    parser.add_argument('-o', '--output_file', dest="output_file", metavar='Output metrics file', type=str,
                        default=None, help='Output metrics file')

    parser.add_argument('-a', '--annotation_file', dest="annotation_file", metavar='annotation_file', type=str,
                        default=None, help='Annotation files')

    parser.add_argument('--use_meteor', dest="use_meteor", metavar='False', type=str,
                        help='profile with predefined fields', default="False")

    parser.add_argument('--for_analysis', dest="for_analysis", metavar='False', type=str,
                        help='profile with predefined fields', default="False")

    args = parser.parse_args()

    export_for_analysis = args.for_analysis == "True"
    use_meteor = args.use_meteor == "True"

    # evaluators
    evaluators = []
    evaluators.append({"name": EM_NAME, "evaluator": SquadEM()})
    evaluators.append({"name": F1_NAME, "evaluator": SquadF1()})
    evaluators.append({"name": ROUGE_L_NAME, "evaluator": Rouge()})
    evaluators.append({"name": CIDER_NAME, "evaluator": Cider()})
    evaluators.append({"name": BLEU_NAME, "evaluator": Bleu(4),
                       "scores_names": [BLEU_NAME + "-%s" % (x + 1) for x in range(4)]})

    if use_meteor:
        evaluators.append({"name": METEOR_NAME, "evaluator": Meteor()})

    # reads annotations form https://github.com/Alab-NII/mrc-heuristics
    annotation_file_prefix = "ANNO"
    annotation_file = args.annotation_file
    if annotation_file is not None and len(annotation_file) > 0 and "->" in annotation_file:
        annotation_file_prefix, annotation_file = tuple([x.strip() for x in annotation_file.split("->")])

    key2annotation = {}
    if annotation_file is not None and len(annotation_file) > 0:
        key2annotation = json.load(open(annotation_file))
        key2annotation = {k: get_features_from_annotaiton(v) for k,v in key2annotation.items()}

    predictions_file_path = args.input_file
    featid2name_file_name = predictions_file_path+".id2featname"
    featid2name_map = {}
    if os.path.exists(featid2name_file_name):
        featid2name_map = json.load(open(featid2name_file_name))

    system_out = predictions_file_path + ".metrics" if args.output_file is None else args.output_file

    # load predictions and references
    word_target_dict_prepare = {}
    word_response_dict_prepare = {}

    tokenizer = PTBTokenizer()
    def prepare_text_list(text_list):
        texts_dict = {i: [{"caption": x}] for i, x in enumerate(text_list)}
        prepared = tokenizer.tokenize(texts_dict)

        prepared_list = [prepared[i][0] for i in range(len(text_list))]
        return prepared_list


    def group_features_by_token(feats_per_view):
        tokens_feats = []
        for tkn_id in range(len(feats_per_view[0])):
            tkn_feats = [view_token_labels[tkn_id] for view_token_labels in feats_per_view]
            tokens_feats.append(tkn_feats)
        return tokens_feats

    def eval_token_features_accuracy(tokens, token_features, gold_tokens):
        feat_values = {}

        gold_tokens_normalized = [x.lower() for x in gold_tokens]
        gold_tokens_normalized = normalize_answer(" ".join(gold_tokens_normalized)).split()
        for tkn, tkn_feats in zip(tokens, token_features):
            tkn_normalized = normalize_answer(tkn)
            if len(tkn_normalized) > 0:
                tkn_val = 0.0
                if tkn_normalized in gold_tokens_normalized:
                    tkn_val = 1.0
                for tkn_feat in tkn_feats:
                    if tkn_feat in feat_values:
                        feat_values[tkn_feat].append(tkn_val)
                    else:
                        feat_values[tkn_feat] = [tkn_val]

        feat_values_reduced = {k: sum(v)/len(v) for k,v in feat_values.items()}
        return feat_values_reduced

    items_meta = []
    with open(predictions_file_path, mode="r") as f_in:
        for i, line in enumerate(f_in):
            item = json.loads(line.strip())

            item_meta = {}
            item_id = item.get("id", i)
            item_meta["id"] = item_id
            item_meta["question_text"] = " ".join(item["meta"]["question_tokens"])
            item_meta["question_len"] = len(item["meta"]["question_tokens"])
            item_meta["passage_len"] = len(item["meta"]["passage_tokens"])
            item_meta["best_span_str"] = item["best_span_str"]
            item_meta["best_span"] = item["best_span"]
            item_meta["gold_span"] = item["meta"]["token_spans"][0]
            item_meta["annotations"] = []
            best_span_semantic_features = item.get("best_span_semantic_features", None)
            best_span_tokens = item.get("best_span_tokens", None)
            item_meta["best_span_tokens"] = best_span_tokens

            if best_span_semantic_features is not None:
                best_span_semantic_features = group_features_by_token(best_span_semantic_features)
                tokens_list_feats = []

                for tkn, tkn_feats in zip(best_span_tokens, best_span_semantic_features):
                    tkn_feats_distinct = list(set([featid2name_map.get(str(x), str(x)).replace("_B-", "").replace("_I-", "")
                                                   for x in tkn_feats if x != 0]))
                    tokens_list_feats.append(tkn_feats_distinct)

                best_span_semantic_features = tokens_list_feats

            item_meta["best_span_semantic_features"] = best_span_semantic_features

            if len(key2annotation) > 0:
                if isinstance(item_id, str):
                    # item id is expected to be "DOC_ID##Q_ID"
                    doc_id, q_id = tuple(item_id.split("##"))
                    q_id = int(q_id)
                    annotation_id = "{0}-{1}".format(doc_id, q_id)
                    curr_annotations = key2annotation.get(annotation_id, [])
                    item_meta["annotations"] = curr_annotations

            if export_for_analysis:
                item_meta["passage_text"] = " ".join(item["meta"]["passage_tokens"])
                item_meta["gold_span_2"] = item["meta"]["token_spans"][-1]

            item_meta["gold_answers_texts"] = item["meta"]["answer_texts"]

            # classification groups
            item_meta["q_wh_type"] = get_wh_type(item_meta["question_text"])

            items_meta.append(item_meta)

            word_response_dict_prepare[i] = [{"caption": item["best_span_str"].lower()}]
            word_target_dict_prepare[i] = [{"caption": x.lower()} for x in item["meta"]["answer_texts"]]


    print("Tokenize with PTBTokenizer:")
    start = time.time()
    word_response_dict = tokenizer.tokenize(word_response_dict_prepare)
    word_target_dict = tokenizer.tokenize(word_target_dict_prepare)

    print("Tokenization of {0} items done in {1}".format(len(word_response_dict), time.time() - start))

    # eval
    overall_metrics_dict = {}

    use_scorer_summary_score = True  # if False, the score is averaged across all. However for BLEU2-,4 FALSE is not correct!
    metrics_per_item = [{} for x in range(len(word_response_dict))]
    for evaluator in evaluators:
        evaluator_obj = evaluator["evaluator"]
        eval_name = evaluator["name"]
        scores_names = evaluator.get("scores_names", [])

        eval_score, eval_scores = evaluator_obj.compute_score(
            word_target_dict, word_response_dict)

        if isinstance(eval_score, list):
            if len(scores_names) != len(eval_score):
                raise ValueError("eval_name:{0} - `eval_socre` contains {1} scores "
                                 "but `scores_suffixes` only has {2} values ".format(eval_name, len(eval_score),
                                                                                     len(scores_names)))

            for score, score_list, score_name in zip(eval_score, eval_scores, scores_names):
                if use_scorer_summary_score:
                    overall_metrics_dict[score_name] = score
                else:
                    # old evaluation. Not correct for BLEU
                    overall_metrics_dict[score_name] = sum(score_list)/len(score_list)

        else:
            if use_scorer_summary_score:
                overall_metrics_dict[eval_name] = eval_score
            else:
                # old evaluation. Not correct for BLEU
                overall_metrics_dict[eval_name] = np.mean(eval_scores)


        # scores per item
        for item_id in range(len(word_response_dict)):
            item_metrics = metrics_per_item[item_id]
            if isinstance(eval_scores, list):
                for score_id in range(len(eval_scores)):
                    scores = eval_scores[score_id]
                    score_name = scores_names[score_id]

                    item_metrics[score_name] = scores[item_id]
            else:
                item_metrics[eval_name] = eval_scores[item_id]



    def add_to_key_list(metrics_dict, key, value):
        if key not in metrics_dict:
            metrics_dict[key] = []

        metrics_dict[key].append(value)

    def create_key_interval_map(min, max, step):
        metrics_intervals = [(x, x + step) for x in range(min, max, step)]
        metrics_intervals = metrics_intervals[:-1] + [(metrics_intervals[-1][0], 100000)]

        fdigits = str(len(str(max)))
        if step > 1:
            metrics_intervals_dict = [("{{0:0{0}d}}-{{1:0{0}d}}".format(fdigits).format(x[0], x[1]) if xid < len(metrics_intervals) - 1
                                       else "{{0:0{0}d}}+".format(fdigits).format(x[0] + 1),
                                       x) for xid, x in enumerate(metrics_intervals)]
        else:
            metrics_intervals_dict = [("{{1:0{0}d}}".format(fdigits).format(x[0], x[1]) if xid < len(metrics_intervals) - 1
                                       else "{{0:0{0}d}}+".format(fdigits).format(x[0] + 1),
                                       x) for xid, x in enumerate(metrics_intervals)]
        
        return metrics_intervals_dict

    passage_length_metrics_range = create_key_interval_map(0, 1400, 200)
    answer_position_metrics_range = create_key_interval_map(0, 1400, 200)
    answer_length_metrics_range = create_key_interval_map(0, 10, 1)
    question_length_metrics_range = create_key_interval_map(0, 18, 1)

    def get_key_by_range_val(tuple_with_key_and_range, val, other_key="OTHER"):
        for range_key, curr_range in tuple_with_key_and_range:
            if val > curr_range[0] and val <= curr_range[1]:
                return range_key
        
        return other_key
        
    # items with meta
    out_analysis_file_name = None
    out_analysis_file_handler = None
    if export_for_analysis:
        out_analysis_file_name = system_out + ".per_item.json"
        out_analysis_file_handler = open(out_analysis_file_name, mode="w")

    for item_id, item_meta in enumerate(items_meta):
        item_meta["system_answer_tokens"] = word_response_dict[item_id][0].split()
        item_meta["system_answer_len"] = len(item_meta["system_answer_tokens"])
        item_meta["system_answer_start"] = item_meta.get("best_span", [-1, -1])[0]
        item_meta["gold_answers_tokens"] = [x.split() for x in word_target_dict[item_id]]
        item_meta["gold_answers_len_min"] = min([len(x) for x in item_meta["gold_answers_tokens"]])
        item_meta["gold_answer_start"] = item_meta.get("gold_span", [-1, -1])[0]

        curr_item_metrics = metrics_per_item[item_id]
        item_meta["metrics"] = curr_item_metrics

        # metrics per question type
        for metric, val in curr_item_metrics.items():
            add_to_key_list(overall_metrics_dict, "QT__{0}__{1}".format(metric, item_meta["q_wh_type"]), val)

        # metrics per length
        for metric, val in curr_item_metrics.items():
            # passage len
            passage_len_val = item_meta["passage_len"]
            passage_len_range_key = get_key_by_range_val(passage_length_metrics_range, passage_len_val)
            add_to_key_list(overall_metrics_dict, "CTX_LEN__{0}__{1}".format(metric, passage_len_range_key), val)
            
            # quesiton len
            answer_len_val = item_meta["gold_answers_len_min"]
            answer_len_range_key = get_key_by_range_val(answer_length_metrics_range, answer_len_val)
            add_to_key_list(overall_metrics_dict, "GANSW_LEN__{0}__{1}".format(metric, answer_len_range_key), val)
            
            # gold question len
            question_len_val = item_meta["question_len"]
            question_len_range_key = get_key_by_range_val(question_length_metrics_range, question_len_val)
            add_to_key_list(overall_metrics_dict, "Q_LEN__{0}__{1}".format(metric, question_len_range_key), val)

            # gold answer location
            answer_pos_val = item_meta["gold_answer_start"]
            answer_pos_range_key = get_key_by_range_val(answer_position_metrics_range, answer_pos_val)
            add_to_key_list(overall_metrics_dict, "GANSW_POS__{0}__{1}".format(metric, answer_pos_range_key), val)

            # add annotations
            for feat in item_meta["annotations"]:
                add_to_key_list(overall_metrics_dict, "{0}__{1}__{2}".format(annotation_file_prefix, metric, feat), val)

        # accuracy per feat type
        if item_meta["best_span_semantic_features"] is not None:
            gold_answers_tokens = list(itertools.chain.from_iterable(item_meta["gold_answers_tokens"]))
            semantic_view_values = eval_token_features_accuracy(item_meta["best_span_tokens"],
                                                                item_meta["best_span_semantic_features"],
                                                                gold_answers_tokens)

            for feat, value in semantic_view_values.items():
                metric = "ACC"
                add_to_key_list(overall_metrics_dict, "{0}__{1}__{2}".format("SEM_VIEW", metric, feat), value)
                add_to_key_list(overall_metrics_dict, "{0}__{1}__{2}__{3}".format("QT__SEM_VIEW", metric, item_meta["q_wh_type"], feat), value)
                if value > 0:
                    add_to_key_list(overall_metrics_dict, "{0}__{1}__{2}__{3}".format("NonZero_QT__SEM_VIEW", metric, item_meta["q_wh_type"], feat), value)

        if out_analysis_file_handler is not None:
            out_analysis_file_handler.write(json.dumps(item_meta))
            out_analysis_file_handler.write("\n")

    if out_analysis_file_handler is not None:
        out_analysis_file_handler.close()
        print("Analysis file exported to \n {0}".format(out_analysis_file_name))
    # reduce mean
    for k in list(overall_metrics_dict.keys()):
        v = overall_metrics_dict[k]
        if isinstance(v, list):
            v_mean = 0.0 if len(v) == 0 else sum(v)/float(len(v))
            overall_metrics_dict[k] = v_mean

            overall_metrics_dict["cnt__"+k] = len(v)

    overall_metrics_dict["cnt_all"] = len(items_meta)

    overall_metrics_names = sorted(list(overall_metrics_dict.keys()), key=lambda aaa: aaa)
    print()
    print("Fields:")
    for key in overall_metrics_names:
        print("\"{0}\",".format(key))

    print()
    # print metrics
    print(json.dumps(overall_metrics_dict, sort_keys=True, indent=4))
    with open(system_out, mode="w") as out_file:
        out_file.write(json.dumps(overall_metrics_dict, indent=4, sort_keys=True))

    print("Metrics saved to {0}".format(system_out))


