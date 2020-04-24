import json
from copy import deepcopy
from os.path import basename

import argparse
import rouge
import logging
import sys

from tools.narrativeqa_eval_generation import get_wh_type

if __name__ == "__main__":
    logFormatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s]: %(levelname)s : %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Enable console logging
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    
    parser = argparse.ArgumentParser(description="Input merged file")
    parser.add_argument('-i', '--input_file', dest="input_file", metavar='CONFIG_FILE', type=str,
                        help='Input file paths with short name. Ex. SHORT_NAME_0:FILE_PATH', required=True)

    parser.add_argument('-o', '--output_file', dest="output_file", metavar='NEW_CONFIG_FILE', type=str, default=None,
                        help='File to save the output', required=True)

    parser.add_argument('-f', '--filter', dest="filter", metavar='filter_name', type=str, default="csv",
                        help='Filter name to use', required=False)
    args = parser.parse_args()

    input_file = args.input_file
    filter = args.filter

    # output file
    output_file = "filtered__".format(basename(input_file))
    if args.output_file is not None:
        output_file = args.output_file

    exp_cnt = 0

    models = ['oracle_full',
              'bidaf',
              'qanet',
              'coref_feats',
              #'sdp_exp_nonexp',
              'sdp_exp',
              #'sdp_ne_nosense',
              'sdp_ne',
              'sentspan3',
              'sdp_exp_nosense',
              'srl_3verbs',
              #'srl_4verbs',
              'multi_srl_sdp_exp',
              'multi_srl_sdp_nonexp',
              'multi_srl_sdp_exp_nonexp',
              'multi_srl_sdp_exp_coref_feats']

    # sort items by id
    with open(input_file, mode="r") as f_pred:
        items = []
        for line_id, line in enumerate(f_pred):
            pred_item = json.loads(line.strip())
            items.append(pred_item)

    items.sort(key=lambda x: x["id"])

    with open(output_file, mode="w") as f_out:
        # Write fields
        f_out.write("id" + "\t")
        f_out.write("context" + "\t")
        f_out.write("WH-type" + "\t")
        f_out.write("question" + "\t")
        f_out.write("answer1" + "\t")
        f_out.write("answer2" + "\t")
        for model_name in models:
            f_out.write("{0}_answer_str".format(model_name) + "\t")
            f_out.write("{0}_score".format(model_name) + "\t")

        for model_name in models:
            f_out.write("{0}_span".format(model_name) + "\t")
        f_out.write("\n")

        # Export items
        for line_id, pred_item in enumerate(items):
            try:
                if pred_item["compare_metrics"].get("qanet", {"rouge-l": 0.0}).get("rouge-l", 0.0) > 1.0:
                    continue

                # filter logic
                if filter == "csv":
                    # context, question, gold_answers
                    f_out.write(pred_item["id"] + "\t")
                    f_out.write(" ".join(pred_item["meta"]["passage_tokens"]).replace("\n", " ").replace("\t", "") + "\t")
                    question_text = " ".join(pred_item["meta"]["question_tokens"])
                    f_out.write(get_wh_type(question_text) + "\t")
                    f_out.write(question_text + "\t")
                    f_out.write(pred_item["meta"]["answer_texts"][0] + "\t")
                    f_out.write(pred_item["meta"]["answer_texts"][1] + "\t")

                    for model_name in models:
                        f_out.write(str(pred_item["compare_best_span_str"].get(model_name, "")).replace("\n", " ").replace("\t", "") + "\t")
                        f_out.write(str(pred_item["compare_metrics"].get(model_name, {"rouge-l":0.0}).get("rouge-l", 0.0)) + "\t")

                    for model_name in models:
                        f_out.write(str(pred_item["compare_best_spans"].get(model_name, [])) + "\t")

                    f_out.write("\n")

                exp_cnt += 1

            except Exception as e:
                logging.exception("Error at line {0} in file {1}:".format(line_id, input_file))
                #logging.error(str(e))
                if "best" in str(e):
                    raise e

            if (line_id+1) % 1000 == 0:
                logging.info("File {0} items processed - {1}".format(line_id + 1, input_file))


    logging.info("Comparison file with {1} exported to {0}".format(output_file, exp_cnt))





