import json
from copy import deepcopy

import argparse
import rouge
import logging
import sys

if __name__ == "__main__":
    logFormatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s]: %(levelname)s : %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Enable console logging
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    
    parser = argparse.ArgumentParser(description="Compare predictions form multiple files")
    parser.add_argument('-i', '--input_files', dest="input_files", metavar='CONFIG_FILE', type=str, nargs='+',
                        help='Input file paths with short name. Ex. SHORT_NAME_0:FILE_PATH',
                        required=True)

    parser.add_argument('-o', '--output_file', dest="output_file", metavar='NEW_CONFIG_FILE', type=str, default=None,
                        help='File to save the output')

    args = parser.parse_args()

    delim = ":"
    input_files = args.input_files

    logging.info("Input files:")
    files_to_compare = {}
    for fi, curr_input_file in enumerate(input_files):
        file_name_split = [x.strip() for x in curr_input_file.split(":")]

        short_name = "SHORT_NAME_{0}".format(fi)
        file_path = "NO_FILE_PATH_YET"
        if len(file_name_split) == 2:
            short_name, file_path = tuple(file_name_split)
        else:
            logging.error("input_files[{0}]:Files should be formatted as SHORT_NAME_0:FILE_PATH "
                  "(Ex. model_lstm:predictions_test.json) but it is:\n"
                  "{1}".format(fi, curr_input_file))
            exit(1)

        if short_name in files_to_compare:
            logging.error(
                "input_files[{0}]: Short name {2} is already declared\n"
                "Input:\n"
                "{1}\n"
                "\n"
                "Conflicted with: \n"
                "{2}:{3}".format(fi, curr_input_file, short_name, files_to_compare[short_name]))
            exit(1)

        files_to_compare[short_name] = file_path
        logging.info("{0}:{1}".format(short_name, file_path))

    # output file
    output_file = "compare__{0}.json".format("__".join(list(sorted(list(files_to_compare.keys())))))
    if args.output_file is not None:
        output_file = args.output_file

    # eval predictions
    rouge_evaluator = rouge.Rouge(metrics=["rouge-l"],
                                        max_n=0,
                                        limit_length=True,
                                        length_limit=100,
                                        length_limit_type='words',
                                        apply_avg=False,
                                        apply_best=False,
                                        alpha=0.5,  # Default F1_score
                                        weight_factor=1.2,
                                        stemming=True)
    def get_item_rouge_l(item):
        predicted = item["best_span_str"]
        references = item["meta"]["answer_texts"]
        eval_scores = rouge_evaluator.get_scores(predicted, references)

        score = {"rouge-l": max(eval_scores["rouge-l"][0]["f"])}
        return score

    predictions_merged = {}
    logging.info("\n")
    logging.info("Merging {0} files...".format(files_to_compare))

    fid = 0
    for short_name, file_path in files_to_compare.items():
        fid += 1
        with open(file_path, mode="r") as f_pred:
            for line_id, line in enumerate(f_pred):
                try:
                    pred_item = json.loads(line.strip())

                    if pred_item["id"] not in predictions_merged:
                        new_item_compare = deepcopy(pred_item)
                        predictions_merged[pred_item["id"]] = new_item_compare

                        curr_metrics = get_item_rouge_l(new_item_compare)
                        new_item_compare["compare_metrics"] = {short_name: curr_metrics}
                        new_item_compare["compare_best_spans"] = {short_name: pred_item.get("best_span", None)}
                        new_item_compare["compare_best_span_str"] = {short_name: pred_item.get("best_span_str", None)}

                        if "metrics" in new_item_compare:
                            del new_item_compare["metrics"]
                        del new_item_compare["best_span"]
                        del new_item_compare["best_span_str"]
                        del new_item_compare["span_start_logits"]
                        del new_item_compare["span_start_probs"]
                        del new_item_compare["span_end_logits"]
                        del new_item_compare["span_end_probs"]
                    else:
                        qa_item_compare = predictions_merged[pred_item["id"]]

                        curr_metrics = get_item_rouge_l(pred_item)
                        qa_item_compare["compare_metrics"][short_name] = curr_metrics
                        qa_item_compare["compare_best_spans"][short_name] = pred_item.get("best_span", None)
                        qa_item_compare["compare_best_span_str"][short_name] = pred_item.get("best_span_str", None)

                except Exception as e:
                    logging.error("Error at line {0} in file {1}:".format(line_id, file_path))
                    logging.error(str(e))
                    if "best" in str(e):
                        raise e

                if (line_id+1) % 1000 == 0:
                    logging.info("File {0}/{2} - {1} items processed".format(fid, line_id + 1, len(files_to_compare)))

    for id in list(predictions_merged.keys()):
        if len(predictions_merged[id]["compare_best_span_str"]) < 2:
            del predictions_merged[id]

    logging.info("{0} unique items collected!".format(len(predictions_merged)))

    predictions_sorted = sorted([(key, item) for key, item in predictions_merged.items()], key=lambda x: x[0])
    with open(output_file, mode="w") as f_out:
        for key, item in predictions_sorted:
            f_out.write(json.dumps(item))
            f_out.write("\n")

    logging.info("Comparison file with {1} exported to {0}".format(output_file, len(predictions_merged)))





