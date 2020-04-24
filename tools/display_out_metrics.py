import datetime

from typing import List

import argparse
import json
import os
import sys

import glob

display_fields_summary = ["BLEU-1", "BLEU-4", "ROUGE-L", "CiDER"]
display_qt_fields = [
    "QT__ROUGE-L__how",
    "QT__ROUGE-L__how far",
    "QT__ROUGE-L__how long",
    "QT__ROUGE-L__how many",
    "QT__ROUGE-L__how much",
    "QT__ROUGE-L__how old",
    "QT__ROUGE-L__other",
    "QT__ROUGE-L__what",
    "QT__ROUGE-L__when",
    "QT__ROUGE-L__where",
    "QT__ROUGE-L__which",
    "QT__ROUGE-L__who",
    "QT__ROUGE-L__why"
]

display_cnt_fields = [
    "cnt_all",
    "cnt__QT__ROUGE-L__how",
    "cnt__QT__ROUGE-L__how long",
    "cnt__QT__ROUGE-L__how many",
    "cnt__QT__ROUGE-L__how much",
    "cnt__QT__ROUGE-L__how old",
    "cnt__QT__ROUGE-L__other",
    "cnt__QT__ROUGE-L__what",
    "cnt__QT__ROUGE-L__when",
    "cnt__QT__ROUGE-L__where",
    "cnt__QT__ROUGE-L__which",
    "cnt__QT__ROUGE-L__who",
    "cnt__QT__ROUGE-L__why",
]

display_passage_len = [
    "CTX_LEN__ROUGE-L__0000-0200",
    "CTX_LEN__ROUGE-L__0200-0400",
    "CTX_LEN__ROUGE-L__0400-0600",
    "CTX_LEN__ROUGE-L__0600-0800",
    "CTX_LEN__ROUGE-L__0800-1000",
    "CTX_LEN__ROUGE-L__1000-1200",
    "CTX_LEN__ROUGE-L__1201+"
]

display_question_len = [
    "Q_LEN__ROUGE-L__01",
    "Q_LEN__ROUGE-L__02",
    "Q_LEN__ROUGE-L__03",
    "Q_LEN__ROUGE-L__04",
    "Q_LEN__ROUGE-L__05",
    "Q_LEN__ROUGE-L__06",
    "Q_LEN__ROUGE-L__07",
    "Q_LEN__ROUGE-L__08",
    "Q_LEN__ROUGE-L__09",
    "Q_LEN__ROUGE-L__10",
    "Q_LEN__ROUGE-L__11",
    "Q_LEN__ROUGE-L__12",
    "Q_LEN__ROUGE-L__13",
    "Q_LEN__ROUGE-L__14",
    "Q_LEN__ROUGE-L__15",
    "Q_LEN__ROUGE-L__16",
    "Q_LEN__ROUGE-L__17",
    "Q_LEN__ROUGE-L__18+",
]

display_answer_len = [
    "GANSW_LEN__ROUGE-L__01",
    "GANSW_LEN__ROUGE-L__02",
    "GANSW_LEN__ROUGE-L__03",
    "GANSW_LEN__ROUGE-L__04",
    "GANSW_LEN__ROUGE-L__05",
    "GANSW_LEN__ROUGE-L__06",
    "GANSW_LEN__ROUGE-L__07",
    "GANSW_LEN__ROUGE-L__08",
    "GANSW_LEN__ROUGE-L__09",
    "GANSW_LEN__ROUGE-L__10+"
]

display_answer_pos = [
    "GANSW_POS__ROUGE-L__0000-0200",
    "GANSW_POS__ROUGE-L__0200-0400",
    "GANSW_POS__ROUGE-L__0400-0600",
    "GANSW_POS__ROUGE-L__0600-0800",
    "GANSW_POS__ROUGE-L__0800-1000",
    "GANSW_POS__ROUGE-L__1000-1200",
    "GANSW_POS__ROUGE-L__OTHER",
]

display_answer_pos_cnt = [
    "cnt__GANSW_POS__ROUGE-L__0000-0200",
    "cnt__GANSW_POS__ROUGE-L__0200-0400",
    "cnt__GANSW_POS__ROUGE-L__0400-0600",
    "cnt__GANSW_POS__ROUGE-L__0600-0800",
    "cnt__GANSW_POS__ROUGE-L__0800-1000",
    "cnt__GANSW_POS__ROUGE-L__1000-1200",
    "cnt__GANSW_POS__ROUGE-L__OTHER",
]

display_passage_len_cnt = [
    "cnt__CTX_LEN__ROUGE-L__0000-0200",
    "cnt__CTX_LEN__ROUGE-L__0200-0400",
    "cnt__CTX_LEN__ROUGE-L__0400-0600",
    "cnt__CTX_LEN__ROUGE-L__0600-0800",
    "cnt__CTX_LEN__ROUGE-L__0800-1000",
    "cnt__CTX_LEN__ROUGE-L__1000-1200",
    "cnt__CTX_LEN__ROUGE-L__1201+"
]

display_question_len_cnt = [
    "cnt__Q_LEN__ROUGE-L__01",
    "cnt__Q_LEN__ROUGE-L__02",
    "cnt__Q_LEN__ROUGE-L__03",
    "cnt__Q_LEN__ROUGE-L__04",
    "cnt__Q_LEN__ROUGE-L__05",
    "cnt__Q_LEN__ROUGE-L__06",
    "cnt__Q_LEN__ROUGE-L__07",
    "cnt__Q_LEN__ROUGE-L__08",
    "cnt__Q_LEN__ROUGE-L__09",
    "cnt__Q_LEN__ROUGE-L__10",
    "cnt__Q_LEN__ROUGE-L__11",
    "cnt__Q_LEN__ROUGE-L__12",
    "cnt__Q_LEN__ROUGE-L__13",
    "cnt__Q_LEN__ROUGE-L__14",
    "cnt__Q_LEN__ROUGE-L__15",
    "cnt__Q_LEN__ROUGE-L__16",
    "cnt__Q_LEN__ROUGE-L__17",
    "cnt__Q_LEN__ROUGE-L__18+",
]

display_annotations = [
    "ANNO__ROUGE-L__ambiguity__1",
    "ANNO__ROUGE-L__comment__",
    "ANNO__ROUGE-L__multicand__0",
    "ANNO__ROUGE-L__multicand__1",
    "ANNO__ROUGE-L__multisent__0",
    "ANNO__ROUGE-L__multisent__1",
    "ANNO__ROUGE-L__relation__causal relation",
    "ANNO__ROUGE-L__relation__coreference",
    "ANNO__ROUGE-L__relation__none",
    "ANNO__ROUGE-L__relation__spatial/temporal",
    "ANNO__ROUGE-L__skill__knowledge reasoning",
    "ANNO__ROUGE-L__skill__meta/whole reasoning",
    "ANNO__ROUGE-L__skill__paraphrasing",
    "ANNO__ROUGE-L__skill__word matching",
    "ANNO__ROUGE-L__subset__easy",
    "ANNO__ROUGE-L__subset__hard",
    "ANNO__ROUGE-L__validity__1",
]

display_annotations_cnt = [
    "cnt__ANNO__ROUGE-L__ambiguity__1",
    "cnt__ANNO__ROUGE-L__comment__",
    "cnt__ANNO__ROUGE-L__multicand__0",
    "cnt__ANNO__ROUGE-L__multicand__1",
    "cnt__ANNO__ROUGE-L__multisent__0",
    "cnt__ANNO__ROUGE-L__multisent__1",
    "cnt__ANNO__ROUGE-L__relation__causal relation",
    "cnt__ANNO__ROUGE-L__relation__coreference",
    "cnt__ANNO__ROUGE-L__relation__none",
    "cnt__ANNO__ROUGE-L__relation__spatial/temporal",
    "cnt__ANNO__ROUGE-L__skill__knowledge reasoning",
    "cnt__ANNO__ROUGE-L__skill__meta/whole reasoning",
    "cnt__ANNO__ROUGE-L__skill__paraphrasing",
    "cnt__ANNO__ROUGE-L__skill__word matching",
    "cnt__ANNO__ROUGE-L__subset__easy",
    "cnt__ANNO__ROUGE-L__subset__hard",
    "cnt__ANNO__ROUGE-L__validity__1",
]


display_answer_len_cnt = [
    "cnt__GANSW_LEN__ROUGE-L__01",
    "cnt__GANSW_LEN__ROUGE-L__02",
    "cnt__GANSW_LEN__ROUGE-L__03",
    "cnt__GANSW_LEN__ROUGE-L__04",
    "cnt__GANSW_LEN__ROUGE-L__05",
    "cnt__GANSW_LEN__ROUGE-L__06",
    "cnt__GANSW_LEN__ROUGE-L__07",
    "cnt__GANSW_LEN__ROUGE-L__08",
    "cnt__GANSW_LEN__ROUGE-L__09",
    "cnt__GANSW_LEN__ROUGE-L__10+"
]

profiles = {
    "narrativeqa_qt": {
        "config": ["dataset_reader->type", "model->type"],
        "metrics":
            ["dev__"+x for x in display_fields_summary + display_qt_fields] +
            ["test__" + x for x in display_fields_summary + display_qt_fields]
    },
    "narrativeqa_all": {
        "config": ["dataset_reader->type", "model->type"],
        "metrics":
            ["dev__"+x for x in display_fields_summary + display_qt_fields + display_passage_len + display_answer_len + display_question_len + display_cnt_fields + display_answer_pos + display_answer_pos_cnt + display_annotations + display_annotations_cnt] +
            ["test__" + x for x in display_fields_summary + display_qt_fields + display_passage_len + display_answer_len + display_question_len + display_cnt_fields + display_answer_pos + display_answer_pos_cnt + display_annotations + display_annotations_cnt]
    },
    "squad_all": {
        "config": ["dataset_reader->type", "model->type"],
        "metrics": []
    }
}

profiles["squad_all"]["metrics"] = [x.replace("ROUGE-L", "SQ_EM") for x in profiles["narrativeqa_all"]["metrics"]] \
                                   + [x.replace("ROUGE-L", "SQ_F1") for x in profiles["narrativeqa_all"]["metrics"]]

profiles["squad_all"]["metrics"] = [x.replace("test__", "test2__", 1) for x in profiles["squad_all"]["metrics"] if x.startswith("test__")]\
                                   + profiles["squad_all"]["metrics"]

date_format = "%Y-%m-%d-%H-%M-%S"
def get_val_by_hier_key(json_item,
            hier_key:List[str],
            raise_key_error=False,
            default=None):
    """
    Gets a value of hierachical json fields. Does not support lists!
    :param json_item: Item to get the values from
    :param hier_key: List of hierarchical keys
    :param raise_key_error: Should it raise error on missing keys or return default field
    :param default: Default value if no key is found
    :return: Retrieved or Default value if no error is raised
    """
    curr_obj = json_item
    res_val = default

    found = True
    for fld in hier_key:
        if not fld in curr_obj:
            found = False
            if raise_key_error:
                raise KeyError("Key {0} not found in object json_item. {1}".format("->".join(["*%s*" % x if x == fld else x  for x in hier_key]),
                                                                                   "Starred item is where hierarchy lookup fails!" if len(hier_key) > 1 else "" ))
            break
        curr_obj = curr_obj[fld]

    if found:
        res_val = curr_obj

    return res_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Display metrics results')

    parser.add_argument('-i', '--input_pattern', dest="input_pattern", metavar='CONFIG_FILE', type=str,
                        help='Input', default="_output/*")

    parser.add_argument('-f', '--fields', dest="fields", nargs='+', metavar='best_epoch;training_loss;best_validation_rouge-4;test_rouge-l',
                        type=str,
                        help='fields to display', default=None)

    parser.add_argument('-cf', '--config_fields', dest="config_fields", nargs='+', metavar='best_epoch;training_loss;best_validation_rouge-4;test_rouge-l',
                        type=str,
                        help='fields to display', default=["dataset_reader->type", "model->type"])

    parser.add_argument('-p', '--profile', dest="profile", metavar='narrativeqa', choices=list(profiles.keys()),
                        type=str,
                        help='profile with predefined fields', default=None)

    parser.add_argument('--skip_empty', dest="skip_empty", metavar='True', type=str,
                        help='profile with predefined fields', default="True")


    args = parser.parse_args()

    skip_empty = args.skip_empty == "True"
    input_pattern = args.input_pattern

    profile = args.profile
    profile_settings = {}
    if profile is not None:
        profile_settings = profiles[profile]

    # metric fields
    metrics_fields = profile_settings.get("metrics", [])
    metrics_fields_input = args.fields if args.fields is not None else []
    for field in metrics_fields_input:
        if field not in metrics_fields:
            metrics_fields.append(field)

    if len(metrics_fields) == 0:
        print("No metrics fields to display!")
        print("You need to specify either a field profile (ex. --profile=narrativeqa)"
              " or a list of fields (ex. --fields best_epoch best_validation_rouge-4 test_rouge-l)!")

        exit(1)

    # config fields
    default_fields = ["serialization_dir",
                      "start_time",
                      "metrics_file",
                      "metrics_time",
                      "duration",
                      "serialization_dir_base"]

    dirs = glob.glob(input_pattern, recursive=False)
    dirs.sort(key=os.path.getctime)

    print("Metrics for: {0}".format(input_pattern))
    report_results_list = []
    for dir in dirs:
        dir = os.path.abspath(dir)
        report_json = {"serialization_dir": dir,
                       "serialization_dir_base": os.path.basename(dir)}

        dir_creation_time = datetime.datetime.fromtimestamp(os.path.getctime(dir))
        report_json["start_time"] = datetime.datetime.strftime(dir_creation_time, date_format)

        # metrics file
        metrics_json_all = {}

        # load dev
        metrics_file = os.path.join(dir, "predictions_dev.json.metrics")
        if not os.path.exists(metrics_file):
            metrics_file = os.path.join(dir, "predictions_dev.json.json.metrics")
        if not os.path.exists(metrics_file):
            metrics_file = os.path.join(dir, "predictions_dev.json.metrics.json")

        if os.path.exists(metrics_file):
            prefix = "dev__"
            metrics_json_dev = {}
            with open(metrics_file, mode="r") as f_metrics:
                metrics_json_dev = json.load(f_metrics)

            for k,v in metrics_json_dev.items():
                metrics_json_all[prefix + k] = v

        # load test
        metrics_file = os.path.join(dir, "predictions_test.json.metrics")
        # if not os.path.exists(metrics_file):
        #     metrics_file = os.path.join(dir, "predictions_test.json.json.metrics")
        # if not os.path.exists(metrics_file):
        #     metrics_file = os.path.join(dir, "predictions_test.json.metrics.json")

        if os.path.exists(metrics_file):
            prefix = "test__"
            metrics_json_test = {}
            with open(metrics_file, mode="r") as f_metrics:
                metrics_json_test = json.load(f_metrics)

            for k, v in metrics_json_test.items():
                metrics_json_all[prefix + k] = v

            # load metrics fields:
            for field in metrics_fields:
                report_json[field] = str(metrics_json_all.get(field, ""))

        # load test2
        metrics_file = os.path.join(dir, "predictions_test2.json.metrics")

        if os.path.exists(metrics_file):
            prefix = "test2__"
            metrics_json_test = {}
            with open(metrics_file, mode="r") as f_metrics:
                metrics_json_test = json.load(f_metrics)

            for k, v in metrics_json_test.items():
                metrics_json_all[prefix + k] = v

            # load metrics fields:
            for field in metrics_fields:
                report_json[field] = str(metrics_json_all.get(field, ""))

        report_results_list.append(report_json)

    # Print all results
    display_fields = default_fields + metrics_fields

    print("-" * 10)
    print("\t".join(display_fields))

    for report_item in report_results_list:
        print("\t".join([report_item.get(fld, "") for fld in display_fields]))

