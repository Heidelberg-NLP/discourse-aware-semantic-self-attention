import datetime

from typing import List

import argparse
import json
import os
import sys
import re
import glob

from allennlp.common import Params


PARAM_EVAL_FILE_DEV = "eval_file_dev"  # "predictions_dev.json"
REGEX_EVAL_FILE_DEV = "predictions_dev.json"
PARAM_EVAL_FILE_TEST = "eval_file_test"  # "predictions_test.json"
REGEX_EVAL_FILE_TEST = "predictions_test.json"
PARAM_EVAL_METRICS_DEV = "eval_metrics_dev"  # "predictions_dev.json.*metrics.*"
REGEX_EVAL_METRICS_DEV = "predictions_dev.json.*metrics.*"
PARAM_EVAL_METRICS_TEST = "eval_metrics_test"  # "predictions_test.json.*metrics.*"
REGEX_EVAL_METRICS_TEST = "predictions_test.json.*metrics.*"

profiles = {
    "narrativeqa": {
        "config": ["dataset_reader->type", "model->type", "are_train_and_val_semantic_views_same"],
        "metrics": ["best_epoch", "training_loss", "best_validation_rouge-l", "test_rouge-l"],
        "other": [PARAM_EVAL_FILE_DEV, PARAM_EVAL_FILE_TEST, PARAM_EVAL_METRICS_DEV, PARAM_EVAL_METRICS_TEST]
    },
    "entailment": {
        "config": ["dataset_reader->type", "model->type"],
        "metrics": ["best_epoch", "training_loss", "best_validation_MACRO_F1", "test_MACRO_F1"],
        "other": [PARAM_EVAL_FILE_DEV, PARAM_EVAL_FILE_TEST, PARAM_EVAL_METRICS_DEV, PARAM_EVAL_METRICS_TEST]
    }
}

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
                raise KeyError("Key {0} not found in object json_item. {1}".format("->".join(["*%s*" % x if x==fld else x  for x in hier_key]),
                                                                                   "Starred item is where hierarchy lookup fails!" if len(hier_key) > 1 else "" ))
            break
        curr_obj = curr_obj[fld]

    if found:
        res_val = curr_obj

    return res_val


def check_if_cofig_sections_are_same(config_json, config_key1, config_key2):
    section1_json = get_val_by_hier_key(config_json, config_key1.split("->"), default={})
    section2_json = get_val_by_hier_key(config_json, config_key2.split("->"), default={})

    if len(section1_json) == 0 or len(section2_json) == 0:
        return True
    else:
        section1_str = json.dumps(section1_json, indent=4, sort_keys=True)
        section2_str = json.dumps(section2_json, indent=4, sort_keys=True)

        return section1_str == section2_str

def get_predictions_params(curr_eval_output_dir):
    eval_params = {PARAM_EVAL_FILE_DEV: "",
                   PARAM_EVAL_FILE_TEST: "",
                   PARAM_EVAL_METRICS_DEV: "",
                   PARAM_EVAL_METRICS_TEST: "",
                   }
    files_in_dir = glob.glob(curr_eval_output_dir + "/*")
    for curr_file in files_in_dir:
        curr_file_base = os.path.basename(curr_file)
        if curr_file_base == REGEX_EVAL_FILE_DEV:
            eval_params[PARAM_EVAL_FILE_DEV] = curr_file
        elif curr_file_base == REGEX_EVAL_FILE_TEST:
            eval_params[PARAM_EVAL_FILE_TEST] = curr_file
        elif re.search(REGEX_EVAL_METRICS_DEV, curr_file_base):
            eval_params[PARAM_EVAL_METRICS_DEV] = curr_file
        elif re.search(REGEX_EVAL_METRICS_TEST, curr_file_base):
            eval_params[PARAM_EVAL_METRICS_TEST] = curr_file

    return eval_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Display metrics results')

    parser.add_argument('-i', '--input_pattern', dest="input_pattern", metavar='CONFIG_FILE', type=str,
                        help='Input', default="_trained_models/\*")

    parser.add_argument('--eval_output_dir', dest="eval_output_dir", metavar='_output', type=str,
                        help='Output dir with eval', default="_output")

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
    eval_output_dir = args.eval_output_dir

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

    other_fields = profile_settings.get("other", [])

    if len(metrics_fields) == 0:
        print("No metrics fields to display!")
        print("You need to specify either a field profile (ex. --profile=narrativeqa)"
              " or a list of fields (ex. --fields best_epoch best_validation_rouge-4 test_rouge-l)!")

        exit(1)

    # config fields
    config_fields = profile_settings.get("config", [])
    config_fields_input = args.config_fields if args.config_fields is not None else []
    for field in config_fields_input:
        if field not in config_fields:
            config_fields.append(field)

    if len(config_fields) == 0:
        print("No config fields to display!")
        print("You need to specify either a field profile (ex. --profile=narrativeqa)"
              " or a list of fields (ex. --config_fields dataset_reader->type model->type)!")

        exit(1)

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
        base_dir = os.path.basename(dir)
        report_json = {"serialization_dir": dir,
                       "serialization_dir_base": base_dir}

        # Check if we have evaluated the dev and test sets as well as metrics
        curr_eval_output_dir = os.path.join(eval_output_dir, base_dir)
        eval_output_params = get_predictions_params(curr_eval_output_dir)
        report_json.update(eval_output_params)

        dir_creation_time = datetime.datetime.fromtimestamp(os.path.getctime(dir))
        report_json["start_time"] = datetime.datetime.strftime(dir_creation_time, date_format)

        # config file
        config_file = os.path.join(dir, "config.json")
        config_json = {}
        if os.path.exists(config_file):
            try:
                config_json = Params.from_file(config_file).as_dict()
            except:
                try:
                    config_json = json.loads(config_file)
                except:
                    config_json = {}

        # load fields:
        for field in config_fields:
            if field == "are_train_and_val_semantic_views_same":
                # heuruistic checks
                are_train_and_val_semantic_views_same = check_if_cofig_sections_are_same(config_json,
                                                                                         "dataset_reader->semantic_views_extractor",
                                                                                         "validation_dataset_reader->semantic_views_extractor")

                report_json["are_train_and_val_semantic_views_same"] = are_train_and_val_semantic_views_same
            else:
                report_json[field] = str(get_val_by_hier_key(config_json, field.split("->"), default=""))

        # metrics file
        metrics_json = {}

        metrics_file = os.path.join(dir, "metrics.json")
        training_completed = False
        if not os.path.exists(metrics_file):
            max_epoch = 100
            for epoch in range(max_epoch):
                metrics_file_epoch = os.path.join(dir, "metrics_epoch_{0}.json".format(epoch))
                if os.path.exists(metrics_file_epoch):
                    metrics_file = metrics_file_epoch
                else:
                    break
        else:
            training_completed = True

        if os.path.exists(metrics_file):
            metrics_file_time = datetime.datetime.fromtimestamp(os.path.getctime(metrics_file))
            report_json["metrics_file"] = os.path.basename(metrics_file)
            report_json["metrics_time"] = datetime.datetime.strftime(metrics_file_time, date_format)
            report_json["duration"] = str(metrics_file_time - dir_creation_time)

            with open(metrics_file, mode="r") as f_metrics:
                metrics_json = json.load(f_metrics)

        if len(metrics_json) == 0:
            if skip_empty:
                continue

        # load metrics fields:
        for field in metrics_fields:
            report_json[field] = str(metrics_json.get(field, ""))

        report_results_list.append(report_json)

    # Print all results
    display_fields = default_fields + config_fields + metrics_fields + other_fields
    print("-" * 10)
    print("\t".join(display_fields))

    for report_item in report_results_list:
        print("\t".join([str(report_item.get(fld, "")) for fld in display_fields]))

