import json
import sys

annotations_value2name_mapping = {
    "skill": {
        '0': "word matching",
        '1': "paraphrasing",
        '2': "knowledge reasoning",
        '3': "meta/whole reasoning",
        '4': "math/logical reasoning",
    },
    "relation": {
        '0': "coreference",
        '1': "causal relation",
        '2': "spatial/temporal",
        '3': "none"
    },
}


annotations_info = {
    "skill": {  # checkboxes
        '0': "word matching",
        '1': "paraphrasing",
        '2': "knowledge reasoning",
        '3': "meta/whole reasoning",
        '4': "math/logical reasoning",
    },
    "relation": {
        '0': "coreference",
        '1': "causal relation",
        '2': "spatial/temporal",
        '3': "none"
    },
    "validity": {
        "0": "0",
        "1": "1"
    },
    "ambiguity": {
        "0": "0",
        "1": "1"
    },
    "multicand": {
        "0": "0",
        "1": "1"
    },
    "multisent": {
        "0": "0",
        "1": "1"
    },
}



def get_features_from_annotaiton(annotation):
    feats = []
    for k,v in annotation.items():
        values = v
        if not isinstance(values, list):
            values = [values]

        val2name = annotations_value2name_mapping.get(k, None)
        for val in values:
            curr_val = val
            if val2name is not None:
                curr_val = val2name.get(curr_val, curr_val)

            curr_feat = "{0}__{1}".format(k, curr_val)
            feats.append(curr_feat)

    return feats


if __name__ == "__main__":
    input_files = sys.argv[1].split("|")

    annotation = {'validity': '1',
                  'ambiguity': '1',
                  'multicand': '1',
                  'skill': ['0', '2'],
                  'multisent': '0',
                  'relation': '3',
                  'subset': 'hard',
                  }

    feats = get_features_from_annotaiton(annotation)

    print(feats)

    exit()
    for input_file in input_files:
        json_item = {}
        with open(input_file, mode="r") as f_in:
            json_item = json.load(f_in)
        print("{0}\t{1}\t{2}".format(input_file, len(json_item), len(list(json_item.keys()))))
