import json
import logging

from typing import Dict, Any

import time


def add_srl_to_sentences(parse, srl_parse):
    """
    This fixes differences in pre-processing some parts of the items like questions.
    The SRL is put outside the sentence
    """
    if "sentences" in parse:
        if not "sentences" in srl_parse:
            parse["sentences"][0]["srl"] = srl_parse
        else:
            for sent_id, sent in enumerate(parse["sentences"]):
                if len(srl_parse["sentences"]) > sent_id:
                    sent["srl"] = srl_parse["sentences"][sent_id]
                else:
                    sent["srl"] = {"verbs": []}
    elif "tokens" in parse:
        if "sentences" in srl_parse:
            raise ValueError("`parse` does not have sentences but `srl_parse` does!"
                             "Not sure how to handle this! Check manually!")
        else:
            parse["srl"] = srl_parse




def find_target_diff_edits(src_keep_fixed, target_to_change):
    """
    Find differences between the SRL parses and the coref parsing.
    This is required due to the difference in processing the SRL and Spacy parsing with Coref, POS, NER.

    returns: A list of insert edits to update the SRL parse.
    """

    diff_edits = []
    diffs_cnt = len(src_keep_fixed) - len(target_to_change)
    target_len = len(target_to_change)
    if diffs_cnt <= 0:
        raise Exception("`src_keep_fixed` length is expected to be higher than `target_to_change` length!")

    # additional token is in the middle
    if diffs_cnt == 1:
        # additional token is at the end
        if src_keep_fixed[-1] != target_to_change[-1]:
            curr_edit = {"pos": len(target_to_change), "token": src_keep_fixed[-1], "op": "insert"}
            diff_edits.append(curr_edit)
        elif src_keep_fixed[0] != target_to_change[0]:
            # additional token is in the beginning
            curr_edit = {"pos": 0, "token": src_keep_fixed[0], "op": "insert"}
            diff_edits.append(curr_edit)

    if diffs_cnt != len(diff_edits):
        target_id = 0

        for src_tkn_id, src_token in enumerate(src_keep_fixed):
            if target_id > len(target_to_change ) -1:
                curr_edit = {"pos": target_id + 1, "token": src_token, "op": "insert"}
                diff_edits.append(curr_edit)

                continue

            if src_token == target_to_change[target_id]:
                target_id += 1
                continue
            else:
                curr_edit = {"pos": target_id + 1, "token": src_token, "op": "insert"}
                diff_edits.append(curr_edit)

            if len(diff_edits) == diffs_cnt:
                break

    return diff_edits


def apply_edits_to_srl_parse(srl_parse, diff_edits):
    """
    Updates the SRL parse with a list of edits.
    """

    edits_offset = 0
    for edit in diff_edits:
        # apply to words
        # apply to verbs
        edit_op = edit["op"]
        edit_pos = edit["pos"] + edits_offset
        edit_token = edit["token"]
        if edit_op == "insert":
            srl_parse["words"].insert(edit_pos, edit_token)
            if 'verbs' in srl_parse:
                for verb_parse in srl_parse["verbs"]:
                    prev_label = 'O' if edit_pos == 0 or edit_pos >= len(verb_parse["tags"]) - 1 else \
                    verb_parse["tags"][edit_pos]
                    next_label = 'O' if edit_pos >= len(verb_parse["tags"]) - 1 else verb_parse["tags"][edit_pos + 1]
                    curr_label = 'O'
                    if next_label[0] == 'I':
                        # If next label is continuation, current word will also be the same label. Maybe not the best handling but breaking an annotation can lead to more mistakes..
                        curr_label = next_label

                    if edit_pos >= len(verb_parse["tags"]) - 1:
                        verb_parse["tags"].append(curr_label)
                    else:
                        verb_parse["tags"].insert(edit_pos, curr_label)

            edits_offset += 1
        else:
            raise Exception("operation `{0}` is not supported for edit {1}".format(edit["op"], str(edit)))


def fix_srl_tokenization_for_parse_field_in_items(items, field_name, remove_srl_if_error=False):
    """
    Updating SRL parses to match the Spacy parsing with Coref, POS, NER
    """

    for item_id, curr_item in enumerate(items):
        for sent_id, sent in enumerate(curr_item[field_name]['sentences']):
            try:
                if not "srl" in sent:
                    sent["srl"] = {"verbs": [], "words": sent['tokens']}

                words_diff_cnt = (len(sent['srl']["words"]) - len(sent['tokens']))
                if words_diff_cnt == 0:
                    continue
                elif words_diff_cnt < 0:
                    try:
                        diff_edits = find_target_diff_edits(sent['tokens'], sent['srl']["words"])
                        apply_edits_to_srl_parse(sent['srl'], diff_edits)
                        if sent['srl']["words"][-1] != sent['tokens'][-1] and words_diff_cnt == -1:
                            continue
                    except Exception as ex:
                        logging.warning("Error when trying to fix SRL. Removing SRL from this sentence!:{0}".format(ex))
            except Exception as e:
                logging.error("Error in item_id {0}, sent_id {1}: {2}".format(item_id, sent_id,e))


def add_sentence_token_offsets(items, field_name):
    """
    Add sentence token offsets for the summary
    """

    for curr_item in items:
        tokens_offset = 0
        for sent in curr_item[field_name]['sentences']:
            sent["tokens_offset"] = tokens_offset
            tokens_offset += len(sent["tokens"])


def add_srl_arguments(srl_parse):
    """
    Extract and add SRL arguments from IOB to the parse.
    """

    for verb_parse in srl_parse["verbs"]:
        verb_arguments = {}
        curr_arg = None
        for token_id, tag in enumerate(verb_parse["tags"]):
            if tag == "O":
                if curr_arg is not None:
                    curr_arg["end"] = token_id
                    verb_arguments[curr_arg["type"]] = curr_arg
                    curr_arg = None
                continue

            elif tag[0] == "B":
                if curr_arg is not None:
                    curr_arg["end"] = token_id
                    verb_arguments[curr_arg["type"]] = curr_arg
                    curr_arg = None

                curr_arg = {}
                curr_arg["type"] = tag[2:]
                curr_arg["start"] = token_id
            elif tag[0] == "I":
                continue

        if curr_arg is not None:
            curr_arg["end"] = token_id
            verb_arguments[curr_arg["type"]] = curr_arg
            curr_arg = None

        verb_parse["arguments"] = verb_arguments


def add_srl_arguments_for_items(items, parse_field):
    """
    Add srl arguments for a list of items.
    """

    for curr_item in items:
        add_srl_arguments_for_sentences(curr_item[parse_field])


def add_srl_arguments_for_sentences(parse):
    """
    Add srl arguments for a list of items.
    """

    for sent in parse['sentences']:
        add_srl_arguments(sent["srl"])

def fix_parse_for_items(items, field_name):
    fix_srl_tokenization_for_parse_field_in_items(items, field_name)
    add_sentence_token_offsets(items, field_name)
    add_srl_arguments_for_items(items, field_name)


def node_from_srl_arg(srl_arg, text, sent_offset, sent_id, verb_id):
    """
    Creates a graph node from SRL aruments
    """

    node = {"name": text,
            "span_start": sent_offset + srl_arg["start"],
            "span_end": sent_offset + srl_arg["end"],
            "group": sent_id,
            "sent_id": sent_id,
            "verb_id": verb_id,
            "type": "SRL",
            "sub_type": srl_arg["type"]
            }

    return node


def node_from_coref_metnion(coref_mention, mention_type):
    """
    Create a graph node from Coref mention.
    """

    node = {"name": coref_mention["text"],
            "span_start": coref_mention["start"],
            "span_end": coref_mention["end"],
            "group": -1,
            "type": "COREF",
            "sub_type": mention_type
            }

    return node


def update_tokens_to_nodes(tokens_to_nodes, node_id, node_start_token, node_end_token, summary_tokens_to_nodes):
    for token_id in range(node_start_token, node_end_token):
        if token_id in summary_tokens_to_nodes:
            tokens_to_nodes[token_id].append(node_id)
        else:
            tokens_to_nodes[token_id] = [node_id]


def build_graph_from_parse_verbs_as_nodes(summary_parse, num_sents=0):
    """
    Build a graph from summary parse.
    """

    # include verbs as nodes
    nodes = []
    edges = []
    summary_tokens_to_nodes = {}

    update_graph_with_srl(edges, nodes, num_sents, summary_parse, summary_tokens_to_nodes)
    update_graph_with_coref(edges, nodes, summary_parse, summary_tokens_to_nodes)

    graph = {"nodes": nodes, "links": edges}

    return graph


def update_graph_with_coref(edges, nodes, summary_parse, summary_tokens_to_nodes):
    # connect each main coref mention
    # to the nodes that contain tokens from the mentions
    for coref_cluster in summary_parse['coref_clusters']:
        main_cluster = coref_cluster["main"]
        main_cluster_node = node_from_coref_metnion(main_cluster, mention_type="main")
        main_cluster_node_id = len(nodes)
        main_cluster_node["id"] = main_cluster_node_id

        nodes.append(main_cluster_node)

        # collect a list of nodes and number of overlapping tokens
        nodes_to_connect = {}
        for mention in coref_cluster["mentions"]:
            for token_id in range(mention["start"], mention["end"]):
                # get nodes for tokens
                token_nodes = summary_tokens_to_nodes.get(token_id, None)
                if token_nodes is None:
                    continue

                token_nodes = list(set(token_nodes))
                for token_node in token_nodes:
                    # update the number of overlapping tokens for the node
                    if token_node in nodes_to_connect:
                        nodes_to_connect[token_node] += 1
                    else:
                        nodes_to_connect[token_node] = 1

        for node_to_connect_id, tokens_cnt in nodes_to_connect.items():
            node_item = nodes[node_to_connect_id]

            curr_link = {'source': node_to_connect_id,
                         'target': main_cluster_node_id,
                         'rel': 'COREF'
                         }
            edges.append(curr_link)

        update_tokens_to_nodes(summary_tokens_to_nodes, main_cluster_node_id, main_cluster["start"],
                               main_cluster["end"], summary_tokens_to_nodes)


def build_graph_with_coref(summary_parse):
    edges = []
    nodes = []
    summary_tokens_to_nodes = {}

    update_graph_with_coref(edges, nodes, summary_parse, summary_tokens_to_nodes)

    graph = {"nodes": nodes, "edges": edges}

    return graph


def update_graph_with_srl(edges, nodes, num_sents, summary_parse, summary_tokens_to_nodes,
                          add_rel_between_args=False,
                          include_prev_verb_rel=False):
    prev_verb_node_id = -1
    prev_verb_sent_id = -1
    for sent_id, sent in enumerate(summary_parse["sentences"]):
        if num_sents > 0 and sent_id > num_sents:
            break

        tokens = sent["tokens"]
        for verb_id, verb_parse in enumerate(sent["srl"]["verbs"]):
            if "arguments" not in verb_parse:
                continue

            if len(verb_parse["tags"]) != len(tokens):
                continue

            arguments = verb_parse["arguments"]

            # verb to node
            if not "V" in arguments:
                continue

            arg_id_to_nodes = {}
            verb_node_id = -1
            for arg_row_id, arg_row_item in enumerate(arguments.items()):
                arg_row_type, arg_row = arg_row_item

                if not add_rel_between_args and arg_row_type != "V":
                    # If this is False, then we add relations only between verbs and arguments!
                    continue

                # Create a node if it is not there yet
                arg_row_node = arg_id_to_nodes.get(arg_row_id, None)
                if arg_row_node is None:
                    arg_row_node = node_from_srl_arg(arg_row, " ".join(tokens[arg_row["start"]: arg_row["end"]]), sent["tokens_offset"],
                                                 sent_id, verb_id)
                    arg_row_node_id = len(nodes)
                    arg_row_node["id"] = arg_row_node_id
                    nodes.append(arg_row_node)
                    arg_id_to_nodes[arg_row_id] = arg_row_node
                    update_tokens_to_nodes(summary_tokens_to_nodes, arg_row_node_id, sent["tokens_offset"] + arg_row["start"],
                                           sent["tokens_offset"] + arg_row["end"], summary_tokens_to_nodes)
                else:
                    arg_row_node_id = arg_row_node["id"]

                # connect to previos verb
                if arg_row_type == "V" and prev_verb_node_id >= 0 and include_prev_verb_rel:
                    verb_node_id = arg_row_node_id
                    curr_link = {'source': verb_node_id,
                                 'target': prev_verb_node_id,
                                 'rel': 'PREV_EVENT%s' % ("_PS" if sent_id != prev_verb_sent_id else ""),
                                 }
                    edges.append(curr_link)



                # connect to other arguments
                for arg_id, arg_item in enumerate(arguments.items()):
                    arg_type, arg = arg_item

                    arg_node = arg_id_to_nodes.get(arg_id, None)
                    if arg_node is None:
                        arg_node = node_from_srl_arg(arg, " ".join(tokens[arg["start"]: arg["end"]]), sent["tokens_offset"],
                                                     sent_id, verb_id)
                        arg_node_id = len(nodes)
                        arg_node["id"] = arg_node_id
                        nodes.append(arg_node)
                        arg_id_to_nodes[arg_id] = arg_node
                        update_tokens_to_nodes(summary_tokens_to_nodes, arg_node_id, sent["tokens_offset"] + arg["start"],
                                               sent["tokens_offset"] + arg["end"], summary_tokens_to_nodes)
                    else:
                        arg_node_id = arg_node["id"]

                    curr_link = {'source': arg_node_id,
                                 'target': arg_row_node_id,
                                 'rel': 'SRL_{0}__{1}'.format(arg_type, arg_row_type),
                                 'sent_id': sent_id,
                                 'verb_id': verb_id,
                                 }
                    edges.append(curr_link)

            prev_verb_sent_id = sent_id
            prev_verb_node_id = verb_node_id


def build_graph_with_srl(summary_parse: Dict[Any, Any],
                         add_rel_between_args: bool,
                         include_prev_verb_rel: bool):
    edges = []
    nodes = []
    summary_tokens_to_nodes = {}

    update_graph_with_srl(edges, nodes, 0, summary_parse, summary_tokens_to_nodes,
                          add_rel_between_args=add_rel_between_args,
                          include_prev_verb_rel=include_prev_verb_rel)

    graph = {"nodes": nodes, "edges": edges}

    return graph


if __name__ == "__main__":
    items = []
    with open("tests/fixtures/data/narrativeqa/third_party/wikipedia/summaries_debug.csv.parsed.jsonl.srl.jsonl", mode="r") as f:
        for line in f:
            items.append(json.loads(line.strip()))

    start = time.time()
    fix_parse_for_items(items)
    print("{0} items fixed in {1}".format(len(items), time.time() - start))

    start = time.time()
    for curr_item in items:
        graph = build_graph_from_parse_verbs_as_nodes(curr_item["summary_parse"])

    print("{0} items graph built in {1}".format(len(items), time.time() - start))

