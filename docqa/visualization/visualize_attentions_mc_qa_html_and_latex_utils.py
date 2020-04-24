import os

import string

def print_command_definition_latex_colors(cand_latex_colors):
    """
    Prints the definitions for latex
    :param cand_latex_colors: Colors to define
    :return:
    """
    print("\\usepackage{{color,soul}}")
    print("\\setul{{0.5ex}}{{0.3ex}}")
    print("\\newcommand{{\\tc}}[2]{{\\setulcolor{{#1}}\\ul{{#2}}\\setulcolor{{black}}}}")
    print("\\newcommand{{\\tcf}}[2]{{\\setulcolor{{#1}}\\ul{{\\textbf{{#2}}}}\\setulcolor{{black}}}}")

    for i, x in enumerate(cand_latex_colors):
        suff = string.ascii_uppercase[i]
        print("\\newcommand{{\\tc{{{1}}}}}[1]{{\\tc{{{0}}}{{#1}}}}".format(x, suff))


def get_item_latex_tokens(story_tokens, highlight_tokens_to_color, answer_token, line_breaks={"."}):
    """
    Generate colorful text tokens to visualize in latex
    :param story_tokens: Tokens to format
    :param highlight_tokens_to_color: Mapping between string tokens and latex colors
    :param answer_token: Answer token to highlight
    :param line_breaks: Tokens to replace with line breaks
    :return: Latex string
    """
    latex_string = ""
    for x in story_tokens:
        tkn = x.lower()
        tkn_latex = tkn
        if tkn in highlight_tokens_to_color:
            if tkn == answer_token:
                tkn_latex = "\\tcf{{{0}}}{{{1}}}".format(highlight_tokens_to_color[tkn], tkn)
            else:
                tkn_latex = "\\tc{{{0}}}{{{1}}}".format(highlight_tokens_to_color[tkn], tkn)
        latex_string = latex_string + " " + tkn_latex
        if tkn in line_breaks:
            latex_string += "\\\\\n"

    return latex_string


def get_item_latex_tokens_fixed(story_tokens, highlight_tokens_to_color, answer_token, line_breaks={"."}):
    """
    Generate colorful text tokens to visualize in latex.
    :param story_tokens: Tokens to format
    :param highlight_tokens_to_color: Mapping between string tokens and latex colors
    :param answer_token: Answer token to highlight
    :param line_breaks: Tokens to replace with line breaks
    :return: Latex string
    """
    latex_string = ""
    for x in story_tokens:
        tkn = x.lower()
        tkn_latex = tkn
        if tkn in highlight_tokens_to_color:
            if tkn == answer_token:
                # formatting is fixed - \\tcf{0}{{{1}}} instead of \\tcf{{{0}}}{{{1}}}
                tkn_latex = "\\tcf{0}{{{1}}}".format(highlight_tokens_to_color[tkn], tkn)
            else:
                # formatting is fixed! {{}}
                tkn_latex = "\\tc{0}{{{1}}}".format(highlight_tokens_to_color[tkn], tkn)
        latex_string = latex_string + " " + tkn_latex
        if tkn in line_breaks:
            latex_string += "\\\\\n"

    return latex_string


def get_item_latex(story_tokens, question_tokens, candidates_tokens, answer_token, highlight_tokens_to_color=None,
                   line_breaks={"."}):
    """
    Generate latex definition for a QA item
    :param story_tokens: Tokens of the story - if any
    :param question_tokens: Question tokens
    :param candidates_tokens: Candidates tokens
    :param answer_token: Answer token
    :param highlight_tokens_to_color: Mapping between tokens and color
    :param line_breaks:  Tokens to replace with line breaks
    :return:
    """
    latex_string = ""

    if highlight_tokens_to_color is None:
        highlight_tokens_to_color = {x: string.ascii_uppercase[i] for i, x in enumerate(candidates_tokens)}

    story_tokens_latex = get_item_latex_tokens_fixed(story_tokens, highlight_tokens_to_color, answer_token, line_breaks)
    latex_string += "\\textbf{Story:}\\\\\n" + story_tokens_latex + "\\\\\n"

    question_tokens_latex = get_item_latex_tokens_fixed(question_tokens, highlight_tokens_to_color, answer_token,
                                                        line_breaks)
    latex_string += "\\textbf{Question:}\\\\\n" + question_tokens_latex + "\\\\\n"

    candidates_tokens_latex = get_item_latex_tokens_fixed(candidates_tokens, highlight_tokens_to_color, answer_token,
                                                          line_breaks)
    latex_string += "\\textbf{Candidates:}\\\\\n" + candidates_tokens_latex + "\\\\\n"

    return latex_string


def get_item_latex_dynamic(story_tokens, question_tokens, candidates_tokens, answer_token,
                           highlight_tokens_to_color=None, line_breaks={"."}):
    """
    Generate latex definition for a QA item
    :param story_tokens: Tokens of the story - if any
    :param question_tokens: Question tokens
    :param candidates_tokens: Candidates tokens
    :param answer_token: Answer token
    :param highlight_tokens_to_color: Mapping between tokens and color
    :param line_breaks:  Tokens to replace with line breaks
    :return:
    """
    latex_string = ""

    if highlight_tokens_to_color is None:
        highlight_tokens_to_color = {x: cand_latex_colors[i] for i, x in enumerate(candidates_tokens)}

    story_tokens_latex = get_item_latex_tokens(story_tokens, highlight_tokens_to_color, answer_token, line_breaks)
    latex_string += "\\textbf{Story:}\\\\\n" + story_tokens_latex + "\\\\\n"

    question_tokens_latex = get_item_latex_tokens(question_tokens, highlight_tokens_to_color, answer_token, line_breaks)
    latex_string += "\\textbf{Question:}\\\\\n" + question_tokens_latex + "\\\\\n"

    candidates_tokens_latex = get_item_latex_tokens(candidates_tokens, highlight_tokens_to_color, answer_token,
                                                    line_breaks)
    latex_string += "\\textbf{Candidates:}\\\\\n" + candidates_tokens_latex + "\n"

    return latex_string


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rc('text', usetex=False)

matplotlib.style.use('seaborn-darkgrid')
import prettyplotlib as ppl
from textwrap import wrap


def get_html_colors_rainbow_rgba(num_colors, leave_opacity_as_format=False):
    """
    Generates html rainbow colors
    :param num_colors: Number of rainbow colors to generate
    :param leave_opacity_as_format: Flag whether to include opacity in the color
    :return:
    """
    colors_raw = cm.rainbow(np.linspace(0, 1, num_colors))
    colors_raw_rba = colors_raw * np.array([255, 255, 255, 1])
    colors_html = ["rgba({0}, {1}, {2}, {3})".format(int(c_rgba[0]), int(c_rgba[1]), int(c_rgba[2]),
                                                     "%s" if leave_opacity_as_format else c_rgba[3]) for c_rgba in
                   colors_raw_rba.tolist()]
    return colors_html


def get_value_and_format(dictionary, key, format_str, default_str):
    """
    Gets a value from dictionary for a given string and format it with format_str
    :param dictionary: Dictionary with values
    :param key: Key to get
    :param format_str: Format to apply
    :param default_str: If key is not found, use this default string
    :return: Formatted string
    """
    res_str = default_str
    if dictionary is not None and key in dictionary:
        res_str = format_str.format(dictionary[key])

    return res_str


def get_template_and_format(dictionary, key, default_str="{0}"):
    """
    Gets a format from dictionary and applies key to it
    :param dictionary: Dictionary with format values
    :param key: Key to get
    :param default_str: Default format if key is not found
    :return: Formatted string
    """
    res_str = default_str
    if dictionary is not None and key in dictionary:
        res_str = dictionary[key]

    return res_str.format(key)


def gen_tokens_content_color(tokens, token_to_back_color, delim=" ", break_sentence=False,
                             tkn_to_border_color=None,
                             word_to_content_template=None,
                             token_to_bord_bottom_color=None,
                             tag="span", tag_attr="",
                             under_border_side="bottom",
                             border_radius=5,
                             border_size=3):
    """
    Make colorful tokens in html
    :param tokens: Tokens for colorize
    :param token_to_back_color: Mapping tokens to background color
    :param delim: Delimiter to use between the tokens
    :param break_sentence: Whether to break sentences
    :param tkn_to_border_color:
    :param word_to_content_template:
    :param token_to_bord_bottom_color:
    :param tag:
    :param tag_attr:
    :param under_border_side:
    :param border_radius:
    :param border_size:
    :return:
    """
    if token_to_back_color is None:
        token_to_back_color = {}
    html_break = "<br/>"

    up_border_side = {"bottom": "top", "left": "right", "top": "bottom", "right": "left"}[under_border_side]
    break_map = {".": html_break, "?": html_break, ";": html_break, "!": html_break, "...": html_break}
    html_tokens = ["<span style=\""
                   + "background-color:" + token_to_back_color.get(x, "none") + ";"
                   + get_value_and_format(tkn_to_border_color, x, "border:2px solid {0};", "")
                   # + get_and_format(tkn_to_border_color, x, "border-"+up_border_side+":2px solid {0};", "")
                   + get_value_and_format(token_to_bord_bottom_color, x,
                                          "border-" + under_border_side + ":4px solid {0};", "")
                   + " border-radius: %spx;" % border_radius +
                   "\" >" + get_template_and_format(word_to_content_template, x) + "</span>"
                   + break_map.get(x, "") for x in tokens]

    html_content = "<" + tag + " " + tag_attr + ">" + delim.join(html_tokens) + "</" + tag + ">\n"
    return html_content


def fill_template(template, item_html):
    """
    Fills a text item template with values
    :param template: Text template (to replace values)
    :param item_html: QA item json
    :return: Filled template with values
    """
    res_text = template

    res_text = res_text.replace("{{ item.id }}", str(item_html["id"]))
    res_text = res_text.replace("{{ item.latex_path_figure_q_to_ch_inter_relative_no_know }}",
                                item_html["latex_path_figure_q_to_ch_inter_relative_no_know"])
    res_text = res_text.replace("{{ item.latex_path_figure_q_and_ch_to_f_relative }}",
                                item_html["latex_path_figure_q_and_ch_to_f_relative"])
    res_text = res_text.replace("{{ item.latex_path_figure_q_to_ch_inter_relative_with_know }}",
                                item_html["latex_path_figure_q_to_ch_inter_relative_with_know"])

    return res_text


from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english') + [",", ";", "!", "--", "?", ";", "."])

import json
import logging
import os
import sys

import argparse
import numpy as np


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.
    Source: https://nolanbconaway.github.io/blog/2017/softmax-numpy
    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def get_field_types_dict(json_item):
    """
    Get the types of the fields of the object
    :param json_item: Object
    :return: If dict, returns a new object with same viewlfs and types as values else, object type
    """
    if isinstance(json_item, dict):
        new_json_item = {}
        for k, v in json_item.items():
            if isinstance(v, dict):
                new_json_item[k] = get_field_types_dict(v)
            else:
                new_json_item[k] = str(type(v))

        return new_json_item
    else:
        return str(type(json_item))


from jinja2 import Environment, FileSystemLoader


def visualize_mappings(input_data, y_labels,
                       x_labels, title,
                       fig_path, show_values=True,
                       color_scheme=plt.cm.YlGn,
                       fig_height=15,
                       fig_width=30,
                       include_title=True,
                       wrap_title=True,
                       keep_y_row=None,
                       keep_x_col=None,
                       softmax_col=False,
                       save_format='png',
                       x_labels_rotation=0,
                       font_size=15,
                       cbar_labels=None):
    if keep_y_row is not None:
        y_labels = [lbl for i, lbl in enumerate(y_labels) if i in keep_y_row]

    if keep_x_col is not None:
        x_labels = [lbl for i, lbl in enumerate(x_labels) if i in keep_x_col]

    improvements_data = input_data
    if keep_y_row is not None:
        improvements_data = improvements_data[keep_y_row]

    if keep_x_col is not None:
        improvements_data = np.transpose(np.transpose(improvements_data, [1, 0])[keep_x_col], [1, 0])

    if softmax_col:
        improvements_data = np.transpose(softmax(np.transpose(np.log(improvements_data), [1, 0]), axis=-1), [1, 0])

    fig, ax = plt.subplots(1)
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    # plt.pcolormesh(fig, ax, my_data)
    plt.pcolor(improvements_data, cmap=color_scheme)
    ax.set_aspect('auto')
    y_filt = -1
    for y in range(improvements_data.shape[0]):
        for x in range(improvements_data.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.2f' % improvements_data[y, x] if show_values else "",
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=font_size,  # 15
                     weight='bold',
                     color='black'
                     )

    ax.set_yticks(np.arange(0.5, len(y_labels) + 0.5, 1))
    ax.set_yticklabels(y_labels, size=font_size, weight='bold')  # 25
    ax.set_xticks(np.arange(0.5, len(x_labels) + 0.5, 1))
    ax.set_xticklabels(x_labels, size=font_size, weight='bold', rotation=x_labels_rotation)  # 17
    ax.tick_params(axis='both', labelsize=font_size)  # 25
    if include_title:
        ax.set_title(title, fontsize=font_size)  # 20

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=font_size)  # 20

    if cbar_labels == "first_and_last":
        cbar_min = np.round(np.min(improvements_data), 1)
        cbar_max = np.round(np.max(improvements_data), 1)
        cbar.ax.set_yticklabels(np.array([cbar_min, "", "", "", cbar_max]), fontsize=12)

    plt.tight_layout()
    if save_format == "png;pdf":
        fig.savefig(fig_path, format='png')
        fig.savefig(fig_path + '.pdf', format='pdf')
    else:
        fig.savefig(fig_path, format=save_format)


def items_reader(input_path):
    """
    Reading items from file
    :param input_path: Input jsonl file
    :return: Iterator over the items
    """

    for line in open(input_path, mode="r"):
        item = json.loads(line.strip())
        yield item


def get_ordered_key_values(typed_values, init_value_name="ctx__ctx", final_value_name="full"):
    changes_names_keys = [x for x in typed_values.keys() if x not in [init_value_name, final_value_name]]
    ordered_keys = [init_value_name] + changes_names_keys + [final_value_name]

    key_values = [typed_values[k] for k in ordered_keys]

    key_values_array = np.asarray(key_values)

    return ordered_keys, key_values_array


def wrap_text(text, wrap_width):
    """
    Wraps text with a fixed width
    :param text: Text to wrap
    :param wrap_width: Width to wrap the text to
    :return: Wraped text (inserted \n)
    """
    wrapped = "\n".join(wrap(text, wrap_width))
    return wrapped


def keep_lowest_ranks_ids(ranks_list, top_ranks):
    """
    Prunes rank results to be lower than a given rank
    :param ranks_list: List of tuples (item, rank)
    :param top_ranks: Top ranked items to keep
    :return: List of the top ranked items
    """
    return [chi for chi, x in enumerate(ranks_list) if x <= top_ranks]


import itertools


def join_lists(list_of_lists):
    """
    Joins list of lists into a single list
    :param list_of_lists: List of lists to join
    :return: Single list
    """
    return list(itertools.chain.from_iterable(list_of_lists))


def get_ranks_2d(values):
    """
    Get the ranks of the values row-wise - for each rows sorts the values and returns ranks
    :param values: 2D values to rank
    :return: Ranks
    """
    ranks = np.zeros(values.shape)
    for ri in range(values.shape[0]):
        ranks[ri] = get_ranks_1d(values[ri])
    return ranks


def get_ranks_1d(values):
    """
    Get the ranks of values
    :param values: 1D value array
    :return: Ranks of the values
    """
    ranks = values.shape[0] - values.argsort().argsort()
    return ranks


def pointers_in_story_to_exported(pointers_lists):
    """
    Converts the input pointers list to a pointers in the text.
    This is specific for the export implemented in the knowledgeable reader!
    :param pointers_lists:
    :return:
    """
    curr_p = 0

    new_p_lists = []
    for p_lst in pointers_lists:
        new_p_lst = []
        for p_lst_id in range(len(p_lst)):
            new_p_lst.append(curr_p)
            curr_p += 1
        new_p_lists.append(new_p_lst)
    return new_p_lists

def export_item_visualization(item,
                              export_item_to_html,
                              export_figs_to_latex,
                              export_item_to_latex,
                              output_dir_items_relative,
                              output_dir_figures_relative,
                              output_dir_base,
                              output_dir_figures,
                              format_type,
                              template_item,
                              template_item_latex,
                              top_choices=8,  # 5 10  # many candidates
                              top_facts=20,  # 15 30
                              keep_last_choices_to_facts=6,  # 10
                              fig_font_size=14,  # figures dimensions
                              figs_width=14,  # 7 12
                              fig_facts_attentions_height=14,  # 7
                              fig_improv_attentions_height=7,
                              fig_titple_wrap=100,  # 60 100
                              q_to_ch_softmax="FinalOnly",
                              export_all=True,
                              show_improved=False,
                              show_made_worse=False,
                              save_format="jpg;png",
                              include_title=True,
                              question_short_name="Q",
                              choice_short_name="C",
                              fig_no_know_height = 2


):
    gold_answer_id = item["gold_label"]
    pred_answer_id = item["predicted_label"]

    if format_type == "cloze_style":
        firendly_names = {
            'ctx__ctx': "D$_{ctx}$, Q$_{ctx}$ (w/o know)",
            'ctx+kn__ctx': "D$_{ctx}$, Q$_{ctx+kn}$",  # We flip the name here as first question is reported!
            'ctx+kn__ctx+kn': "      D$_{ctx+kn}$, Q$_{ctx+kn}$",
            'ctx__ctx+kn': "D$_{ctx+kn}$, Q$_{ctx}$",
            'final': "Ensemble"
        }
    else:
        firendly_names = {
            'ctx__ctx': "%s$_{ctx}$, %s$_{ctx}$ (w/o know)" % (question_short_name, choice_short_name),
            'final': "Final"
        }
        possible_interactions = [["ctx", "ctx"], ["ctx+kn", "ctx"],
                                 ["ctx", "ctx+kn"], ["ctx+kn", "ctx+kn"],
                                 ["kn", "kn"], ["kn", "ctx+kn"],
                                 ["ctx+kn", "kn"], ["kn", "ctx+kn"],
                                 ["ctx+kn", "kn"],
                                 ["kn", "ctx"],
                                 ["ctx", "kn"]]
        for inter in possible_interactions:
            inter_name_key = "{0}__{1}".format(inter[0], inter[1])
            if inter_name_key == "ctx__ctx":
                continue
            inter_name_firendly = "%s$_{%s}$, %s$_{%s}$" % (question_short_name, inter[0], choice_short_name, inter[1])
            firendly_names[inter_name_key] = inter_name_firendly

    # question to choices
    row_keys, values_array = get_ordered_key_values(item["attentions"]["att_q_to_ch"],
                                                    init_value_name="ctx__ctx",
                                                    final_value_name="final")

    if q_to_ch_softmax == "True":
        values_array = softmax(values_array, axis=-1)

    if q_to_ch_softmax == "FinalOnly":
        values_array[-1] = softmax(values_array[-1], axis=-1)

    ranks = get_ranks_2d(values_array)
    ranks_no_know = ranks[0]
    ranks_no_know_inter_key = row_keys[0]
    ranks_final = ranks[-1]

    keep = False

    if export_all:
        keep = True
    elif show_improved or show_made_worse:
        curr_improve = (ranks_no_know[gold_answer_id] > 1 and ranks_final[gold_answer_id] == 1)
        curr_make_worse = (ranks_no_know[gold_answer_id] == 1 and ranks_final[gold_answer_id] > 1)

        if show_improved and curr_improve:
            keep = True

        if show_made_worse and curr_make_worse:
            keep = True

    if not keep:
        return

    if "story_tokens" in item:
        print("Story:")
        story_tokens = [x.lower() for x in item["story_tokens"]]
        print(" ".join(story_tokens))
        print("")
    else:
        story_tokens = []

    print("Question:")
    question_tokens = [x.lower() for x in item["question_tokens"]]

    print(" ".join(question_tokens))
    print("")

    print("Candidates:")
    candidates_tokens = [" ".join(x) for x in item["choice_tokens_list"]]
    print(" ".join(candidates_tokens))

    print("")
    candidates_tokens_set = set(candidates_tokens)
    question_tokens_set = set(
        [x for x in question_tokens if x not in STOP_WORDS and x not in candidates_tokens_set])

    # colors
    candidate_colors = get_html_colors_rainbow_rgba(len(candidates_tokens))
    gold_answer_token = candidates_tokens[gold_answer_id]

    border_color_map = {x: 'blue' for x in question_tokens_set}
    border_color_map[gold_answer_token] = 'black'

    cand_token_to_color_map = {x: candidate_colors[i] for i, x in enumerate(candidates_tokens)}
    cand_token_to_template_map = {x: "{0}<sub>c" + str(i + 1) + "</sub>" for i, x in
                                  enumerate(candidates_tokens)}

    story_tokens_color_content = gen_tokens_content_color(story_tokens, cand_token_to_color_map, delim=" ",
                                                          break_sentence=True,
                                                          tkn_to_border_color=border_color_map,
                                                          word_to_content_template=cand_token_to_template_map,
                                                          tag="p")
    cand_tokens_color_content = gen_tokens_content_color(candidates_tokens, cand_token_to_color_map, delim="  ",
                                                         break_sentence=True,
                                                         tkn_to_border_color=border_color_map,
                                                         word_to_content_template=cand_token_to_template_map,
                                                         tag="span")

    question_tokens_color_content = gen_tokens_content_color(question_tokens, cand_token_to_color_map,
                                                             delim="  ",
                                                             break_sentence=True,
                                                             tkn_to_border_color={gold_answer_token: "black"},
                                                             word_to_content_template=cand_token_to_template_map,
                                                             tag="span")

    # latex content
    item_story_latex = get_item_latex(story_tokens, question_tokens, candidates_tokens, gold_answer_token)

    candidate_frequencies = [1 for x in item["choice_tokens_list"]]
    print("candidate_frequencies:")
    print(candidate_frequencies)
    if "choice_tokens_list_to_facts" in item:
        candidate_frequencies = [len(x) for x in item["choice_tokens_list_to_facts"]]

        print("Frequences:")
        print(candidate_frequencies)

    keep_choice_ids = keep_lowest_ranks_ids(ranks_final.tolist(), top_choices)
    # [chi for chi, x in enumerate(ranks_final.tolist()) if x <= top_choices]
    print("keep_choice_ids:%s" % str(keep_choice_ids))

    # get stats
    stats = {}
    # check if the right rank is changed

    max_freq = max(candidate_frequencies)
    stats["freq_max_freq"] = max_freq

    # check if no_know result got it correctly
    stats["freq_gold_is_in_most_freq"] = max_freq == candidate_frequencies[gold_answer_id]

    print(row_keys)
    for inter_id, row in enumerate(row_keys):
        print(inter_id)
        curr_inter_key = row_keys[inter_id]
        curr_inter_id = inter_id

        # stats["freq_curr_pred_pred_is_in_most_freq__" + curr_inter_key] = max_freq == candidate_frequencies[curr_inter_id]

        stats["curr_pred_gold_rank__" + curr_inter_key] = ranks[curr_inter_id][gold_answer_id]
        if ranks[curr_inter_id][gold_answer_id] == 1:
            stats["curr_pred_is_correct__" + curr_inter_key] = True

            if inter_id > 0:
                stats["curr_pred_is_improved_correct__" + curr_inter_key] = (ranks_no_know[gold_answer_id] > 1)
        else:
            stats["curr_pred_is_correct__" + curr_inter_key] = False
            if inter_id > 0:
                stats["curr_pred_is_worsen_wrong__" + curr_inter_key] = (ranks_no_know[gold_answer_id] == 1)

    if (not export_all) and np.array_equal(ranks_no_know, ranks_final):
        logging.warning("Initial and final predictions are the same. Not interesting... Skip...")
        return None

    item_html = {}
    item_html["id"] = item["id"]
    item_html["stats_list_txt"] = ["{0}={1}".format(k, v) for k, v in stats.items()]
    item_html["data"] = item
    item_html["choice_frequencies"] = candidate_frequencies
    item_html["story_html"] = story_tokens_color_content
    item_html["question_html_color"] = question_tokens_color_content
    item_html["question_html"] = " ".join(item["question_tokens"])
    item_html["choices_html_color"] = cand_tokens_color_content  #

    item_html["choices_html"] = [" ".join(x) for x in item["choice_tokens_list"]]
    item_html["facts_html"] = [" ".join(x) for x in item["facts_tokens_list"]]
    item_html["save_format"] = save_format
    item_html["gold_facts"] = item.get("gold_facts", {})

    item_base_id = "item_{0}".format(item["id"])
    item_html_path_relative = "{0}/{1}.html".format(output_dir_items_relative, item_base_id)
    item_html_path = "{0}/{1}".format(output_dir_base, item_html_path_relative)

    path_figure_q_and_ch_to_f_relative = "{1}_{2}_.png".format(output_dir_figures_relative, item_base_id,
                                                               "q_and_ch_to_f")
    path_figure_q_and_ch_to_f = "{0}/{1}_{2}_.png".format(output_dir_figures, item_base_id, "q_and_ch_to_f")

    path_figure_q_to_ch_inter_relative = "{1}_{2}.png".format(output_dir_figures_relative, item_base_id,
                                                              "q_to_ch_inter")
    path_figure_q_to_ch_inter = "{0}/{1}_{2}.png".format(output_dir_figures, item_base_id, "q_to_ch_inter")
    path_figure_q_to_ch_inter_with_know = path_figure_q_to_ch_inter + "-with-know.png"
    path_figure_q_to_ch_inter_no_know = path_figure_q_to_ch_inter + "-no-know.png"

    item_html["path_figure_q_and_ch_to_f_relative"] = path_figure_q_and_ch_to_f_relative
    item_html["path_figure_q_to_ch_inter_relative"] = path_figure_q_to_ch_inter_relative
    item_html["path_figure_q_to_ch_inter_relative_no_know"] = path_figure_q_to_ch_inter_relative + "-no-know.png"
    item_html["path_figure_q_to_ch_inter_relative_with_know"] = path_figure_q_to_ch_inter_relative + "-with-know.png"
    item_html["item_html_path_relative"] = item_html_path_relative

    item_html["latex_path_figure_q_and_ch_to_f_relative"] = path_figure_q_and_ch_to_f_relative.replace(".", "_")
    item_html["latex_path_figure_q_to_ch_inter_relative_no_know"] = path_figure_q_to_ch_inter_relative.replace(
        ".",
        "_") + "-no-know_png"
    item_html[
        "latex_path_figure_q_to_ch_inter_relative_with_know"] = path_figure_q_to_ch_inter_relative.replace(
        ".", "_") + "-with-know_png"
    import string

    # keys to visualiza
    choices_ids = [string.ascii_uppercase[x] for x in range(len(item["choice_tokens_list"]))]
    choices_keys = [" ".join([xx for xx in x if "@" not in xx]) for x in item["choice_tokens_list"]]
    facts_keys = [" ".join([xx for xx in x if "@" not in xx]) for x in item["facts_tokens_list"]]
    question_text = "Q: " + " ".join([xx for xx in item["question_tokens"] if "@" not in xx])
    question_text = question_text.replace("<unknwn>", "unk").replace("<UNKNWN>", "UNK")

    # questions to facts
    row_keys_facts = [wrap_text(x, 30) for x in facts_keys]
    print("row_keys_facts:%s" % str(row_keys_facts))
    # col_keys = [question_text] + choices_keys
    col_keys = ["XXXXX"] + choices_keys

    attentions_to_facts = np.asarray(
        [item["attentions"]["att_q_to_f"]["src1"]] + item["attentions"]["att_ch_to_f"]["src1"]).transpose()
    attentions_to_facts = attentions_to_facts[:len(facts_keys)]

    print("attentions_to_facts.shape:%s" % (str(attentions_to_facts.shape)))

    if format_type == "cloze_style":
        choices_pointers = pointers_in_story_to_exported(item["choice_tokens_list_to_facts"])
        print("choices_pointers:%s" % str(choices_pointers))
        choices_pointers_in_facts_to_keep = join_lists(
            [x[-keep_last_choices_to_facts:] for xi, x in enumerate(choices_pointers) if xi in keep_choice_ids])
        # plus question as zero index
        choices_pointers_in_facts_to_keep = [0] + [x + 1 for x in choices_pointers_in_facts_to_keep]
        print("choices_pointers_in_facts_to_keep:%s" % str(choices_pointers_in_facts_to_keep))
    else:
        choices_pointers_in_facts_to_keep = range(len(item["choice_tokens_list"]) + 1)
    attentions_to_facts_picked = np.transpose(
        np.transpose(attentions_to_facts, [1, 0])[choices_pointers_in_facts_to_keep])

    facts_avg_ranks = get_ranks_1d(np.max(attentions_to_facts_picked, axis=-1))
    print("facts_avg_ranks:%s" % str(facts_avg_ranks))

    facts_ids_to_keep = keep_lowest_ranks_ids(facts_avg_ranks, top_facts)

    print("facts_ids_to_keep:%s" % str(facts_ids_to_keep))

    # export to single html only some...
    curr_pred_is_improved_correct__final = stats.get("curr_pred_is_improved_correct__final", False)
    curr_pred_is_worsen_wrong__final = stats.get("curr_pred_is_worsen_wrong__final", False)
    if export_all or curr_pred_is_improved_correct__final or curr_pred_is_worsen_wrong__final:
        # choices to question values
        choices_keys_wrapped = ['\n'.join(wrap(x, 15)) for x in choices_keys]

        print(row_keys)
        row_keys_friendly = [firendly_names.get(x, x) for x in row_keys]

        # no knowledge

        reverse_rows = True
        if reverse_rows:
            values_array = values_array[::-1]
            row_keys_friendly = row_keys_friendly[::-1]

        # picture with first row only
        keep_y_row_q_to_ch = [len(row_keys_friendly) - 1]
        visualize_mappings(values_array, y_labels=row_keys_friendly, x_labels=choices_keys_wrapped,
                           title=wrap_text(question_text, fig_titple_wrap),
                           fig_path=path_figure_q_to_ch_inter + "-no-know.png",
                           color_scheme=plt.cm.Blues,
                           fig_height=fig_no_know_height,  # 10
                           fig_width=figs_width,  # 20
                           keep_x_col=keep_choice_ids,
                           keep_y_row=keep_y_row_q_to_ch,
                           include_title=include_title,
                           save_format=save_format,
                           font_size=13,
                           cbar_labels="first_and_last")

        # picture with knowledge improvements only
        keep_y_row_q_to_ch = range(len(row_keys_friendly))[:-1]
        visualize_mappings(values_array, y_labels=row_keys_friendly, x_labels=choices_keys_wrapped,
                           title=wrap_text(question_text, fig_titple_wrap),
                           fig_path=path_figure_q_to_ch_inter + "-with-know.png",
                           color_scheme=plt.cm.Blues,
                           fig_height=fig_improv_attentions_height,  # 10
                           fig_width=figs_width,  # 20
                           keep_x_col=keep_choice_ids,
                           keep_y_row=keep_y_row_q_to_ch,
                           include_title=False,
                           save_format=save_format,
                           font_size=13)

        # full picture
        visualize_mappings(values_array, y_labels=row_keys_friendly, x_labels=choices_keys_wrapped,
                           title=wrap_text(question_text, fig_titple_wrap),
                           fig_path=path_figure_q_to_ch_inter,
                           color_scheme=plt.cm.Blues,
                           fig_height=fig_improv_attentions_height,  # 10
                           fig_width=figs_width,  # 20
                           keep_x_col=keep_choice_ids,
                           include_title=include_title,
                           save_format=save_format,
                           x_labels_rotation=0,  # =75
                           font_size=fig_font_size)

        # facts
        logging.info("Single HTML for item {0}...".format(item_base_id))

        col_keys = [question_short_name] + choices_keys
        if "story_tokens" in item:
            # cbt
            choices_keys_facts = []
            for ch_k_i, ch_k in enumerate(choices_keys):
                choices_keys_facts.extend(candidate_frequencies[ch_k_i] * [ch_k])

            col_keys = [question_short_name] + choices_keys_facts
        col_keys_wrapped = ['\n'.join(wrap(x, 15)) for x in col_keys]

        # remove duplicates
        curr_facts = []
        print(row_keys_facts)
        for fid, fact in enumerate(row_keys_facts):
            if fact in curr_facts:
                facts_ids_to_keep = [fc for fc in facts_ids_to_keep if fc != fid]
                continue
            curr_facts.append(fact)

        visualize_mappings(attentions_to_facts, y_labels=row_keys_facts, x_labels=col_keys_wrapped,
                           title=wrap_text(question_text, fig_titple_wrap),
                           fig_path=path_figure_q_and_ch_to_f, show_values=False, color_scheme=plt.cm.Oranges,
                           fig_height=fig_facts_attentions_height,  # 30
                           fig_width=figs_width,  # 30
                           keep_y_row=facts_ids_to_keep,
                           keep_x_col=choices_pointers_in_facts_to_keep,
                           include_title=False,
                           softmax_col=True,
                           save_format=save_format,
                           x_labels_rotation=75,
                           font_size=fig_font_size
                           )

        # save single item html
        if export_item_to_html:
            with open(item_html_path, mode="w") as f_item:
                f_item.write(template_item.render({"item": item_html}))

        if export_figs_to_latex:
            with open(output_dir_base + "/items/figures/item_{0}_fig.tex".format(item_html["id"]),
                      mode="w") as f_item:
                f_item.write(fill_template(template_item_latex, item_html))

        if export_item_to_latex:
            with open(output_dir_base + "/items/figures/item_{0}_story.tex".format(item_html["id"]),
                      mode="w") as f_item:
                f_item.write(item_story_latex.replace("\n", "\n %"))

    logging.info("Done!")

    return item_html


