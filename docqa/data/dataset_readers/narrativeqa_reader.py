# from pipeline.common.utils import read_csv_to_json_list, load_json_list
#
from typing import List

from docqa.utils.processing_utils import read_csv_to_json_list, load_json_list, iterate_json_list


def get_item_filter_by_docids_and_sets(doc_ids: List[str] = None, set_names: List[str] = None):
    """
    Generate function to be used for filtering
    :param doc_ids:
    :param set_names:
    :return:
    """

    doc_ids_set = set(doc_ids) if doc_ids is not None else None
    set_names_set = set(set_names) if set_names is not None else None

    # filter by set_names
    if set_names_set is not None and len(set_names_set) > 0:
        def item_filter_func_set_names(x):
            return True if x["set"] in set_names_set else False
    else:
        def item_filter_func_set_names(x):
            return True

    # filter by doc_ids
    if doc_ids_set is not None and len(doc_ids_set) > 0:
        def item_filter_func_doc_ids(x):
            return True if x["document_id"] in doc_ids_set else False
    else:
        def item_filter_func_doc_ids(x):
            return True

    def item_filter_func(x):
        return item_filter_func_set_names(x) and item_filter_func_doc_ids(x)

    return item_filter_func


class NarrativeQaReader(object):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

        self.docusments_meta_file = "{0}/documents.csv".format(dataset_folder)
        self.questions_file = "{0}/qaps.csv".format(dataset_folder)
        self.summaries_file = "{0}/third_party/wikipedia/summaries.csv".format(dataset_folder)

        # file formatting
        self.story_content_file_format = "{0}/raw_data/{{0}}.content".format(dataset_folder)


    def load_documents_meta_from_csv(self, doc_ids=None, documents_meta_file=None, set_names=None):
        """
        Loads documents metadata
        :param doc_ids:
        :return:
        """
        separator = ","

        item_filter_func = get_item_filter_by_docids_and_sets(doc_ids, set_names)

        # documents data
        if documents_meta_file is None:
            documents_meta_file = self.docusments_meta_file
        csv_meta_file = documents_meta_file

        field_names = ["document_id", "set", "kind", "story_url",
         "story_file_size", "wiki_url", "wiki_title",
         "story_word_count", "story_start", "story_end"]


        # load items
        items_list = read_csv_to_json_list(csv_meta_file,
                                            field_names=field_names,
                                            separator=separator,
                                            json_filer_func=item_filter_func)


        return items_list


    def load_questions_from_csv(self, doc_ids=None, questions_file=None, set_names=None):
        """
        Load questions
        :param doc_ids:
        :return:
        """
        separator = ","

        item_filter_func = get_item_filter_by_docids_and_sets(doc_ids, set_names)

        # questions data
        if questions_file is None:
            questions_file = self.questions_file

        csv_meta_file = questions_file

        # questions fields
        field_names = ["document_id", "set", "question",
                       "answer1", "answer2",
                       "question_tokenized",
                       "answer1_tokenized",
                       "answer2_tokenized"]

        # load questions
        items_list = read_csv_to_json_list(csv_meta_file,
                                           field_names=field_names,
                                           separator=separator,
                                           json_filer_func=item_filter_func)

        return items_list





    def load_json_list(self, jsonl_file, doc_ids=None, set_names=None):
        """
        Load json and filter by doc_ids or set (train, dev, test)
        :param jsonl_file: JSONL file path
        :param doc_ids: Load only specific doc_ids
        :return: Filtered items if doc_ids is not None, else all items
        """

        item_filter_func = get_item_filter_by_docids_and_sets(doc_ids, set_names)

        # load questions
        items_list = load_json_list(jsonl_file, filter_func=item_filter_func)

        return items_list

    def iterate_json_list(self, jsonl_file, doc_ids=None, set_names=None):
        """
        Load json and filter by doc_ids or set (train, dev, test)
        :param jsonl_file: JSONL file path
        :param doc_ids: Load only specific doc_ids
        :return: Filtered items if doc_ids is not None, else all items
        """

        item_filter_func = get_item_filter_by_docids_and_sets(doc_ids, set_names)

        # load questions
        for item in iterate_json_list(jsonl_file, filter_func=item_filter_func):
            yield item


    def load_summaries_from_csv(self, doc_ids=None, summaries_file=None, set_names=None):
        """
        Load summaries
        :param doc_ids:
        :return: List of summaries
        """
        separator = ","

        item_filter_func = get_item_filter_by_docids_and_sets(doc_ids, set_names)

        # summaries data
        if summaries_file is None:
            summaries_file = self.summaries_file
        csv_meta_file = summaries_file

        # summaries fields
        field_names = ["document_id", "set", "summary", "summary_tokenized"]

        # load summaries
        items_list = read_csv_to_json_list(csv_meta_file,
                                           field_names=field_names,
                                           separator=separator,
                                           json_filer_func=item_filter_func)

        return items_list


