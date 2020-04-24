#!/usr/bin/env python
#
# File Name : rouge.py
#
# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

import numpy as np
import pdb
from tools.pycocoevalcap.squad.evaluate_v11 import metric_max_over_ground_truths, exact_match_score, f1_score


class SquadEM():
    '''
    Computing squad EM

    '''
    def __init__(self):
        pass

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert(len(candidate) == 1)
        assert(len(refs) > 0)

        exact_match = metric_max_over_ground_truths(
            exact_match_score, candidate[0], refs)

        return exact_match

    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py
        :param hypo_for_image: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values
        :param ref_for_image: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        score = []
        for id in imgIds:
            hypo = res[id]
            ref  = gts[id]

            score.append(self.calc_score(hypo, ref))

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "Rouge"
