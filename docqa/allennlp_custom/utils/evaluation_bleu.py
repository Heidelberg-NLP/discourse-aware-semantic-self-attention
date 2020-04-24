
from nltk.translate.bleu_score import sentence_bleu


# print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
# print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
# print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
# print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))

BLEU_TYPE_WEIGHTS = {
    "BLEU1": (1, 0, 0, 0),
    "BLEU2": (0.5, 0.5, 0, 0),
    "BLEU3": (0.33, 0.33, 0.33, 0),
    "BLEU4": (0.25, 0.25, 0.25, 0.25)
}


def eval_bleu_batch(references_list, candidate_list, weights):
    blue_scores = []
    cand_and_refs = zip(references_list, candidate_list)
    for reference, candidate in cand_and_refs:
        if len(candidate) == 0:
            score = 0
        else:
            score = sentence_bleu(reference, candidate, weights=weights)
        blue_scores.append(score)

    return blue_scores


def eval_bleu_batch_by_type(references_list, candidate_list, bleu_type):
    weights = BLEU_TYPE_WEIGHTS[bleu_type]

    bleu_scores = eval_bleu_batch(references_list, candidate_list, weights)

    return bleu_scores


def eval_bleu1_batch(references_list, candidate_list):
    return eval_bleu_batch_by_type(references_list, candidate_list, "BLEU1")


def eval_bleu4_batch(references_list, candidate_list):
    return eval_bleu_batch_by_type(references_list, candidate_list, "BLEU4")



if __name__ == "__main__":
    #candidates_list = [".", ]
    candidate = ["Las", "Vegas"]
    references = [["She", "was", "a", "Las", "Vegas", "showgirl", "."], ["Showgirl"]]

    print("references:%s" % str(references))
    print("candidate:%s" % str(candidate))
    bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(references, candidate, weights=(0, 1, 0, 0))
    bleu3 = sentence_bleu(references, candidate, weights=(0, 0, 1, 0))
    bleu4 = sentence_bleu(references, candidate, weights=(0, 0, 0, 1))

    print("BLEU1: %s" % bleu1)
    print("BLEU2: %s" % bleu2)
    print("BLEU3: %s" % bleu3)
    print("BLEU4: %s" % bleu4)

    from nltk.translate.bleu_score import sentence_bleu

    reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
    candidate = ['this', 'a', 'test']
    score = sentence_bleu(reference, candidate)
    print(score)

