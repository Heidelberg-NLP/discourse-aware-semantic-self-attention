from docqa.commands.evaluate_custom import EvaluateCustom
from allennlp.commands import main as main_allennlp

from docqa.commands.evaluate_qanet_semantic_flat import EvaluateQaNetSemanticFlat


def main(prog: str = None) -> None:
    subcommand_overrides = {
        "evaluate_custom": EvaluateCustom(),
        "evaluate_qanet_semantic_flat": EvaluateQaNetSemanticFlat(),
    }
    main_allennlp(prog, subcommand_overrides=subcommand_overrides)
