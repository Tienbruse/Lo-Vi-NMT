import argparse
from bleurt import score as bleurt_score
import sacrebleu


def evaluate_metrics(reference_file: str, hypothesis_file: str, bleurt_model: str):
    """
    Đọc 2 file corpus (reference và hypothesis), tính BLEURT và sacreBLEU.

    Args:
        reference_file (str): Đường dẫn tới file chứa câu tham chiếu, mỗi dòng một câu.
        hypothesis_file (str): Đường dẫn tới file chứa câu dự đoán, mỗi dòng một câu.
        bleurt_model (str): Đường dẫn tới model BLEURT (thư mục hoặc file .pbz2).

    Returns:
        dict: {
            'bleurt_scores': List[float],
            'bleurt_avg': float,
            'sacrebleu_score': float
        }
    """
    # Đọc dữ liệu
    with open(reference_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]
    with open(hypothesis_file, 'r', encoding='utf-8') as f:
        hypotheses = [line.strip() for line in f]

    if len(references) != len(hypotheses):
        raise ValueError("Reference and hypothesis must have the same number of lines.")

    # Tính BLEURT
    bleurt_scorer = bleurt_score.BleurtScorer(bleurt_model)
    bleurt_scores = bleurt_scorer.score(references=references, candidates=hypotheses)
    bleurt_avg = float(sum(bleurt_scores) / len(bleurt_scores))

    # Tính sacreBLEU
    # sacreBLEU expects list of hypotheses and list of list of references
    sacrebleu_result = sacrebleu.corpus_bleu(hypotheses, [references])
    sacrebleu_score = sacrebleu_result.score

    return {
        'bleurt_scores': bleurt_scores,
        'bleurt_avg': bleurt_avg,
        'sacrebleu_score': sacrebleu_score
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BLEURT and sacreBLEU on two corpora.")
    parser.add_argument('--refs', required=True, help='File tham chiếu (one sentence per line)')
    parser.add_argument('--hyps', required=True, help='File dự đoán (one sentence per line)')
    parser.add_argument('--bleurt_model', required=True, help='Đường dẫn tới BLEURT model')
    args = parser.parse_args()

    results = evaluate_metrics(args.refs, args.hyps, args.bleurt_model)
    print(f"BLEURT average score: {results['bleurt_avg']:.4f}")
    print(f"sacreBLEU score: {results['sacrebleu_score']:.2f}")

    # Nếu muốn xem chi tiết scores
    # for i, sc in enumerate(results['bleurt_scores']):
    #     print(f"{i+1}: {sc:.4f}")

if __name__ == '__main__':
    main()
