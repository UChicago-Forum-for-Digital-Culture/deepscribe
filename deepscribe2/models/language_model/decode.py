import numpy as np
from nltk.lm.api import LanguageModel
from nltk.util import everygrams


def beam_search_decoder(
    topk_chars: np.ndarray, model: LanguageModel, beam_width: int = 3
):
    seq_len, topk = topk_chars.shape
    beams = [([], 1e6)]  # (sequence, entropy)

    for i in range(seq_len):
        new_beams = []
        for seq, _ in beams:
            for j in range(topk):
                new_seq = seq + [topk_chars[i, j]]
                # compute everygrams
                grams = everygrams(new_seq, max_len=model.order)
                # Compute entropy using the model
                entropy = model.entropy(grams)
                new_beams.append((new_seq, entropy))

        # Sort beams based on entropy - lower is better
        new_beams.sort(key=lambda x: x[1])

        # Keep only the top-k beams
        beams = new_beams[:beam_width]

    best_seq, best_entropy = beams[0]
    return best_seq
