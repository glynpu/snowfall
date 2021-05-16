from kaldialign import edit_distance
def load_text(file_name):
    texts = []
    with open(file_name) as f:
        for line in f:
            # import pdb; pdb.set_trace()
            texts.append(line.strip().split(" ")[1:])
    return texts

def main():
    ref_file='dump/raw/test_clean/text'
    # hyp_file='exp/kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave/decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_clean/text'
    # hyp_file='exp/hyp.text'
    hyp_file='exp/hyp_2021-05-11.text'
    refs = load_text(ref_file)
    hyps = load_text(hyp_file)
    # s = ''
    # for ref, hyp in zip(refs, hyps):
    #     s += f'ref={ref}\n'
    #     s += f'hyp={hyp}\n'
    dists = [edit_distance(r, h) for r, h in zip(refs, hyps)]
    errors = {
        key: sum(dist[key] for dist in dists)
        for key in ['sub', 'ins', 'del', 'total']
    }
    total_words = sum(len(ref) for ref in refs)
    # Print Kaldi-like message:
    # %WER 8.20 [ 4459 / 54402, 695 ins, 427 del, 3337 sub ]
    print(
        f'%WER {errors["total"] / total_words:.2%} '
        f'[{errors["total"]} / {total_words}, {errors["ins"]} ins, {errors["del"]} del, {errors["sub"]} sub ]'
    )


main()
