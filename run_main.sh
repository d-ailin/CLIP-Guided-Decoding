# For COCO
# clip-guided decoding
bash run.sh 0,1,2 mscoco_captions clip_guided 500 'cuda:2' 'F_ours' ' run.algo.sampling.top_k=5 run.algo.sampling.top_p=1 run.algo.sampling.num_return_sequences=3 run.algo.scoring.alpha=0.01 run.algo.scoring.prob_type=lennorm_sum_log '

# top-k sampling baseline
bash run.sh 0,1,2 mscoco_captions rsp_sampling 500 'cuda:1' 'F_baseline' ' run.algo.sampling.top_k=5 run.algo.sampling.num_beams=1 run.algo.sampling.top_p=1 run.algo.sampling.num_return_sequences=1 ' 


# For Nocaps
# clip-guided decoding
bash run.sh 0,1,2 nocaps clip_guided 500 'cuda:2' 'F_ours' ' run.domain=near-domain run.algo.sampling.top_k=5 run.algo.sampling.top_p=1 run.algo.sampling.num_return_sequences=3 run.algo.scoring.alpha=0.01 run.algo.scoring.prob_type=lennorm_sum_log '
bash run.sh 0,1,2 nocaps clip_guided 500 'cuda:2' 'F_ours' ' run.domain=out-domain run.algo.sampling.top_k=5 run.algo.sampling.top_p=1 run.algo.sampling.num_return_sequences=3 run.algo.scoring.alpha=0.01 run.algo.scoring.prob_type=lennorm_sum_log '

# top-k sampling baseline
bash run.sh 0,1,2 nocaps rsp_sampling 500 'cuda:1' 'F_baseline' ' run.domain=near-domain run.algo.sampling.top_k=5 run.algo.sampling.num_beams=1 run.algo.sampling.top_p=1 run.algo.sampling.num_return_sequences=1 ' 
bash run.sh 0,1,2 nocaps rsp_sampling 500 'cuda:1' 'F_baseline' ' run.domain=out-domain run.algo.sampling.top_k=5 run.algo.sampling.num_beams=1 run.algo.sampling.top_p=1 run.algo.sampling.num_return_sequences=1 ' 


# ablation study
bash run.sh 0,1,2 mscoco_captions clip_guided 500 'cuda:2' 'F_ablation' ' run.algo.sampling.top_k=5 run.algo.sampling.top_p=1 run.algo.sampling.num_return_sequences=3 run.algo.scoring.max_cand_num=1 run.algo.scoring.alpha=0.01 run.algo.scoring.prob_type=lennorm_sum_log '
bash run.sh 0,1,2 mscoco_captions clip_guided 500 'cuda:2' 'F_ablation' ' run.algo.sampling.top_k=5 run.algo.sampling.top_p=1 run.algo.sampling.num_return_sequences=5 run.algo.scoring.max_cand_num=1 run.algo.scoring.alpha=0.01 run.algo.scoring.prob_type=lennorm_sum_log '
bash run.sh 0,1,2 mscoco_captions clip_guided 500 'cuda:2' 'F_ablation' ' run.algo.sampling.top_k=5 run.algo.sampling.top_p=1 run.algo.sampling.num_return_sequences=5 run.algo.scoring.max_cand_num=1 run.algo.scoring.alpha=0 run.algo.scoring.prob_type=lennorm_sum_log '
bash run.sh 0,1,2 mscoco_captions clip_guided 500 'cuda:2' 'F_ablation' ' run.algo.sampling.top_k=5 run.algo.sampling.top_p=1 run.algo.sampling.num_return_sequences=5 run.algo.scoring.max_cand_num=1 run.algo.scoring.alpha=1 run.algo.scoring.beta=0 run.algo.scoring.prob_type=lennorm_sum_log '

