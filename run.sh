seed=$1
ds=$2
algo=$3
test_sample_num=$4
device=$5
tag=$6
others=$7

clip_model_name=$8
clip_model_pretrain=$9

if [ -z "$test_sample_num" ]; then
    # If the argument is empty, use a default value
    test_sample_num=30
fi

if [ -z "$device" ]; then
    # If the argument is empty, use a default value
    device='cuda:0'
fi

if [ -z "$tag" ]; then
    # If the argument is empty, use a default value
    tag=''
fi

if [ -z "$others" ]; then
    # If the argument is empty, use a default value
    others=''
fi

# Check if the argument is empty
if [ -z "$clip_model_name" ]; then
    # If the argument is empty, use a default value
    clip_model_name="ViT-SO400M-14-SigLIP-384"
    clip_model_pretrain="webli"
fi

if [ $ds == "mscoco_captions" ] || [ $ds == 'flickr8k' ] || [ $ds == 'flickr30k' ] || [ $ds == 'nocaps' ] || [ $ds == 'mscoco' ] || [ $ds == 'gqa' ] || [ $ds == 'aokvqa' ]; then
    
    if [ $ds == "mscoco_captions" ]; then
        data_path=${path}/coco
        # data_path=/home/data/coco
    fi

    if [ $ds == "nocaps" ]; then
        data_path=${path}/nocaps/
        # data_path=/home/data/flickr30k/
    fi
    
    # clip_configs=(
    #     'ViT-L-14-336,openai'
    #     'ViT-SO400M-14-SigLIP-384,webli'

    # )

    # for config in ${clip_configs[@]}; do
    #     IFS=',' read -ra c_list <<< "${config}"
    #     echo $c_list

        model_configs=(
            'llava_v1_5,7b'
            'mplug_owl2,llama2-7b'
            # 'blip2_vicuna_instruct,vicuna7b'
        )

        for m_config in ${model_configs[@]}; do
            IFS=',' read -ra m_list <<< "${m_config}"
            echo $m_list


            q_types=(
                describe_detailed
            )
            for q_type in ${q_types[@]}; do
                random_seeds=(0)
                for s in ${random_seeds[@]}; do

                    if [ $ds == 'mscoco_captions' ] || [ $ds == 'nocaps' ]; then
                        # python main.py -m run=${ds} run.seed=$1 run.q_type=${q_type} run.qa_model.model_name=${m_list[0]} run.qa_model.model_type=${m_list[1]} run.algo.name=${algo} run.device=${device} run.algo.clip.model_name=${clip_model_name} run.algo.clip.model_pretrain=${clip_model_pretrain} run.tag=${tag} run.test_sample_num=${test_sample_num} task=generation,eval ${others}
                        python main.py -m run=${ds} run.seed=$1 run.q_type=${q_type} run.qa_model.model_name=${m_list[0]} run.qa_model.model_type=${m_list[1]} run.algo.name=${algo} run.device=${device} run.algo.clip.model_name=${clip_model_name} run.algo.clip.model_pretrain=${clip_model_pretrain} run.tag=${tag} run.test_sample_num=${test_sample_num} ${others} task=generation,eval 
                    fi

                done
            done
        # done
    done
fi

