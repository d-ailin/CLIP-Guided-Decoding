# CLIP-Guided Decoding

**Seeing is Believing: Mitigating Hallucination in Large Vision-Language Models via CLIP-Guided Decoding**: [[Paper](https://arxiv.org/abs/2402.15300)]

[Ailin Deng](https://d-ailin.github.io), [Zhirui Chen](https://zchen42.github.io/), [Bryan Hooi](https://bhooi.github.io/)

## Setup
run `setup.sh` to install basic dependencies. (Recommend using conda or other virtual environment before running set-up)

### Models Installation
* LLaVA: [https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
    <details>
        <summary>We use LLaVA-1.5 with github version tag v1.1.3</summary>

        ```bash
            git clone --depth 1 --branch v1.1.3 https://github.com/haotian-liu/LLaVA.git
            cd LLaVA
            pip instal -e .
        ```

    </details>

* InstructBLIP: [https://github.com/salesforce/LAVIS](https://github.com/salesforce/LAVIS)
* mPLUG-Owl2: [https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)

Install custom transformers after installing the models:
```
    cd dep/transformers_custom/transformers-4.31.0
    pip install -e .
```
The modifications compared with the original code are in `src/generation/utils.py` to return raw logits.



## Inference
We provide easy inference code at [inference.ipynb](./inference.ipynb).




### For Evaluation
Note that MySQL and Java are required in evaluation as package pycocoevalcap's requirements.

The COCO samples we tested can be accessed via [link](https://drive.google.com/drive/folders/1r-zZPRRJSv6yoHBzpElGMB3fyALSwNpa?usp=sharing). The json files contains generated responses (with top-k sampling here) with different random seeds. The mscoco id is "image_id" for each item in the json file.

#### COCO Data Structure
Download data from [here](https://cocodataset.org/#download). You could organize the downloaded data like:
```
    coco_test_karpathy.json
    val2014/
        - COCO_val2014_000000358301.jpg
        - COCO_val2014_000000455735.jpg
        - ...
    annotations/
        - captions_val2014.json
        - instances_val2014.json
        - person_keypoints_val2014.json
        - ...

```
After data preparation, change the `data_path` in `conf/mscoco_captions.yaml`.

## Run Tests
see `run_main.sh` and parameter arguments in `conf/mscoco_captions.yaml`



<!-- cocoevalcap files manipulation -->

## Possible Issues 
* (Evaluation) change the spice.py in pycocoeval package to enable larger cpu size (16 or 32 or 64G) to avoid memory error when using spice
```
# change '-Xmx8G' to '-Xmx16G' in spice.py
```
* (Evaluation) When using pycocoeval to compute BLEU/METEOR/ROUGE/SPICE metrics, it will raise an assertation issue as pycocoeval will evaluate all COCO samples but we only need to eval a subset of the dataset. You could remove the assertation and assign `imgIds` with `res.keys()`.


## Acknowledgement
CHAIR metrics implementation: [https://github.com/LisaAnne/Hallucination](https://github.com/LisaAnne/Hallucination)

MMVet Evaluation: [https://github.com/yuweihao/MM-Vet](https://github.com/yuweihao/MM-Vet)

## Citation
```
@article{deng2024seeing,
  title         = {Seeing is Believing: Mitigating Hallucination in Large Vision-Language Models via CLIP-Guided Decoding},
  author        = {Deng, Ailin and Chen, Zhirui and Hooi, Bryan},
  year          = {2024},
  journal       = {arXiv preprint arXiv:2402.15300},
  archivePrefix = {arXiv},
  eprint        = {2402.15300},
}
```