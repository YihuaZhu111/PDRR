# PDRR

Here is the code of the Paper: **Beyond Chains: Bridging Large Language Models and Knowledge Bases in Complex Question Answering**.

This paper has been accepted by **AAAI2026** Main.

# First, environment

please check the requirement.txt file.

# Second, Freebase

Please check freebase_readme.md file.


# Third, Run the code

```
python main.py \
    --dataset cwq \
    --remove_unnecessary_rel True \
    --LLM_type gpt \
    --engine gpt-4o \
    --question_type_from LLM \
    --openai_api_key your_openai_api_key
```

# Finally, Evaluation

```
python eval.py \
--dataset cwq \
--output_file your_output_file 
```

# Citation
If you want to cite this paper or want to use this code, please cite the following paper:

## Arxiv:
```
@article{zhu2025beyond,
  title={Beyond Chains: Bridging Large Language Models and Knowledge Bases in Complex Question Answering},
  author={Zhu, Yihua and Liu, Qianying and Aizawa, Akiko and Shimodaira, Hidetoshi},
  journal={arXiv preprint arXiv:2505.14099},
  year={2025}
}
```

## AAAI 2026:
```
will upload soon
```
