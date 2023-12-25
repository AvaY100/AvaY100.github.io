



<div align="center">

<img src="https://cdn.discordapp.com/attachments/941582479117127680/1111543600879259749/20230526075532.png" width="400px">

**Tool Learning, Model & Data**

<p align="center">
   <a href="#model"><b>Model</b></a> •
  <a href="#data">Data Release</a> •
</p>

</div>

<div align="center">

![Dialogues](https://img.shields.io/badge/Tool\_Num-9-red?style=flat-square)
![Dialogues](https://img.shields.io/badge/Current\_Dataset\_Size-98K-red?style=flat-square)
![Dialogues](https://img.shields.io/badge/Total\_API\_Call-312K-red?style=flat-square)
![Dialogues](https://img.shields.io/badge/Tool\_LLaMA-Released-green?style=flat-square)

</div>




This project aims to construct *open-source, large-scale, single-tool, multi-tool* tool learning data powered by Turbo APIs to facilitate the construction of powerful language models with general tool-use capability.
In consideration of factors such as safeguarding privacy, **we do not directly use any data available on the Internet as prompts**.

*Please note that current released data is still not the final version. We are conducting extensive post-processing to reduce hallucination and ensure coherence to further improve the data quality.*


<br>
<div align="center">
<img src="https://cdn.discordapp.com/attachments/941582479117127680/1111210433307750451/ToolLLaMA.png" width="700px">
</div>
<br>

## Data

The dataset is intended solely for research and educational purposes and should not be construed as reflecting the opinions or views of the creators, owners, or contributors of this dataset. And it is distributed under [CC BY NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use).


| Tool           | Query num | Step num | Step per query |
|----------------|-----------|----------|----------------|
| Weather        | 9827      | 23740    | 2.4            |
| Chemical       | 8585      | 29916    | 3.5            |
| Translation    | 10267     | 23011    | 2.2            |
| Map            | 7305      | 23325    | 3.2            |
| Stock          | 11805     | 32550    | 2.8            |
| Meta analysis  | 2526      | 15725    | 6.2            |
| Bing search    | 31089     | 102088   | 3.3            |
| Wolfram        | 16130     | 56169    | 3.5            |
| Database       | 1264      | 6347     | 5              |

### Data Release
[Explore](http://39.101.77.220/) the data before downloading, or use [Atlas explorer](https://atlas.nomic.ai/map/0ce65783-c3a9-40b5-895d-384933f50081/a7b46301-022f-45d8-bbf4-98107eabdbac).

- 🤗 [Huggingface Datasets Host](https://huggingface.co/datasets/stingning/ultrachat)

Direct Download links:
- [Data]](https://cloud.tsinghua.edu.cn/f/0a27393192ad46a5a081/?dl=1)
- [Model](https://cloud.tsinghua.edu.cn/f/1f7abdf2d2564cb4b338/?dl=1)

### Data Format
Each line in the downloaded data file is a json dict containing the data id and dialogue data in a list format. Below is an example line.

```JSON
{
    "prompt": "Answer the following questions as best you can. Specifically, you have access to the following APIs:\n\nget_translation: . Your input should be a json (args json schema): {{\"text\" : string, \"tgt_lang\" : string, }} The Action to trigger this API should be get_translation and the input parameters should be a json dict string. Pay attention to the type of parameters.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [get_translation]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times, max 7 times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin! Remember: (1) Follow the format, i.e,\nThought:\nAction:\nAction Input:\nObservation:\nFinal Answer:\n (2) Provide as much as useful information in your Final Answer. (3) Do not make up anything, and if your Observation has no link, DO NOT hallucihate one. (4) If you have enough information and want to stop the process, please use \nThought: I have got enough information\nFinal Answer: **your response. \n The Action: MUST be one of the following:get_translation\nQuestion: {input}\n Agent scratchpad (history actions):\n {agent_scratchpad}",
    "query": "My intention is to convert the data provided in ما هي الأقسام الثلاثة للقوات المسلحة؟ into Arabic(ara).\n",
    "chains": [
        {
            "thought": "I need to use the get_translation API to convert the text into Arabic.",
            "action": "get_translation",
            "action_input": "{\"text\": \"What are the three branches of the military?\", \"tgt_lang\": \"ara\"}",
            "observation": "\"ما هي الفروع الثلاثة للجيش ؟\""
        }
    ],
    "answer": "The translation of \"What are the three branches of the military?\" into Arabic is \"ما هي الفروع الثلاثة للجيش ؟\"."
}

```




## Model

We have trained a state-of-the-art LLaMA model on ToolLLaMA dataset, see the performance in the [report](https://arxiv.org/abs/2305.14233). 
UltraLLaMA-13b and stronger models will be released soon.

We provide training code to fine-tune [LLaMa](https://github.com/facebookresearch/llama) (however we are not distributing the weights of LLaMa) on ToolLLaMA in [`.src/`](src), the training is accelerated by [BMTrain](https://github.com/OpenBMB/BMTrain).

- Download the released data and put it under `./data`

- Run `train_bm.py`, for example:

  ```bash
  WANDB_MODE="offline" torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:50003 train_bm.py --tensorboard ./ultrachat_llama_tb_2 --save_step 5000 --logging_step 100
  ```

We also provide a training script to fine-tune GPT-J on UltraChat in [`.src/train_legacy/`](src), which is implemented with [OpenPrompt](https://github.com/thunlp/OpenPrompt)

- Download the released data and put it under `./data`
- Run `accelerate launch train.py` to start training


## Fine-tuning

### Data preprocessing
- Download our newly released tool data and run the data_preprocess.py:
    ```bash
    python data_preprocess.py --tool_data_path tool_data/weather.json --output_path playground/data/weather_processed.json
    ```

### Training
- Run the training scripts:
    ```bash
    bash scripts/train_tool_llama_7b.sh
    ```

## Inference on BMTools Platform

### BMTools Setup
- First clone BMTools under current directory and prepare the setup settings of [BMTools](https://github.com/OpenBMB/BMTools):
    ```bash
    git clone git@github.com:OpenBMB/BMTools.git
    cd BMTools
    pip install --upgrade pip
    pip install -r requirements.txt
    python setup.py develop
    ```
- Then follow the setup instruction (2.1.1) in [BMTools](https://github.com/OpenBMB/BMTools) to build local tools environment:(Add your api keys to secret_keys.sh, then start the local tools)
    ```bash
    source secret_keys.sh
    python host_local_tools.py
    ```

### Inference with Your Fine-tuned ToolLLaMA
- Run the inference script:
    ```bash
    bash scripts/inference_tool_llama.sh
    ```
- Input your query in the terminal.


## Construction of ToolLLaMA

The general idea of ToolLLaMA is to train a LLM in our supervised data which then will support in [BMTools](https://github.com/OpenBMB/BMTools).
Each sector of ToolLLaMA has its own challenges and requires particular strategy designs. 
We will specify the construction process once a sector of ToolLLaMA is released.

### Model Experiment
- Machine Evaluation 
We randomly sample 100 chain steps in each tool to build our machine evaluation testbed. On average, there are 27 final steps and 73 intermediate tool calling steps. We evaluate the final steps with Rouge-L and the intermediate steps with ExactMatch.

| model_name                   | Downsampling | Beam size | Overall - Final Answer | Overall - Action | Overall - Input |
|------------------------------|--------------|-----------|------------------------|------------------|-----------------|
| cpmbee-finetuned             | 0.05         | 1         | **0.55**               | 0.64             | 0.40            |
| llama7b-finetuned            | 0.05         | 1         | 0.27                   | **0.77**         | 0.53            |
| vicuna7b-finetuned           | 0.05         | 1         | 0.42                   | 0.53             | 0.40            |
| llama7b-finetuned            | 0.5          | 1         | 0.35                   | 0.67             | 0.50            |
| llama7b-finetuned            | 0.7          | 1         | 0.29                   | 0.74             | **0.56**        |

- Human Evaluation
We randomly sample 10 query in each of the following tools: Weather, Map, Stock, Translation, Chemical and WolframAlpha. We evaluate the pass rate of tool calling process, final answer, and the final answer comparison with chatgpt.

| model_name                   | Downsampling | Beam size |  Tool Calling Process  |   Final Answer   |   Comparison   |
|------------------------------|--------------|-----------|------------------------|------------------|----------------|
| llama7b-finetuned            | 0.05         | 1         | **90%**                | **76.7%**        | 11.7%/60%/28.3%|


## News
- May 26, 2023: We release our first version of ToolLLaMA.

## To Do
- [ ] Release the rest part of the data for tools appered in BMTools.
- [ ] We will release our paper in arxiv soon.
- [ ] There will be a Chinese version of ToolLLaMA.


## Limitations

- Auto-generated data may contain hallucination and other formats of false facts. This issue will mainly appear in the first sector of the data. 
- To address the issue, more extensive post-processing will be conducted.

## Citation
Feel free to cite the repo if you think ToolLLaMA is useful.

```bibtex
@misc{ToolLLaMA,
  author = {Qin, Yujia and Liang, Shihao and Zhu, Kunlun and Tian, Runchu and Yan, Lan and etc.},
  title = {ToolLLaMA: An open-sourced dataset and model for Tool Learning},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/thunlp/ToolLLaMA},
}
```
# avayanlan.github.io
