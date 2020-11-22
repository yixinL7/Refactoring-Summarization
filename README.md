# Refactoring-Summarization
Code for our paper:
"RefSum: Refactoring Neural Summarization"

## Outline
* ### [Install](https://github.com/yixinL7/Refactoring-Summarization#how-to-install)
* ### [Train your Refactor](https://github.com/yixinL7/Refactoring-Summarization#how-to-run)
* ### [Off-the-shelf Tool: Refactoring your Models](https://github.com/yixinL7/Refactoring-Summarization#off-the-shelf-refactoring)
* ### [Dataset](https://github.com/yixinL7/Refactoring-Summarization#data)
* ### [Results](https://github.com/yixinL7/Refactoring-Summarization#results)




## 1. How to Install

### Requirements
- `python3`
- `conda create --name env --file spec-file.txt`
- `pip3 install -r requirements.txt`

### Description of Codes
- `main.py` -> training and evaluation procedure
- `model.py` -> Refactor model
- `data_utils.py` -> dataloader
- `utils.py` -> utility functions
- `preprocess.py` -> data preprocessing
- `demo.py` -> off-the-shelf refactoring


## 2. How to Run

### Hyper-parameter Setting
You may specify the hyper-parameters in `main.py`.
### Train
```
python main.py --cuda --gpuid [list of gpuid] -l
```
### Fine-tune
```
python main.py --cuda --gpuid [list of gpuid] -l --model_pt [model path]
```
### Evaluate
```
python main.py --cuda --gpuid [single gpu] -e --model_pt [model path] --model_name [model name]
```



## 3. Off-the-shelf Refactoring
You may use our model with you own data by running
```
python demo.py DATA_PATH MODEL_PATH RESULT_PATH
```
`DATA_PATH` is the path of you data, which should be a file of which each line is in json format: `{"article": str, "summary": str, "candidates": [str]}`. 

`RESULT_PATH` is the path of the result of which each line is a candidate summary.

## 4. Data
We use four datasets for our experiments.

- CNN/DailyMail -> https://github.com/abisee/cnn-dailymail
- XSum -> https://github.com/EdinburghNLP/XSum
- PubMed -> https://github.com/armancohan/long-summarization
- WikiHow -> https://github.com/mahnazkoupaee/WikiHow-Dataset

You can find the processed data for all of our experiments here [TODO: ADD LINK]. After downloading, you should put the data in `./data` directory.

## 5. Results


### CNNDM
#### Reranking BART
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| BART     | 44.26   | 21.12   | 41.16   |
| Refactor | 45.15   | 21.70   | 42.00   |

#### Reranking GSum
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| GSum     | 45.93   | 22.30   | 42.68   |
| Refactor | 46.18   | 22.36   | 42.91   |

#### System-Combination (BART and pre-trained Refactor)
|                            | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------------------------|---------|---------|---------|
| BART                       | 44.26   | 21.12   | 41.16   |
| pre-trained Refactor       | 44.13   | 20.51   | 40.29   |
| Summary-Level Combination  | 45.04   | 21.61   | 41.72   |
| Sentence-Level Combination | 44.93   | 21.48   | 41.42   |






