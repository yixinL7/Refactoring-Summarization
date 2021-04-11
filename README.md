# Refactoring-Summarization
Code for our paper:
"RefSum: Refactoring Neural Summarization", NAACL 2021.

<img src="https://github.com/yixinL7/Refactoring-Summarization/blob/main/intro-gap.png" width="500">

We present a model, Refactor, which can be used either as a base system or a meta system for text summarization.
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

You can find the processed data for all of our experiments [here](https://drive.google.com/drive/folders/1QvlxYVyEN1tGzzzNrfAcNIui56qdhezL?usp=sharing). After downloading, you should put the data in `./data` directory.

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Experiment</th>
    <th>Link</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="6">CNNDM</td>
    <td>Pre-train</td>
    <td><a href="https://drive.google.com/file/d/1kcwR0PswyBXWGrNJBcg7Et65keSSsXoc/view?usp=sharing" target="_blank" rel="noopener noreferrer">Download</a></td>
  </tr>
  <tr>
    <td>BART Reranking</td>
    <td><a href="https://drive.google.com/file/d/1GfwqDpFBPV3jOaCUtGRt8FRlUak9YzyV/view?usp=sharing" target="_blank" rel="noopener noreferrer">Download</a></td>
  </tr>
  <tr>
    <td>GSum Reranking</td>
    <td><a href="https://drive.google.com/file/d/1hue7r7tU-9o1pnNuHC6wDV4bCFpwtK95/view?usp=sharing" target="_blank" rel="noopener noreferrer">Download</a></td>
  </tr>
  <tr>
    <td>Two-system Combination (System-level)</td>
    <td><a href="https://drive.google.com/file/d/1WIf9WvKX90fHxVCR5ywb0Kd5mZJgu9cz/view?usp=sharing" target="_blank" rel="noopener noreferrer">Download</a></td>
  </tr>
  <tr>
    <td>Two-system Combination (Sentence-level)</td>
    <td><a href="https://drive.google.com/file/d/1z0EFkOtTXriarv7tR3KY3D_Sssx4yHEQ/view?usp=sharing" target="_blank" rel="noopener noreferrer">Download</a></td>
  </tr>
  <tr>
    <td>Three-system Combination (System-level)</td>
    <td><a href="https://drive.google.com/file/d/1sklrdsA_UxNAYeK1helUJ_ZdhdcltRZz/view?usp=sharing" target="_blank" rel="noopener noreferrer">Download</a></td>
  </tr>
  <tr>
    <td rowspan="2">XSum</td>
    <td>Pre-train</td>
    <td><a href="https://drive.google.com/file/d/1fSPJDmkBakYcfOhAF_UlLCbThR6h1O74/view?usp=sharing" target="_blank" rel="noopener noreferrer">Download</a></td>
  </tr>
  <tr>
    <td>PEGASUS Reranking</td>
    <td><a href="https://drive.google.com/file/d/1ZqdooQ4YwwRg4qab3lEUu-Wr7NV11gKe/view?usp=sharing" target="_blank" rel="noopener noreferrer">Download</a></td>
  </tr>
  <tr>
    <td rowspan="2">PubMed</td>
    <td>Pre-train</td>
    <td><a href="https://drive.google.com/file/d/1l_LmeNPRTv_L9GPctFYNZVp5gp0t7DDG/view?usp=sharing" target="_blank" rel="noopener noreferrer">Download</a></td>
  </tr>
  <tr>
    <td>BART Reranking</td>
    <td><a href="https://drive.google.com/file/d/1lW3VefPnPs664qy5o4Qub9IpIH2YfWHt/view?usp=sharing" target="_blank" rel="noopener noreferrer">Download</a></td>
  </tr>
  <tr>
    <td rowspan="2">WikiHow</td>
    <td>Pre-train</td>
    <td><a href="https://drive.google.com/file/d/1p2Us8qvKqwgQcE6ZIUR5-umMtBxGJ2ef/view?usp=sharing" target="_blank" rel="noopener noreferrer">Download</a></td>
  </tr>
  <tr>
    <td>BART Reranking</td>
    <td><a href="https://drive.google.com/file/d/1HELUaZm4FpOXZ1hF5n4nqtsDyNHUygZL/view?usp=sharing" target="_blank" rel="noopener noreferrer">Download</a></td>
  </tr>
</tbody>
</table>

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

#### System-Combination (BART, pre-trained Refactor and GSum)
|                            | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------------------------|---------|---------|---------|
| BART                       | 44.26   | 21.12   | 41.16   |
| pre-trained Refactor       | 44.13   | 20.51   | 40.29   |
| GSum                       | 45.93   | 22.30   | 42.68   |
| Summary-Level Combination  | 46.12   | 22.46   | 42.92   |

### XSum
#### Reranking PEGASUS
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| PEGASUS  | 47.12   | 24.46   | 39.04   |
| Refactor | 47.45   | 24.55   | 39.41   |

### PubMed
#### Reranking BART
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| BART     | 43.42   | 15.32   | 39.21   |
| Refactor | 43.72   | 15.41   | 39.51   |

### WikiHow
#### Reranking BART
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| BART     | 41.98   | 18.09   | 40.53   |
| Refactor | 42.12   | 18.13   | 40.66   |




