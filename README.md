## SMAE
This is the code for "Learning Sentiment Memories for Sentiment Modification without Parallel Data".

## Environment and Dependency
 - Ubuntu 16.04
 - Python 3.5
 - Tensorflow 1.4
 - nltk 3.2.5
 
## Data
  [yelp](https://www.yelp.com/dataset/challenge)
  
## Usage
CUDA_VISIBLE_DEVICES=0 python3 main.py   
To run this code, you first need to process the dataset into the specific format. We provide the sample file. During running, several files will be created.

## Cite
If you use this code for your research, please cite the following paper:
```
  @inproceedings{zhang2018learning,  
  author = {Zhang, Yi and Xu, Jingjing and Yang, Pengcheng and Sun, Xu},  
  title = {Learning Sentiment Memories for Sentiment Modification without Parallel Data},  
  booktitle = {EMNLP 2018},  
  year = {2018}  
  }  
 ```
