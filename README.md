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
To run this code, you first need to process the dataset into the specific format. We provide the sample file in dataset to facilitate runing code. But these files are just a small piece of the complete training set and are not enough to get a satisfied performance. You can process the data provided by yelp to get enough training set. During running, several files will be created.

The full dataset we used in our experiments can be found here: [full_data](https://drive.google.com/drive/folders/1NdmsZi221PPb3Uh5tyLpm_yCxZdzaXxi?usp=sharing)

Feel free to contact me if you have any questions:)

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
