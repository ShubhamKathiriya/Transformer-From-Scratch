# Transformer 

## Implemented [**Attention Is All You Need**](https://arxiv.org/pdf/1706.03762) from Scratch


## Main Architecture

<div style="display: flex; justify-content: space-between;">
  <img src="https://vitalflux.com/wp-content/uploads/2023/08/encoder-decoder-architecture-2-768x359.png" alt="Image 1" width="600" height ="500" >
  <img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" alt="Image 2" width="200" height ="500">
</div>





## Self Attention

<div style="text-align: center;"">
  <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*GIVM8Wat6Vq8W7Eff-f_5w.png" alt="Image 1" width="800" height ="500" >
</div>


## File Description:-

#### 1. config.py -> It has all parameter values and file paths those has been use in model
#### 2. utils.py -> General Purpose function like validation, Dataset and model specific function like multihead attention, etc.
#### 3. tokenization.py -> contain BPE tokenization and it's helper function
#### 4. Encoder.py -> Contain Architecture of encoder
#### 5. Decoder.py -> Contain Architecture of decoder
#### 6. transformer.py -> Contain Architecture of transformer
#### 7. train.py -> for train the transformer model
#### 8. test.py -> for test the transformer model

#### 9. ted-talks-corpus -> cotaining english, french Dataset in train,dev,test split

## Task:- 
- English to French Machine Translation


## Command

### For training (as config file )
```python
python training.py
```


### For testing (as config file )
```python
python test.py
```



## Reference:-
#### 1. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
#### 2. [Jay Alammar Transformer Blog](https://jalammar.github.io/illustrated-transformer/)
#### 3. [HarvardNLP The Annotated Transformer Blog](https://nlp.seas.harvard.edu/2018/04/03/attention.html#model-architecture)