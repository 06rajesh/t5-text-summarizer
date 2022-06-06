# t5-text-summarizer

implementation of t5 huggingface model for summarizing texts using pytorch and pytorch lighting.
<br/>`ipynb` file. Ready to run on google colab.

T5 huggingface: https://huggingface.co/docs/transformers/model_doc/t5 <br/>
Kaggle News Summary Dataset: https://www.kaggle.com/datasets/sunnysai12345/news-summary

### Example:
#### Text:
```
Bollywood actor Kangana Ranaut is surely one talented woman. After wooing the heart of the nation 
with her epic performance in 'Queen', the actor is now reportedly all set to make her directorial 
debut.Ms Ranaut who has returned from the New York Film Academy where she completed a two month 
course on screenplay writing, seems to be in no mood for wasting time. After having recently 
produced and directed a short feature film called The Touch, she now has a chick flick on her 
mind.It has also been added that Kangana will be taking a lot of inspiration from her personal 
experiences and will not venture into the business of remakes.
```

#### Output
```
Actress Kangana Ranaut is set to make her directorial debut after wooing the heart of the nation 
with her epic performance in 'Queen'. The actress has returned from New York Film Academy where 
she completed a two month course on screenplay writing. She will also be taking inspiration from 
her personal experiences and will not venture into the business of remakes.
```