# HuffPost article analysis

We applied Natural Language Processing (NLP) techniques to analyze 200k HuffPost articles from Jan 2012 to May 2018 [**here is what we found**](https://docs.google.com/presentation/d/1P-vAfgQsC63P3sMW9y8pGnOvvKWK-ntkp_nhodvwwG0/edit?usp=sharing)

![image](final-project/images/wordcloud.png)

## Main results

### Evolution of discusson topics per year

![image](final-project/images/topics_by_year.png)

We can see that the articles have become increasingly politicized over time, with the topic of President Trump becoming more and more prominent. Also, we can see that the topic did not peak in 2016, when there was an election, but continued to grow until the end of the data in 2018.

### Article classification
Each article is labeled in one of 29 categories, corresponding to the section of the website where the article was published.
![image](final-project/images/top_10_categories.png)

This allows us to perform text-classification of the news articles, in particular we tried the following methods:
* Naive bayes and Logistic regression are very fast to train and provide good results.
* CNN/LSTM are very slow to train and their results don't compete with other methods.
* The transformer model is a fine-tuning of DistilBERT and provides the best results but with high training and evaluation times

![image](final-project/images/metrics.png)
