# Predicting Customer Ratings

People write about their opinions and experiences with products and services all the time. Wouldn’t it be nice to quantify the positivity or negativity of their words so that businesses could streamline their responsiveness to consumer preferences? In my project, I sought to predict how consumers would rate products, services and businesses based on the text of a review they left.

The main question I was trying to answer was if it’s possible to predict a positive or negative customer review given the text of the review. Specifically, could I create an accurate classification model that correctly differentiates between 1, 2, 3, 4, and 5-star reviews based on the text of the review.

Data was obtained from two Kaggle repositories. The Yelp data is 1,000 reviews by 1,000 different users about 1,000 different businesses. The data obtained from Kaggle was a small portion of the very large Yelp dataset that I initially tried to work with. According to its Kaggle dataset curator, the Amazon dataset consists of reviews from October 1999 to October 2012. Included are 568,454 reviews by 256,059 users about 74,258 products. Overall, the combined dataset has more 5 star reviews than all other categories combined, but There was still a good enough variety of data to put into the models.

This data helped me answer my question because the data included real product or service reviews and how they scored the product or service, so by looking at the words used by real people I was able to tell over time the common words or phrases indicating positive or negative reviews and even the magnitude of positivity or negativity with reasonable accuracy.

## Combining the data


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
data = pd.read_csv("amazon/Reviews.csv")
```


```python
amzn = data[["Score", "Text"]]
amzn.columns = ["stars", "text"]
```


```python
dataY = pd.read_csv("yelp.csv")
yelp = dataY[["stars", "text"]]
```


```python
df = amzn.append(yelp).reset_index(drop = True)
```

Not much data cleaning had to occur other than the removal of English stop words, but a key part in the analysis was creating a numeric representation of text sentiment. 

For this, I used the VADER sentiment analysis tool (https://github.com/cjhutto/vaderSentiment) which was designed for analyzing social media posts. 

I decided to use this method because I’m assuming that people post similarly on social media and in online reviews since in both situations people are writing to a semi-unknown audience in a casual setting. 

The VADER compound score that was used is a number in the range (-1, 1) where scores at the edges of the range are more extremely positive or negative, and scores close to zero are generally neutral.

## Feature Creation: Getting the VADER Compound Sentiment Score


```python
from nltk import wordpunct_tokenize, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
```


```python
tokens = df['text'].apply(lambda x: wordpunct_tokenize(x))
df['tokens'] = tokens
```


```python
sw = stopwords.words('english')
```


```python
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in sw])
```


```python
sid = SentimentIntensityAnalyzer()
```


```python
df['scores'] = df['tokens'].apply(lambda x: sid.polarity_scores(" ".join(x))['compound'])
```

Once I calculated the sentiment score, this what the dataframe looked like.


```python
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stars</th>
      <th>text</th>
      <th>tokens</th>
      <th>scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>I have bought several of the Vitality canned d...</td>
      <td>[I, bought, several, Vitality, canned, dog, fo...</td>
      <td>0.9413</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Product arrived labeled as Jumbo Salted Peanut...</td>
      <td>[Product, arrived, labeled, Jumbo, Salted, Pea...</td>
      <td>0.0762</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>This is a confection that has been around a fe...</td>
      <td>[This, confection, around, centuries, ., It, l...</td>
      <td>0.8073</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>If you are looking for the secret ingredient i...</td>
      <td>[If, looking, secret, ingredient, Robitussin, ...</td>
      <td>0.4404</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Great taffy at a great price.  There was a wid...</td>
      <td>[Great, taffy, great, price, ., There, wide, a...</td>
      <td>0.9468</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>578449</th>
      <td>3</td>
      <td>First visit...Had lunch here today - used my G...</td>
      <td>[First, visit, ..., Had, lunch, today, -, used...</td>
      <td>0.8807</td>
    </tr>
    <tr>
      <th>578450</th>
      <td>4</td>
      <td>Should be called house of deliciousness!\n\nI ...</td>
      <td>[Should, called, house, deliciousness, !, I, c...</td>
      <td>0.9925</td>
    </tr>
    <tr>
      <th>578451</th>
      <td>4</td>
      <td>I recently visited Olive and Ivy for business ...</td>
      <td>[I, recently, visited, Olive, Ivy, business, l...</td>
      <td>0.9937</td>
    </tr>
    <tr>
      <th>578452</th>
      <td>2</td>
      <td>My nephew just moved to Scottsdale recently so...</td>
      <td>[My, nephew, moved, Scottsdale, recently, bunc...</td>
      <td>-0.9491</td>
    </tr>
    <tr>
      <th>578453</th>
      <td>5</td>
      <td>4-5 locations.. all 4.5 star average.. I think...</td>
      <td>[4, -, 5, locations, .., 4, ., 5, star, averag...</td>
      <td>0.9628</td>
    </tr>
  </tbody>
</table>
<p>578454 rows × 4 columns</p>
</div>


There was enough data in each star category in general to train models, but one might be interested in the distribution of star ratings in the datasets and combined.

The Amazon dataset was many times larger than the Yelp dataset, so the distribution of the combined dataset is dominated by the Amazon data distribution of star ratings.

Below we see first the Amazon distribution of star ratings followed by the Yelp distribution, and finally the combined dataset.


```python
plt.hist(amzn['stars'], bins=10)
plt.title("Distribution of Amazon Star Ratings"); plt.xlabel("stars"); plt.ylabel("count"); plt.ylim((0,400000)); plt.show()

plt.hist(yelp['stars'], bins=10)
plt.title("Distribution of Yelp Star Ratings"); plt.xlabel("stars"); plt.ylabel("count"); plt.ylim((0,400000)); plt.show()

plt.hist(df['stars'], bins=10)
plt.title("Combined Distribution of Star Ratings"); plt.xlabel("stars"); plt.ylabel("count"); plt.ylim((0,400000)); plt.show()
```




I used Naïve Bayes and Random Forest models to predict star rating. 

The Naïve Bayes model was used as a baseline model because it would be fast, and I wouldn’t have to tune any parameters. 

Random forests were helpful because I was more familiar with this method and that would aid in the interpretation of the models. 

I first analyzed the original dataset with all 5 categories as targets and then simplified the targets to either a positive or a negative review by removing neutral 3-star review, putting 1 and 2-star reviews together, and putting 4 and 5-star reviews together. 

Here is the code for the two sets of models:

## Models to predict star rating


```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
```


```python
train, test = train_test_split(df, test_size=.3, stratify=df.stars, random_state=315)
```

### Naive Bayes


```python
nb = MultinomialNB()

nb.fit(np.array(train['scores'] + 1).reshape(-1, 1), train['stars'])

yhat11 = nb.predict(np.array(test['scores'] + 1).reshape(-1, 1))
```



### Random forests


```python
rfc = RandomForestClassifier()

rfc.fit(np.array(train['scores']).reshape(-1, 1), train['stars'])

yhat12 = rfc.predict(np.array(test['scores']).reshape(-1, 1))
```

## Models to detect a positive star rating

In this section I used a different target variable - 'positive' to train models to differentiate between negative reviews (1&2 stars) and positive reviews (4&5 stars).


```python
df_no_3 = df.loc[df['stars'] != 3,:].reset_index()
df_no_3['positive'] = df_no_3['stars'].apply(lambda x: 1 if x > 3 else 0)
train_2, test_2 = train_test_split(df_no_3, test_size=.3, stratify=df_no_3['positive'], random_state=754)
```

### Naive Bayes


```python
nb = MultinomialNB()

nb.fit(np.array(train_2['scores'] + 1).reshape(-1, 1), train_2['positive'])

yhat21 = nb.predict(np.array(test_2['positive'] + 1).reshape(-1, 1))
```

### Random forests


```python
rfc = RandomForestClassifier()

rfc.fit(np.array(train_2['scores'] + 1).reshape(-1, 1), train_2['positive'])

yhat22 = rfc.predict(np.array(test_2['positive'] + 1).reshape(-1, 1))
```

## Model Evaluation

To evaluate all models, I looked at the confusion matrices and calculated the model accuracy. Accuracy is defined to be the total number of correct classifications divided by the total number of classifications. That was found by dividing the sum of the values on the diagonal by the sum of all numbers in the matrix. 

As can be seen below, the models heavily classified toward 5-star ratings or positive ratings. This is most likely a result of the skewed contents of the training dataset. Although I stratified the sample from the dataset, there was an overwhelming majority of 5-star ratings which caused an extreme imbalance in the data seen by the models. For this reason, even though the models’ accuracy was high, it was because the models only really learned how to classify a review as a 5-star review, and because most of the time the models were exposed to 5-star reviews, most of the time they were right! 

In fact, I believe the only reason that the accuracy is higher for the positive / negative differentiation is because 3-star reviews were removed which slightly lowered the number of false positives.



```python
m11 = confusion_matrix(test['stars'], yhat11); acc11 = m11[4,4] / m11.sum() #Accuracy
m12 = confusion_matrix(test['stars'], yhat12); acc12 = (m12[0,0] + m12[1,1]+ m12[2,2]+ m12[3,3]+ m12[4,4]) / m12.sum() #Accuracy
m21 = confusion_matrix(test_2['positive'], yhat21); acc21 = (m21[0,0] + m21[1,1]) / m21.sum() #Accuracy
m22 = confusion_matrix(test_2['positive'], yhat22); acc22 = (m22[0,0] + m22[1,1]) / m22.sum() #Accuracy
```


```python
print("5-STAR:\t   Naive Bayes\n\n", m11, "\n\n\tAccuracy: ", acc11)
print("\n\n5-STAR:\t   Random Forests\n\n", m12, "\n\n\tAccuracy: ", acc12)
print("\n\n\n(Pos/Neg): Naive Bayes\n\n", m21, "\n\n\tAccuracy: ", acc21)
print("\n\n(Pos/Neg): Random Forests\n\n", m22, "\n\n\tAccuracy: ", acc22)
```

    5-STAR:	   Naive Bayes
    
     [[     0      0      0      0  15905]
     [     0      0      0      0   9209]
     [     0      0      0      0  13230]
     [     0      0      0      0  25255]
     [     0      0      0      0 109938]] 
    
    	Accuracy:  0.6335133141635502
    
    
    5-STAR:	   Random Forests
    
     [[  5299    320    275    241   9770]
     [  1315    516    147    131   7100]
     [   986    119    579    161  11385]
     [   757    106    120    585  23687]
     [  2042    341    411    572 106572]] 
    
    	Accuracy:  0.6543330817059186
    
    
    
    (Pos/Neg): Naive Bayes
    
     [[     0  25114]
     [     0 135192]] 
    
    	Accuracy:  0.8433371177622796
    
    
    (Pos/Neg): Random Forests
    
     [[     0  25114]
     [     0 135192]] 
    
    	Accuracy:  0.8433371177622796





## Conclusion

These results obviously present an unsatisfying answer to the original research question of whether or not a model can be made which, with reasonable accuracy, classifies reviews correctly based on the text of the review. My best models had 84% accuracy, but as I wrote earlier, these models aren’t robust by any means.

The strength in this project came from the low dimensionality of the classification task. Essentially, using the VADER method I took reviews and mapped them to a line segment, and then I trained models to divide the line segment into 5 regions representing the five-star ratings. However, the VADER method that allows for this simplicity is possibly the biggest weakness of my project because it requires the assumption that a piece of text can be assigned a scaler and that writing styles in social media contexts and online Amazon/Yelp contexts are the same.

To improve this project, I would want to either find a more balanced dataset or cut back on the number of 5-star reviews in this current dataset. I would also want to have equal source representation between Amazon and Yelp, and maybe draw from more sources of online review text. I would also better familiarize myself with the VADER method of text sentiment analysis to make sure it is indeed appropriate for this task and explore their other three-dimensional measures of text sentiment instead of the compound score to see if there are any additional advantages.
