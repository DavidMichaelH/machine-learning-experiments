import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Read the labled, unlabled, and sample data from the specified files
labeled = pd.pandas.read_csv("data/labeled_data.csv")
unlabeled = pd.pandas.read_csv("data/unlabeled_data.csv")
sample = pd.pandas.read_csv("data/sample_submission.csv")

# add a new column, text_length, to the labeled dataframe.
# The values in this column are the length of each text in the text column
labeled['text_length'] = labeled.text.apply(lambda x: len(x.split(' ')))

# This code is printing the minimum and maximum values in the text_length column of the labeled dataframe.
print("min =", labeled['text_length'].min(), ", max =", labeled['text_length'].max())

# Get a list of tweets genuinely referring to a disaster
disaster_tweets = list(labeled[labeled["target"] == 1].text.values)

# Create a word cloud of the disaster tweets to illustrate word frequency for this population.
wordcloud = WordCloud(stopwords=STOPWORDS).generate(str(disaster_tweets))

plt.imshow(wordcloud)
plt.axis("off")
plt.show()
