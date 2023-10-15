from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk

# Sample post content
post_content = "Andrew Wiles was a mathematician who proved Fermat's last theorem. "

# Tokenize the post content into sentences
sentences = nltk.sent_tokenize(post_content)

# Convert sentences to a bag of words representation
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(sentences)

# Apply LDA
num_topics = 5  # You can adjust the number of topics as needed
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Get the top words for each topic
feature_names = vectorizer.get_feature_names_out()
topics = []
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-5:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    topics.append(top_words)

# Print a single set of 5 tags
print("Generated Tags:")
print(', '.join(topics[0]))
