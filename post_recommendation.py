from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

posts_visited = [
    ("Integration", "Integration is an important part of calculus, forming a strong foundation for the subject with a variety of intricacies."),
    ("Coffee","Coffee is a very good liquid for consumption, providing lots of energy at a relatively low cost."),
    ("Trains","I like trains.")
]

user_posts = [" ".join([title, content]) for (title, content) in posts_visited]

# Titles and content of all posts excluding posts already viewed
total_posts = [
    ("Topelitz Square", "The topelitz square is one of the most difficult problems in mathematics, with a variety of interesting proposed solutions."),     
    ("Cancer Research", "Cancer is one of the most deadly diseases in the world, but thanks to the work of incredible doctors there a wide variety of available treatments."),
    ("Phineas and Ferb", "Phineas and Ferb is the greatest TV show of all time, with a lot of interesting storylines.")]

all_posts = [" ".join([title, content]) for (title, content) in total_posts]

# Create TfidfVectorizer to convert text to numerical vectors
vectorizer = TfidfVectorizer(stop_words='english')
post_vectors = vectorizer.fit_transform(all_posts + user_posts)  # Include user posts for vectorization

# Compute cosine similarity between user posts and all posts
user_post_vectors = vectorizer.transform(user_posts)
cosine_similarities = cosine_similarity(user_post_vectors, post_vectors[:-len(user_posts)])  # Exclude user posts for similarity calculation

# Get indices of all posts and sort them based on similarity
all_post_indices = range(len(total_posts))
sorted_indices = sorted(all_post_indices, key=lambda x: cosine_similarities[0][x], reverse=True)

# Get recommended posts based on sorted indices
recommended_posts = [(total_posts[i][0], total_posts[i][1]) for i in sorted_indices]

# Print recommended posts
print("Recommended Posts:")
for title, content in recommended_posts:
    print("Title:", title)
    print("Content:", content)
    print("----")
