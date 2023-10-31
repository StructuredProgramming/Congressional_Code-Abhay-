from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import MultifieldParser
from whoosh import qparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
posts_visited = [
    ("Integration", "Integration is an important part of calculus, forming a strong foundation for the subject with a variety of intricacies."),
    ("Coffee", "Coffee is a very good liquid for consumption, providing lots of energy at a relatively low cost."),
    ("Trains", "I like trains.")
]

total_posts = [
    ("Topelitz Square", "The Topelitz square is one of the most difficult problems in mathematics, with a variety of interesting proposed solutions."),
    ("Cancer Research", "Cancer is one of the most deadly diseases in the world, but thanks to the work of incredible doctors there a wide variety of available treatments."),
    ("Phineas and Ferb", "Phineas and Ferb is the greatest TV show of all time, with a lot of interesting storylines.")
]

# Schema definition for the Whoosh index
schema = Schema(title=TEXT(stored=True), body=TEXT(stored=True), post_id=ID(stored=True))

# Create or open the Whoosh index
index_dir = "C:\\users\\abhay\\Whoosh"
if not index.exists_in(index_dir):
    ix = create_in(index_dir, schema)
else:
    ix = open_dir(index_dir)

# Indexing total_posts
with ix.writer() as writer:
    for idx, (title, body) in enumerate(total_posts):
        writer.add_document(title=title, body=body, post_id=str(idx))

# Extract content from posts_visited and total_posts
visited_content = [content for _, content in posts_visited]
all_content = [content for _, content in total_posts]

# Convert content to numerical vectors using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
content_vectors = vectorizer.fit_transform(all_content)

# Compute cosine similarity between visited posts and all posts
visited_vectors = vectorizer.transform(visited_content)
cosine_similarities = cosine_similarity(visited_vectors, content_vectors)

# Get indices of all posts and sort them based on similarity
all_post_indices = range(len(total_posts))
sorted_indices = sorted(all_post_indices, key=lambda x: cosine_similarities[0][x], reverse=True)

# Get recommended posts based on sorted indices (excluding visited posts)
recommended_posts = [(total_posts[i][0], total_posts[i][1]) for i in sorted_indices if total_posts[i] not in posts_visited]

# Print recommended posts
print("Recommended Posts:")
for title, content in recommended_posts:
    print("Title:", title)
    print("Content:", content)
    print("----")
