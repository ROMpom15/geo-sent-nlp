# cluster.py
# S. Conrad Jones

from mybert import embed_sentence
from data import kmeans, get_review_sentences
import numpy as np
import matplotlib.pyplot as plt   
from wordcloud import WordCloud

# Develop shape and lengths
n = 1000 # number of review sentences
reviews = get_review_sentences(N=n)    # returns n review sentences as a List of strings
len_np_vec = len(reviews)
your_numpy_matrix = np.empty(shape=(n,768))

# Fill in all the rows with your DeBERTa embeddings
for i in range(n):
  rev_vec = embed_sentence(reviews[i])	    
  numpyvec = rev_vec.detach().numpy()
  n_vec = np.reshape(numpyvec, (1,len(numpyvec))) # https://numpy.org/devdocs/reference/generated/numpy.reshape.html
  your_numpy_matrix[i] = n_vec

# Cluster and print clusters

# OPTION 1: Runs k-means cluster with hard-coded k-clusters
k = 23
labels = kmeans(your_numpy_matrix, k=k)   # labels each row with a cluster ID
# OPTION 2: Runs agglomerative clustering that discovers the \# of clusters
# import sklearn.cluster
# agg = sklearn.cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=0.15, affinity="cosine", linkage="average", compute_distances=True)
# labels = agg.fit(your_numpy_matrix).labels_
# k = len(labels)
lab_list = [[]]* k # https://stackoverflow.com/questions/65473101/why-is-10-the-way-to-create-a-list-of-10-empty-lists-in-python-instead-of
with open('clusters.txt','a') as f: # https://www.w3schools.com/python/python_file_write.asp
  for j in range(k):
    # write the headers
    h = f'\n***** Cluster {j} *****\ncontent\n'
    print(h)
    f.write(f'{h}\n')
    lab_list[j].append(h)
    for i in range (len(labels)):
      # write the reviews
      if labels[i] == j:
        print(reviews[i])
        f.write(f'{reviews[i]}\n')
        lab_list[j].append(reviews[i])
    
while True:
  inp = (input('Enter a cluster number [0-9] to view as a cloud: '))
  if inp == 'quit':
    print('Goodbye!')
    break
  text = str()
  textl = lab_list[int(inp)]
  for sent in textl:
    if sent.startswith('\n*'):
      pass
    else:
      text = f'{text} {sent}'
  text = text.replace(r'app | task | \n','')
  cloud = WordCloud(width=480, height=480, margin=0).generate(text)    

  # Now popup the display of our generated cloud image.
  plt.imshow(cloud, interpolation='bilinear')
  plt.axis("off")
  plt.margins(x=0, y=0)
  plt.savefig(f'cloud_num_{inp}.png') #https://stackoverflow.com/questions/77507580/userwarning-figurecanvasagg-is-non-interactive-and-thus-cannot-be-shown-plt-sh
  