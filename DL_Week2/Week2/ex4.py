import numpy as np
import re
from collections import Counter
# Open file
with open('story.txt','r') as file:
    text = file.read()

#Remove punctuation characters
text = re.sub(r'[^\w\s]', '',text)

#Split text into words
words = text.lower().split()

#Count the occurence of each word
count = Counter(words)

#Print the top 100
top = count.most_common(100) #Use this command to acquire the top 100 word
for word,count in top:
    print(f'{word}:{count}')