# Ex8: Let user type 2 words in English as input. Print out the output
# which is the shortest chain according to the following rules:
# - Each word in the chain has at least 3 letters
# - The 2 input words from user will be used as the first and the last words of the chain
# - 2 last letters of 1 word will be the same as 2 first letters of the next word in the chain
# - All the words are from the file wordsEn.txt
# - If there are multiple shortest chains, return any of them is sufficient

# Begin of the program
# Enter the words
import random
word1 = input("Enter the input: ")
word2 = input("Enter the output: ")

# Read the wordsEn.txt
with open('/home/hiromi01/Downloads/wordsEn.txt', 'r') as f:
    words = f.read().splitlines()

# Filter the word with 3 letters
words = [w for w in words if len(w) >= 3]

# Filter the chain that has 2 last letters of 1 word will be the same as 2 first letters of the next word in the chain
chains = []
for w1 in words:
    if w1[:2] != word1[-2:]:
        continue
    for w2 in words:
        if w2[:2] != w1[-2:] or w2[-2:] != word2[:2]:
        continue
        chains.append([word1, w1, w2, word2])

# Filter the shortest chain
shortest_chains = []
shortest_len = float('inf')
for chain in chains:
    if len(chain) < shortest_len:
        shortest_chains = [chain]
        shortest_len = len(chain)
    elif len(chain) == shortest_len:
        shortest_chains.append(chain)

# Print the chain
if shortest_chains:
    chain = random.choice(shortest_chains)
    print('Shortest chain:', ' -> '.join(chain))
else:
    print('No chain found')