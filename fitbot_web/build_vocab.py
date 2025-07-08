import json 
from collections import Counter

with open("big_fitness_dataset.txt", "r", encoding="utf-8") as f: lines = f.readlines()

questions, answers = [], [] 
for i in range(0, len(lines), 2): 
    if i+1 < len(lines) and lines[i].startswith("Q:") and lines[i+1].startswith("A:"): 
        q = lines[i].replace("Q:", "").strip().lower() 
        a = lines[i+1].replace("A:", "").strip().lower() 
        questions.append(q) 
        answers.append(a)

special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"] 
word_freq = Counter(" ".join(questions + answers).split()) 
vocab = special_tokens + sorted(word_freq.keys()) 
word2index = {word: idx for idx, word in enumerate(vocab)}

with open("model/vocab.json", "w", encoding="utf-8") as f: json.dump(word2index, f, ensure_ascii=False, indent=2)

print(f"vocab.json saved to model/ with {len(word2index)} tokens.")
