from datasets import load_dataset

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
print(dataset[0])