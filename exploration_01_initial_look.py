from datasets import load_dataset

# Downloads CUAD (cached locally after first run)
dataset = load_dataset("theatticusproject/cuad-qa", revision="refs/pr/6")

print("Dataset structure:")
print(dataset)

print("\n--- First example ---")
example = dataset["train"][0]
for key, value in example.items():
    if isinstance(value, str) and len(value) > 300:
        print(f"{key}: {value[:300]}... [truncated, full length: {len(value)}]")
    else:
        print(f"{key}: {value}")
