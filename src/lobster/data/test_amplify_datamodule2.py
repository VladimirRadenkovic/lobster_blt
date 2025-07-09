from transformers import AutoModel
from transformers import AutoTokenizer
from datasets import load_dataset

# Load AMPLIFY and tokenizer
#model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True)

# Move the model to GPU (required due to Flash Attention)
#model = model.to("cuda")
model = ''
tokenizer = ''
# Load the UniProt validation set
dataset = load_dataset("chandar-lab/UR100P", data_dir="UniProt", split="test", streaming=True)

for sample in dataset:
    # Protein
    print("Sample: ", sample["name"], sample["sequence"])

    # Tokenize the protein
    input = tokenizer.encode(sample["sequence"], return_tensors="pt")
    print("Input: ", input)

    # Move to the GPU and make a prediction
    #input = input.to("cuda")
    #output = model(input)
   # print("Output: ", output)

    #break
