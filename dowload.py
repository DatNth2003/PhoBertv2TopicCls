from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "vinai/phobert-base-v2"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./phobert-v2")
tokenizer.save_pretrained("./phobert-v2")
