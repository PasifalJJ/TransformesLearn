from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoding = tokenizer( "我们很高兴向您展示🤗 Transformers 库。" )
print(encoding)

pt_batch = tokenizer(
    ["我们非常高兴向您展示 🤗 Transformers 库。" , "我们希望您不要讨厌它。"],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
pt_outputs = pt_model(**pt_batch)
print(pt_outputs)

from torch import nn
pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)

from transformers import AutoConfig
my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)

from transformers import AutoModel
my_model = AutoModel.from_config(my_config)