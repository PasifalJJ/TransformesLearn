import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
#
# # 指定检查点，检查点实际就是最后训练数据的存档
# checkpoint = "bert-base-uncased"
# # 加载分词器/标记器
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# # 加载模型
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# # 测试训练句子
# sequences = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "This course is amazing!",
# ]
# # 对数据进行编码
# batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
# print(batch)
# batch["labels"] = torch.tensor([1, 1])
#
# # AdamW为一个优化器 主要是用来更新模型参数 目的最小化损失函数。
# # 损失函数：用于衡量模型预测值与真实值之间的差异。
# # 训练模型的过程实际上就是寻找一组模型参数，使得损失函数的值最小
# optimizer = AdamW(model.parameters())
# # loss的目的是对传入的数据进行前向传播来计算损失. 损失值为模型预测和真实标签的差异度量
# loss = model(**batch).loss
# # 执行反向传播计算 计算损失函数对每个参数的梯度
# loss.backward()
# # 使用优化器根据梯度更新模型的参数 优化器根据梯度下降,更新每一个参数,减少损失函数的值
# optimizer.step()

from datasets import load_dataset
# 加载数据集
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

# 打印train 0数据
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])

'''

DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
'''


