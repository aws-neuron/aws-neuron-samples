from transformers import pipeline

classifier = pipeline("text-classification", model = "./results/")

print(classifier("大変すばらしい商品でした。感激です。"))
print(classifier("期待していた商品とは異なりました。残念です。"))
