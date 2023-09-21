from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/mnt/workspace/THUDM/chatglm2-6b", trust_remote_code=True).cuda()
model = model.eval()
response,history = model.chat(tokenizer, "你好", history=[])
print(response)
response,history = model.chat(tokenizer, "晚上睡不着怎么办", history=history)
print(response)
