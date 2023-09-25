import os
import torch

from transformers import AutoModel, AutoTokenizer, AutoConfig


model_path = "THUDM/chatglm-6b"
# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
prefix_state_dict = torch.load(
    os.path.join("output/adgen-chatglm2-6b-pt/checkpoint-300", "pytorch_model.bin"))

new_prefix_state_dict = {}

for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()


def display_answer(model, query, history=[]):
    for response, history in model.stream_chat(
            tokenizer, query, history=history):
    return history

if __name__ == '__main__':
    query = input("\n用户：")  # "类型#上衣\*材质#牛仔布\*颜色#白色\*风格#简约\*图案#刺绣\*衣样式#外套\*衣款式#破洞"
    answer = display_answer(model, query)
    print(answer)