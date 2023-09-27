from transformers import AutoModel, AutoTokenizer


model_path = "/mnt/workspace/THUDM/chatglm-6b"
# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# 微调前
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
model = model.eval()


def display_answer(model, query, history=[]):
    for response, history in model.stream_chat(tokenizer, query, history=history):
    return history


if __name__ == '__main__':
    query = input("\n用户:")  # "类型#上衣\*材质#牛仔布\*颜色#白色\*风格#简约\*图案#刺绣\*衣样式#外套\*衣款式#破洞"
    answer = display_answer(model, query)
    print(answer)