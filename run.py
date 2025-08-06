from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    model_id = 'openai/gpt-oss-20b'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        max_memory={i: '20GiB' for i in range(4)},
    )
    input_token_ids = tokenizer.apply_chat_template(
        conversation=[{'role': 'user', 'content': 'リーマン予想とは何ですか。'}],
        return_tensors='pt',
        return_dict=True,
    )
    output_token_ids = model.generate(
        input_token_ids['input_ids'].to(model.device),
        attention_mask=input_token_ids['attention_mask'],
        max_new_tokens=128,
    )
    output = tokenizer.decode(output_token_ids[0][input_token_ids['input_ids'].size(1):])
    print('===== ANSWER ====\n', output)


if __name__ == '__main__':
    main()
