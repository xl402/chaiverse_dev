import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from chatbot import Chatbot, BotConfig, PromptFormatter


class GPT2Formatter(PromptFormatter):
    memory_template: str = "### You are GPT2, your role is to respond according to {bot_name}'s persona:\n{memory}\n"
    prompt_template: str = "### Model Input:\n{prompt}\n"
    bot_template: str = "{bot_name}: {message}\n"
    user_template: str = "{user_name}: {message}\n"
    response_template: str = "### Model Response:\n{bot_name}:"


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    bot_config = BotConfig.from_json(os.path.join('resources', 'vampire_queen.json'))

    chatbot = Chatbot(model, tokenizer, device='cpu')

    generation_params = {
            'do_sample': True,
            'temperature': 0.9,
            'max_new_tokens': 100,
            }

    formatter = GPT2Formatter()
    chatbot.chat(bot_config, formatter, generation_params=generation_params, debug=True)
