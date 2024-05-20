import json
import re

from pydantic import BaseModel, Field, validator


class Chatbot():
    def __init__(self, model, tokenizer, device=0):
        self.model = model
        self.tokenizer = tokenizer
        self.chat_history = None
        self.max_input_tokens = 512
        self.device = device

    def chat(self, bot_config, formatter, generation_params=None, debug=False):
        bot_name = bot_config.bot_label
        self.chat_history = [{'message': bot_config.first_message, 'sender': bot_name}]
        self._pprint(bot_name, bot_config.first_message, True)
        truncator = ConversationTruncator(formatter, self.max_input_tokens)
        user_input = input('You: ')
        while user_input != 'exit':
            self.chat_history.append({'message': user_input, 'sender': 'You'})
            payload = truncator.truncate(bot_name, bot_config.memory, bot_config.prompt, self.chat_history)
            if debug:
                print('§' * 100)
                print(payload)
                print('§' * 100)
            response = self._generate_response(payload, generation_params)
            self._pprint(bot_name, response, True)
            self.chat_history.append({'message': response, 'sender': bot_name})
            user_input = input('You: ')

    def _generate_response(self, payload, generation_params):
        encoded_input = self.tokenizer(payload, return_tensors="pt", padding=True, truncation=True).to(self.device)
        gen_tokens = self.model.generate(**encoded_input, **generation_params)
        gen_text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)[len(payload):]
        return self._truncate_response(gen_text)

    def _pprint(self, actor_label, message, is_bot):
        color = '\033[96m' if is_bot else '\033[93m'
        print(f'{color}{actor_label}: {message}\033[0m')

    def _truncate_response(self, response):
        response = response.split('\n')[0]
        return self._truncate_till_close(response)

    def _truncate_till_close(self, text):
        last_close = max(text.rfind('.'), text.rfind('?'), text.rfind('!'))
        if last_close == -1:
            return text
        return text[:last_close + 1]


class BotConfig:
    def __init__(self, bot_label, first_message, memory, prompt):
        self.bot_label = bot_label
        self.first_message = first_message
        self.memory = memory
        self.prompt = prompt

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return cls(data['bot_label'], data['first_message'], data['memory'], data['prompt'])


class PromptFormatter(BaseModel):
    memory_template: str = Field(
        title="Memory template",
        default = "{bot_name}'s Persona: {memory}\n####\n"
    )
    prompt_template: str = Field(
        title="Prompt template",
        default="{prompt}\n<START>\n"
    )
    bot_template: str = Field(
        title="Bot message template",
        default="{bot_name}: {message}\n"
    )
    user_template: str = Field(
        title="User message template",
        default="{user_name}: {message}\n"
    )
    response_template: str = Field(
        title="Bot response template",
        default="{bot_name}:"
    )

    @validator("memory_template")
    def validate_memory(cls, memory_template):
        if "{memory}" not in memory_template:
            raise ValueError("Formatter's memory_template must contain '{memory}'!")
        return memory_template

    @validator("prompt_template")
    def validate_formatter(cls, prompt_template):
        if "{prompt}" not in prompt_template:
            raise ValueError("Formatter's prompt_template must contain '{prompt}'!")
        return prompt_template

    @validator("bot_template")
    def validate_bot(cls, bot_template):
        if "{message}" not in bot_template:
            raise ValueError("Formatter's bot_template must contain '{message}'!")
        return bot_template

    @validator("user_template")
    def validate_user(cls, user_template):
        if "{message}" not in user_template:
            raise ValueError("Formatter's user_template must contain '{message}'!")
        return user_template


class ConversationTruncator():
    def __init__(self, formatter, context_window_length=600):
        self.formatter = formatter
        self.context_window_length = context_window_length

    def truncate(self, bot_name, memory, prompt, chat_history):
        memory_formatted = self.formatter.memory_template.format(
            bot_name=bot_name,
            memory=memory
        )

        prompt_formatted = self.formatter.prompt_template.format(
            prompt=prompt
        )

        conversation_history = ""
        for message in chat_history:
            if message['sender'] == bot_name:
                conversation_history += self.formatter.bot_template.format(
                    bot_name=bot_name,
                    message=message['message']
                )
            else:
                conversation_history += self.formatter.user_template.format(
                    user_name=message['sender'],
                    message=message['message']
                )

        response_prompt = self.formatter.response_template.format(
            bot_name=bot_name
        )

        max_n_words = self.context_window_length - len(response_prompt.split())
        truncated_conversation = self._truncate_conversation(
            memory_formatted,
            prompt_formatted,
            conversation_history,
            max_n_words=max_n_words
        )
        full_formatted_conversation = truncated_conversation + response_prompt
        return full_formatted_conversation

    def _truncate_conversation(self, memory_formatted, prompt_formatted, conversation_history, max_n_words):
        max_memory_words = max_n_words // 2
        words_memory = re.split(r'(?<=\s)', memory_formatted)
        if len(words_memory) > max_memory_words:
            words_memory = words_memory[:max_memory_words]
            memory_formatted = ''.join(words_memory).rstrip()

        remaining_length = max_n_words - len(words_memory)
        words_prompt = re.split(r'(?<=\s)', prompt_formatted)
        words_history = re.split(r'(?<=\s)', conversation_history)
        while len(words_prompt) + len(words_history) > remaining_length and words_prompt:
            words_prompt.pop()
        prompt_formatted = ''.join(words_prompt)

        while len(words_history) > remaining_length and words_history:
            words_history.pop(0)

        conversation_history = ''.join(words_history)
        return memory_formatted + prompt_formatted + conversation_history
