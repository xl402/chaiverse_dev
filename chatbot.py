import json

from pydantic import BaseModel, Field, validator


class Chatbot():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.chat_history = None
        self.max_input_tokens = 512

    def chat(self, bot_config, formatter, generation_params=None, debug=False):
        self.chat_history = [('Bot', bot_config.first_message)]
        self._pprint(bot_config.bot_label, bot_config.first_message, True)
        user_input = input('You: ')
        while user_input != 'exit':
            self.chat_history.append(('User', user_input))
            payload = self._get_payload(bot_config, formatter, debug)
            response = self._generate_response(payload, generation_params)
            self._pprint(bot_config.bot_label, response, True)
            self.chat_history.append(('Bot', response))
            user_input = input('You: ')

    def _truncate_response(self, response):
        return response.split('\n')[0]

    def _generate_response(self, payload, generation_params):
        encoded_input = self.tokenizer(payload, return_tensors="pt", padding=True, truncation=True)
        gen_tokens = self.model.generate(**encoded_input, **generation_params)
        gen_text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)[len(payload):]
        return self._truncate_response(gen_text)

    def _pprint(self, actor_label, message, is_bot):
        color = '\033[96m' if is_bot else '\033[93m'
        print(f'{color}{actor_label}: {message}\033[0m')

    def _get_payload(self, bot_config, formatter, debug):
        chat_history = self._format_chat_history(bot_config, formatter)
        bot_name = bot_config.bot_label
        memory = formatter.memory_template.format(bot_name=bot_name, memory=bot_config.memory)
        prompt = formatter.prompt_template.format(prompt=bot_config.prompt)
        response = formatter.response_template.format(bot_name=bot_name)
        prompt = self._truncate_prompt(prompt, chat_history, self.max_input_tokens - len(memory))
        payload = memory + prompt + chat_history + response
        if debug:
            print('§' * 10)
            print(payload)
            print('§' * 10)
        return payload

    def _truncate_prompt(self, prompt, chat_history, limit):
        if len(prompt) + len(chat_history) > limit:
            truncated_length = max(0, limit - len(chat_history))
            prompt = prompt[:truncated_length]+'\n'
        return prompt

    def _format_chat_history(self, bot_config, formatter):
        out = []
        bot_name = bot_config.bot_label
        for sender, message in self.chat_history:
            if sender == 'Bot':
                out.append(formatter.bot_template.format(bot_name=bot_name, message=message))
            else:
                out.append(formatter.user_template.format(user_name='You', message=message))
        out = self._truncate_conversation(out)
        return ''.join(out)

    def _truncate_conversation(self, conversation_list):
        truncated_conversation = []
        total_character_count = 0
        for convo in conversation_list[::-1]:
            total_character_count += len(convo)
            if total_character_count <= self.max_input_tokens:
                truncated_conversation.append(convo)
        return truncated_conversation[::-1]


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
