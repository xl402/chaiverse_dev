import os
import re

from datasets import load_dataset
from tqdm import tqdm

from chatbot import PromptFormatter, ConversationTruncator

HF_TOKEN = os.environ.get('HF_TOKEN')


def load_data(dataset_name):
    ds = load_dataset(dataset_name, token=HF_TOKEN)
    return ds['train'].to_pandas()


def get_dpo_data(df):
    df['formatted_conversations'] = format_conversation(df, formatter)
    out = df[['formatted_conversations', 'selected_response', 'rejected_response']]
    rename = {'formatted_conversations': 'prompt',
              'selected_response': 'chosen',
              'rejected_response': 'rejected'}
    out = out.rename(columns=rename)
    return out


def format_conversation(df, formatter, context_window_length=600):
    formatted_conversations = []
    truncator = ConversationTruncator(formatter, context_window_length=context_window_length)
    for index, row in tqdm(df.iterrows()):
        full_formatted_conversation = truncator.truncate(
                row['bot_name'],
                row['memory'],
                row['prompt'],
                row['chat_history']
            )
        formatted_conversations.append(full_formatted_conversation)
    return formatted_conversations


if __name__ == '__main__':
    df = load_data('Jellywibble/avb_cleaned')
    formatter = PromptFormatter()
    dpo_data = get_dpo_data(df)
