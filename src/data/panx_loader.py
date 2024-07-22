from collections import defaultdict
from datasets import DatasetDict
from datasets import load_dataset

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

from tqdm import tqdm

import numpy as np
import pandas as pd

class PANX_dataloader:

    def __init__(self, langs: list, nrows: int):
        self.langs = langs
        self.nrows = nrows
        self.dataset = None

    def get_multilingual_dataset(self, langs: list) -> defaultdict:
        panx_ch = defaultdict(DatasetDict)
        
        pbar = tqdm(total=len(langs), desc="Fetching Language Data")
        for lang in langs:
            ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
            panx_ch[lang] = ds

            pbar.update(1)

        return panx_ch
    
    def extract_tag_names(self, dataset: defaultdict, lang: str) -> defaultdict:
        tags = dataset[lang]["train"].features["ner_tags"].feature

        def create_tag_names(batch):
            return {"ner_tags_str": ' '.join([tags.int2str(idx) for idx in batch["ner_tags"]])}

        def combine_text_tokens(batch):
            return {"tokens_str": ' '.join(batch["tokens"])}

        dataset = dataset[lang].map(create_tag_names)
        dataset = dataset.map(combine_text_tokens)
        return dataset
    
    def load_data(self) -> pd.DataFrame:
        panx_ch = self.get_multilingual_dataset(self.langs)
        lang_dataframes = []

        pbar = tqdm(total=len(self.langs), desc="Processing Language Data")
        for lang in self.langs:
            lang_data = self.extract_tag_names(panx_ch, lang)
            
            train_df = pd.DataFrame(lang_data['train'])
            val_df = pd.DataFrame(lang_data['validation'])
            test_df = pd.DataFrame(lang_data['test'])
            
            complete_lang_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            complete_lang_df['lang'] = lang

            if self.nrows > len(complete_lang_df):
                lang_dataframes.append(complete_lang_df)
            else:
                lang_dataframes.append(complete_lang_df.sample(n=self.nrows, random_state=42))

            pbar.update(1)

        multilingual_df = pd.concat(lang_dataframes)
        print('Concatenated all data together')
        
        return multilingual_df
    
    def load_training_data(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        multilingual_df = self.load_data()
        multilingual_df = multilingual_df.sample(frac=1, random_state=42)

        # Calculate the indices for splitting
        total_len = len(multilingual_df)
        train_end = int(train_ratio * total_len)
        val_end = train_end + int(val_ratio * total_len)

        # Perform the split
        df_train, df_val, df_test = np.split(multilingual_df, [train_end, val_end])
        print("Split data into train, val, and test sets")

        return df_train, df_val, df_test
