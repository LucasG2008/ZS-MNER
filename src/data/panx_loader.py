from collections import defaultdict
from datasets import DatasetDict
from datasets import load_dataset

from tqdm import tqdm

import numpy as np
import pandas as pd

class PANX_dataloader:

    def __init__(self, langs: list, nrows: int):
        self.langs = langs
        self.nrows = nrows
        self.dataset = None

    def get_multilingual_dataset(self, langs: list) -> defaultdict:
        """
        Get multilingual NER Xtreme PAN-X dataset from huggingface

        Input:
        - langs: List of language codes to get data for

        Output: 
        - panx_ch: Complete PAN-X dataset
        """
        panx_ch = defaultdict(DatasetDict)
        
        pbar = tqdm(total=len(langs), desc="Fetching Language Data")
        for lang in langs:
            ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
            panx_ch[lang] = ds

            pbar.update(1)

        return panx_ch
    
    def extract_tag_names(self, dataset: defaultdict, lang: str) -> defaultdict:
        """
        Get string representation of NER tags from Xtreme PAN-X dataset

        Input:
            - dataset: Complete PAN-X dataset
            - lang: Language code

        Output:
            - dataset: PAN-X dataset for specified language with NER tag names
        """
        tags = dataset[lang]["train"].features["ner_tags"].feature

        def create_tag_names(batch):
            return {"ner_tags_str": ' '.join([tags.int2str(idx) for idx in batch["ner_tags"]])}

        def combine_text_tokens(batch):
            return {"tokens_str": ' '.join(batch["tokens"])}

        dataset = dataset[lang].map(create_tag_names)
        dataset = dataset.map(combine_text_tokens)
        return dataset
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from the PANX_dataloader.

        Returns:
            pd.DataFrame: The concatenated dataframes from all languages.
        """

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
            lang_dataframes.append(complete_lang_df.sample(n=self.nrows, random_state=42))

            pbar.update(1)

        multilingual_df = pd.concat(lang_dataframes)
        
        return multilingual_df
    
    def load_training_data(self):
        """
        Load training data for the model by splitting the multilingual dataset into training, validation, and test sets.

        Returns:
            df_train: DataFrame - Training data
            df_val: DataFrame - Validation data
            df_test: DataFrame - Test data
        """

        multilingual_df = self.load_data()
        df_train, df_val, df_test = np.split(multilingual_df.sample(frac=1, random_state=42), 
                                            [int(0.8*len(multilingual_df)), 
                                            int(0.9*len(multilingual_df))])
        return df_train, df_val, df_test
