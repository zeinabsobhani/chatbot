from kaggle.api.kaggle_api_extended import KaggleApi
import os
import pandas as pd
from bs4 import BeautifulSoup
import re
from pandarallel import pandarallel
import warnings
import tiktoken

class KaggleDataDownloader:
    """
    https://github.com/Kaggle/kaggle-api?tab=readme-ov-file#api-credentials
    """
    def __init__(self, kaggle_path, output_path = "data/raw"):
        self.kaggle_path = kaggle_path
        self.output_path = output_path
        self.api = KaggleApi()
        self.api.authenticate()

    def download_extract(self):
        self.api.dataset_download_files(self.kaggle_path, path=self.output_path, unzip = True)


class DataLoader:
    def __init__(self, path = "data/raw"):
        self.path = path

    def list(self):
        files = os.listdir(self.path)
        print(f"{len(files)} files found in output folder")
        return files
    
    def load_file(self,filename, **kwargs):
        df = pd.read_csv(self.path+"/"+filename, encoding='iso-8859-1', **kwargs)
        return df
    
    def load_all(self, **kwargs):
        files = self.list()
        out = []
        for file in files:
            df = pd.read_csv(self.path+"/"+file, encoding='iso-8859-1', **kwargs)
            out.append(df)
        return out

class TextCleaner:
    def __init__(self):
        self.parallel = os.environ.get('Pandarallel')
        if self.parallel is None:
            self.parallel = True
        elif self.parallel.isin(['False','false','f']):
            self.parallel = False
        elif self.parallel.isin(['True','true','t']):
            self.parallel = True
        else:
            warnings.warn("Unknown Pandarallel variable"); print("Pandarallel environment variable should be true or false")

        
    @staticmethod
    def remove_tags(txt):
        txt = ''.join(BeautifulSoup(txt, 'html.parser').findAll(string=True))
        return txt
    
    @staticmethod
    def remove_newlines(txt):
        txt = re.sub(r'[\n]+','\n', txt)
        return txt
    
    def clean_text(self,txt):
        txt = self.remove_tags(txt)
        txt = self.remove_newlines(txt)
        return txt
    
    def clean_text_column(self, df_col):
        if self.parallel:
            pandarallel.initialize(progress_bar=True)
            df_col = df_col.parallel_apply(self.clean_text)
        else:
            df_col = df_col.apply(self.clean_text)
        return df_col
    
    @staticmethod
    def get_tokenizer(model = 'cl100k_base'):
        return tiktoken.get_encoding(model)

    @staticmethod
    def get_tokens(text, tokenizer):
        
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    def get_tokens_column(self, df_col):
        tokenizer = self.get_tokenizer()

        # if self.parallel:
        #     pandarallel.initialize(progress_bar=True)
        #     df_col = df_col.parallel_apply(self.get_tokens, args=(tokenizer,))
        # else:
        # there is an issue with pandarallel here. Skip this for now. 
        df_col = df_col.apply(self.get_tokens, args = (tokenizer,))
        return df_col
    
    
class Preprocessor(TextCleaner):
    def __init__(self, df_questions, df_answers , max_l = None):
        super().__init__()
        self.df_questions = df_questions
        self.df_answers = df_answers
        self.max_l = max_l

    
    def preprocess(self):
        self.df_questions['clean_body'] = self.clean_text_column(self.df_questions['Body'])

        # attach title and body of the question
        self.df_questions['titlebody'] = self.df_questions['Title'] + "\n" + self.df_questions['clean_body']

        self.df_questions['titlebody_tokens'] = self.get_tokens_column(self.df_questions['titlebody'])
        # filter length
        if self.max_l:
            df = df[df['titlebody_tokens']<=self.max_l]

        # calculate some metrics. These will be the vectors' metadata
        answers_metrics = self.df_answers.groupby('ParentId').agg(
            answers_count = ('Id','count'),
            max_answer_scores=('Score', 'max'),
            mean_answer_scores=('Score', 'mean'),
            sum_answer_scores=('Score', 'sum')
            )
        
        self.df_questions = self.df_questions.merge(answers_metrics, left_on = 'Id', right_index = True)

        return self.df_questions


# dl = DataLoader()
# df_questions = dl.load_file("Questions.csv").iloc[:10]
# df_answers = dl.load_file("Answers.csv")

# p = Preprocessor(df_questions, df_answers ,)
# df_processed = p.preprocess()
# print(df_processed.head())
# print(len(df_processed))

# print("---")
