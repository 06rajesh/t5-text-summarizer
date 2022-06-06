import json
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import termcolor
import textwrap
from transformers import T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer, PreTrainedModel, T5Config, T5_PRETRAINED_CONFIG_ARCHIVE_MAP
import tqdm
from typing import Union

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sns.set(style="whitegrid", palette="muted", font_scale=1.2)
rcParams['figure.figsize'] = 16, 10

df = pd.read_csv("data/news_summary.csv", encoding='latin-1')
df = df[["text", "ctext"]]
df.columns = ["summary", "text"]
df = df.dropna()
print(df.head())

train_df, test_df = train_test_split(df, test_size=0.1)
print(train_df.shape, test_df.shape)

### Dataset
class NewsSummaryDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: T5Tokenizer,
            text_max_token_len: int = 512,
            summary_max_token_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row["text"]

        text_encoding = self.tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        summary_encoding = self.tokenizer(
            data_row["summary"],
            max_length=self.summary_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        labels = summary_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            text=text,
            summary=data_row["summary"],
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding["attention_mask"].flatten()
        )

### Data Module
class NewsSummaryDataModule():
    def __init__(
            self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            tokenizer: T5Tokenizer,
            batch_size: int = 4,
            text_max_token_len: int = 512,
            summary_max_token_len: int = 128
    ):
        super(NewsSummaryDataModule, self).__init__()

        self.train_df = train_df
        self.test_df = test_df

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

        self.setup()

    def setup(self):
        self.train_dataset = NewsSummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

        self.test_dataset = NewsSummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )



MODEL_NAME = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
MODEL_PATH = "checkpoints/best-checkpoints-old"

text_token_counts, summary_token_counts = [], []

for idx, row in train_df.iterrows():
    text_token_count = len(tokenizer.encode(row["text"]))
    text_token_counts.append(text_token_count)

    summary_token_count = len(tokenizer.encode(row["summary"]))
    summary_token_counts.append(summary_token_count)

# fig, (ax1, ax2) = plt.subplots(1, 2)
#
# sns.histplot(text_token_counts, ax=ax1)
# ax1.set_title("Full text token counts")
#
# sns.histplot(summary_token_counts, ax=ax2)
# ax2.set_title("Summary text token counts")
#
# plt.show()

N_EPOCHS = 2
BATCH_SIZE = 4

data_module = NewsSummaryDataModule(train_df, test_df, tokenizer, batch_size=BATCH_SIZE)

### Model
class NewsSummaryModel(PreTrainedModel):

    def __init__(self, load_model_from=None):
        conf = T5Config()
        super(NewsSummaryModel, self).__init__(conf)
        pretrained = MODEL_NAME
        if load_model_from:
            pretrained = load_model_from
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained, return_dict=True)

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        # self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        # self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    # def test_step(self, batch, batch_idx):
    #     input_ids = batch["text_input_ids"]
    #     attention_mask = batch["text_attention_mask"]
    #     labels = batch["labels"]
    #     labels_attention_mask = batch["labels_attention_mask"]
    #
    #     loss, outputs = self(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         decoder_attention_mask=labels_attention_mask,
    #         labels=labels
    #     )
    #
    #     self.log("test_loss", loss, prog_bar=True, logger=True)
    #     return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)

# checkpoint_callback = ModelCheckpoint(
#     dirpath="checkpoints",
#     filename="best-checkpoints-old",
#     save_top_k=1,
#     verbose=True,
#     monitor="val_loss",
#     mode="min"
# )

# logger = TensorBoardLogger("lightning_logs", name="news-summary")
# trainer = pl.Trainer(
#     logger=logger,
#     max_epochs=N_EPOCHS,
#     gpus=None,
#     progress_bar_refresh_rate=10,
#     accelerator='cpu',
# )

def calculate_validation_loss(model: NewsSummaryModel, val_loader: DataLoader) -> float:
    running_loss = running_items = 0.0
    print("Calculating Validation loss")
    model.eval()
    for idx, batch in enumerate(val_loader):
        loss = model.validation_step(batch, idx)

        running_loss += loss.item()
        running_items += val_loader.batch_size

    return running_loss / running_items


def fit(model: NewsSummaryModel, datamodule: NewsSummaryDataModule, n_epochs:int = 2, refresh_rate:int = 5, model_save_path: Union[str, None] = None):
    optimizer = model.configure_optimizers()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    last_saved_loss = 9999

    running_loss = running_items = 0.0
    epochs_trained = 0
    epoch_iterator = tqdm.trange(epochs_trained, n_epochs, desc="Summary Epoch", position=0, leave=True)
    for epoch, _ in enumerate(epoch_iterator):
        train_iterator = tqdm.tqdm(train_loader, desc="Iteration", position=0, leave=True)
        for idx, batch in enumerate(train_iterator, 1):
            loss = model.training_step(batch, idx)

            # backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_items += train_loader.batch_size

            if idx%refresh_rate == 0:
                train_iterator.set_description(
                    "train_loss = %.4f" % (running_loss / running_items)
                )

        val_loss = calculate_validation_loss(model, val_loader)
        epoch_iterator.set_description(
            "Summary Epoch, val_loss = %.4f" % (val_loss)
        )

        if val_loss < last_saved_loss and model_save_path != None:
            model.save_pretrained(model_save_path)
            print("Model save to path for the {} validation loss : {}".format(val_loss, model_save_path))
            last_saved_loss = val_loss

model = NewsSummaryModel(load_model_from=MODEL_PATH)
fit(model, data_module, n_epochs=N_EPOCHS, refresh_rate=20, model_save_path=MODEL_PATH)





