import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def normalize_source(text: str):
    text = textwrap.dedent(text).strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.splitlines()]


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": normalize_source(text),
    }


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": normalize_source(text),
    }


def notebook(title: str, cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_INSTALL_MARKDOWN = """
# Environment Setup

This notebook installs the shared environment for the whole teaching sequence.
The repository bootstrap installs from the shared `requirements.txt`, which is configured
to pull the CUDA-enabled PyTorch wheel used for this Windows teaching environment.

In the code cell below we:

- locate the project root
- run the repository bootstrap script
- print progress so students can see the environment being prepared
"""


COMMON_INSTALL_CODE = """
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
bootstrap_script = PROJECT_ROOT / "scripts" / "bootstrap_env.py"

print(f"Project root: {PROJECT_ROOT}")
print(f"Running bootstrap script: {bootstrap_script}")
subprocess.check_call([sys.executable, str(bootstrap_script)])
print("Environment ready. If PyTorch was replaced during bootstrap, restart the kernel once before continuing.")
"""


def notebook_one():
    cells = [
        md(
            """
            # Mini Transformer From Scratch

            ## What We Are Going To Build

            This notebook is the "inside the machine" view of an LLM.
            We will start with raw text, turn it into tokens, build a miniature decoder-only
            transformer in PyTorch, train it on Shakespeare, inspect its attention patterns,
            and generate new text from the trained model.

            ## Why This Notebook Exists

            Large language models can feel mysterious because many teaching examples jump
            straight from tokenized data to a giant pretrained checkpoint. Here we slow down
            just enough to make the pipeline visible:

            - raw text -> tokens
            - tokens -> embeddings
            - embeddings -> self-attention
            - self-attention -> transformer blocks
            - transformer blocks -> next-token logits
            - logits -> softmax probabilities -> sampled text

            ## Learning Outcomes

            By the end of the notebook, students should be able to explain:

            - why tokenization is needed
            - how embeddings and positional information work together
            - why causal masking matters for GPT-style models
            - what the attention equation is doing
            - how the model is trained with cross-entropy loss
            - what "memory" means in a fixed-context decoder
            - how temperature and top-k sampling change generation

            ## Teaching Notes

            - This notebook is designed to be visual and intuitive first, mathematical second.
            - The attention and softmax math is explicitly shown, but we do not stay in derivations for long.
            - There is checkpoint-aware logic later so class can load a saved model if full training is too slow.
            """
        ),
        md(COMMON_INSTALL_MARKDOWN),
        code(COMMON_INSTALL_CODE),
        md(
            """
            ## Imports, Plotting, and Reproducibility

            Before we touch the model, we set up the tools we will reuse throughout the notebook.
            The theory idea is simple: experiments are easier to teach when they are reproducible
            and when plots use a consistent visual style.

            In the code cell below we:

            - import PyTorch, plotting, and dataset utilities
            - set random seeds
            - detect whether a GPU is available
            - create the artifact folders used for checkpoints
            """
        ),
        code(
            """
            import math
            import random
            from collections import Counter
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import torch
            import torch.nn as nn
            from datasets import load_dataset
            from IPython.display import clear_output, display
            from tqdm.auto import tqdm

            sns.set_theme(style="whitegrid", context="talk")
            plt.rcParams["figure.figsize"] = (10, 6)
            plt.rcParams["axes.spines.top"] = False
            plt.rcParams["axes.spines.right"] = False

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)

            PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
            ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
            CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {DEVICE}")
            if DEVICE == "cuda":
                print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("GPU not detected. The notebook still runs, but training will be slower.")
            """
        ),
        md(
            """
            ## Notebook Configuration

            This notebook supports two modes:

            - a faster classroom walkthrough
            - a longer side-project training run

            The theory idea is that teaching notebooks should not be hostage to wall-clock time.
            We want one set of cells that can either train a small model live or load a saved checkpoint.

            In the code cell below we:

            - choose the active preset
            - define the model size and training schedule
            - point to the checkpoint location used later
            - display the active choices in a table
            """
        ),
        code(
            """
            RUN_FULL_SIDE_PROJECT = False
            USE_EXISTING_CHECKPOINT_IF_AVAILABLE = True

            PRESETS = {
                "walkthrough": {
                    "batch_size": 32,
                    "block_size": 128,
                    "n_embed": 192,
                    "n_head": 6,
                    "n_layer": 4,
                    "dropout": 0.2,
                    "learning_rate": 3e-4,
                    "max_iters": 500,
                    "eval_interval": 50,
                    "eval_batches": 25,
                },
                "side_project": {
                    "batch_size": 48,
                    "block_size": 160,
                    "n_embed": 256,
                    "n_head": 8,
                    "n_layer": 6,
                    "dropout": 0.2,
                    "learning_rate": 2e-4,
                    "max_iters": 1800,
                    "eval_interval": 100,
                    "eval_batches": 40,
                },
            }

            ACTIVE_PRESET = "side_project" if RUN_FULL_SIDE_PROJECT else "walkthrough"
            CONFIG = PRESETS[ACTIVE_PRESET].copy()
            CHECKPOINT_PATH = CHECKPOINT_DIR / f"mini_transformer_{ACTIVE_PRESET}.pt"

            config_df = pd.DataFrame(
                [{"setting": key, "value": value} for key, value in CONFIG.items()]
            )
            display(config_df)
            print(f"Checkpoint path: {CHECKPOINT_PATH}")
            """
        ),
        md(
            """
            ## Step 1: Load the Raw Dataset

            Language models start with text, not tensors.
            Before we tokenize anything, it is useful to inspect the raw corpus so students can see
            what kind of language distribution the model will learn from.

            In the code cell below we:

            - load `karpathy/tiny_shakespeare` from Hugging Face
            - collapse the dataset into one training string
            - print a preview of the corpus
            - visualize the most frequent characters and the line-length distribution
            """
        ),
        code(
            """
            dataset = load_dataset("karpathy/tiny_shakespeare")
            split_name = "train" if "train" in dataset else list(dataset.keys())[0]
            text_column = "text" if "text" in dataset[split_name].column_names else dataset[split_name].column_names[0]

            text_rows = dataset[split_name][text_column]
            if isinstance(text_rows, str):
                raw_text = text_rows
            elif len(text_rows) == 1:
                raw_text = text_rows[0]
            else:
                raw_text = "\\n".join(text_rows)

            print(f"Loaded split: {split_name}")
            print(f"Characters in corpus: {len(raw_text):,}")
            print()
            print(raw_text[:900])

            char_counts = Counter(raw_text)
            frequency_df = (
                pd.DataFrame(char_counts.items(), columns=["character", "count"])
                .sort_values("count", ascending=False)
                .head(20)
            )
            line_lengths = pd.Series([len(line) for line in raw_text.splitlines() if line.strip()])

            fig, axes = plt.subplots(1, 2, figsize=(18, 5))
            sns.barplot(data=frequency_df, x="count", y="character", palette="viridis", ax=axes[0])
            axes[0].set_title("Most frequent characters")
            axes[0].set_xlabel("Count")
            axes[0].set_ylabel("Character")

            sns.histplot(line_lengths, bins=30, color="teal", ax=axes[1])
            axes[1].set_title("Distribution of non-empty line lengths")
            axes[1].set_xlabel("Characters per line")
            axes[1].set_ylabel("Number of lines")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 2: Build a Tokenizer From Scratch

            A model cannot consume raw Python strings directly.
            We need a mapping from symbols to integers.

            For a first-principles teaching notebook, a character-level tokenizer is ideal because:

            - every token is easy to inspect
            - the vocabulary is tiny
            - students can directly see how text becomes integers

            In the code cell below we:

            - create a character vocabulary
            - build `stoi` and `itos` lookup tables
            - define encode and decode functions
            - visualize a short text snippet and its token ids
            """
        ),
        code(
            """
            vocabulary = sorted(set(raw_text))
            vocab_size = len(vocabulary)
            stoi = {ch: idx for idx, ch in enumerate(vocabulary)}
            itos = {idx: ch for ch, idx in stoi.items()}

            def encode(text):
                return [stoi[ch] for ch in text]

            def decode(token_ids):
                return "".join(itos[idx] for idx in token_ids)

            encoded_text = torch.tensor(encode(raw_text), dtype=torch.long)
            sample_text = "ROMEO:"
            sample_tokens = encode(sample_text)

            print(f"Vocabulary size: {vocab_size}")
            display(
                pd.DataFrame(
                    {
                        "character": list(sample_text),
                        "token_id": sample_tokens,
                    }
                )
            )

            fig, axes = plt.subplots(1, 2, figsize=(16, 4))
            token_stream = encoded_text[:80].cpu().numpy()
            axes[0].plot(token_stream, marker="o", linewidth=2, color="slateblue")
            axes[0].set_title("First 80 token ids in the corpus")
            axes[0].set_xlabel("Token position")
            axes[0].set_ylabel("Token id")

            sns.heatmap(
                np.array(sample_tokens).reshape(1, -1),
                annot=np.array(list(sample_text)).reshape(1, -1),
                fmt="",
                cmap="magma",
                cbar=False,
                ax=axes[1],
            )
            axes[1].set_title("Character -> token id mapping for a sample word")
            axes[1].set_xlabel("Character position")
            axes[1].set_yticks([])
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 3: Create Training and Validation Splits

            Training a next-token predictor means teaching the model to map a context
            to the token that comes one step later.

            So instead of using labels from a separate file, we create labels by shifting the text.
            This is the core language-modeling trick: the next token is the target.

            In the code cell below we:

            - split the integer token stream into train and validation regions
            - build a small context/target example
            - visualize how each input position predicts the next token
            """
        ),
        code(
            """
            split_index = int(0.9 * len(encoded_text))
            train_data = encoded_text[:split_index]
            val_data = encoded_text[split_index:]

            demo_block_size = 24
            demo_context = train_data[:demo_block_size]
            demo_target = train_data[1 : demo_block_size + 1]

            shift_df = pd.DataFrame(
                {
                    "position": list(range(demo_block_size)),
                    "input_char": [itos[int(token)] if itos[int(token)] != "\\n" else "\\\\n" for token in demo_context],
                    "input_id": demo_context.tolist(),
                    "target_char": [itos[int(token)] if itos[int(token)] != "\\n" else "\\\\n" for token in demo_target],
                    "target_id": demo_target.tolist(),
                }
            )

            print(f"Training tokens: {len(train_data):,}")
            print(f"Validation tokens: {len(val_data):,}")
            display(shift_df)

            fig, ax = plt.subplots(figsize=(14, 3))
            sns.heatmap(
                np.vstack([demo_context.cpu().numpy(), demo_target.cpu().numpy()]),
                cmap="crest",
                cbar=True,
                yticklabels=["input ids", "shifted targets"],
                xticklabels=list(range(demo_block_size)),
                ax=ax,
            )
            ax.set_title("Language modeling target = input shifted by one step")
            ax.set_xlabel("Token position inside one training chunk")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 4: Sample Random Batches

            Transformers do not train on one giant uninterrupted stream at a time.
            We randomly crop many short context windows so the model sees a diverse mix of positions.

            In the code cell below we:

            - define a `get_batch` function
            - sample a random batch from the training split
            - visualize the token-id grid the model will see during one optimization step
            """
        ),
        code(
            """
            def get_batch(split):
                source = train_data if split == "train" else val_data
                starts = torch.randint(len(source) - CONFIG["block_size"] - 1, (CONFIG["batch_size"],))
                x = torch.stack([source[start : start + CONFIG["block_size"]] for start in starts])
                y = torch.stack([source[start + 1 : start + CONFIG["block_size"] + 1] for start in starts])
                return x.to(DEVICE), y.to(DEVICE)

            xb, yb = get_batch("train")
            print(f"Batch shape for inputs: {tuple(xb.shape)}")
            print(f"Batch shape for targets: {tuple(yb.shape)}")

            fig, ax = plt.subplots(figsize=(14, 6))
            sns.heatmap(xb[:10].cpu().numpy(), cmap="rocket", ax=ax)
            ax.set_title("Token ids in a random training batch")
            ax.set_xlabel("Position inside the context window")
            ax.set_ylabel("Batch row")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 5: Turn Token Ids Into Embeddings

            Token ids are categorical labels, not continuous representations.
            The embedding layer learns a dense vector for each token, and positional embeddings tell the
            model where each token lives inside the sequence.

            In the code cell below we:

            - define a tiny embedding module
            - look at token and position embeddings separately
            - visualize the first few embedding dimensions as heatmaps
            """
        ),
        code(
            """
            class TokenPositionEmbedding(nn.Module):
                def __init__(self, vocab_size, block_size, n_embed):
                    super().__init__()
                    self.token_embedding = nn.Embedding(vocab_size, n_embed)
                    self.position_embedding = nn.Embedding(block_size, n_embed)

                def forward(self, idx):
                    batch_size, time_steps = idx.shape
                    token_vectors = self.token_embedding(idx)
                    positions = torch.arange(time_steps, device=idx.device)
                    position_vectors = self.position_embedding(positions)
                    return token_vectors, position_vectors

            embedding_preview = TokenPositionEmbedding(vocab_size, CONFIG["block_size"], CONFIG["n_embed"]).to(DEVICE)
            token_vectors, position_vectors = embedding_preview(xb[:1])
            combined_vectors = token_vectors + position_vectors.unsqueeze(0)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            sns.heatmap(token_vectors[0, :12, :24].detach().cpu(), cmap="coolwarm", ax=axes[0])
            axes[0].set_title("Token embeddings")
            axes[0].set_xlabel("Embedding dimension")
            axes[0].set_ylabel("Token position")

            sns.heatmap(position_vectors[:12, :24].detach().cpu(), cmap="crest", ax=axes[1])
            axes[1].set_title("Position embeddings")
            axes[1].set_xlabel("Embedding dimension")
            axes[1].set_ylabel("Sequence position")

            sns.heatmap(combined_vectors[0, :12, :24].detach().cpu(), cmap="magma", ax=axes[2])
            axes[2].set_title("Combined signal sent into attention")
            axes[2].set_xlabel("Embedding dimension")
            axes[2].set_ylabel("Token position")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 6: Inspect the Attention Equation

            The core attention equation is:

            `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) V`

            Intuitively:

            - queries ask: "what information am I looking for?"
            - keys say: "what kind of information do I contain?"
            - values carry the actual content to mix together
            - softmax turns raw compatibility scores into probabilities
            - the causal mask prevents looking into the future

            In the code cell below we:

            - create one attention head by hand
            - compute raw scores, masked scores, and attention probabilities
            - visualize the causal mask and the final weighted lookup
            """
        ),
        code(
            """
            attention_demo_length = 12
            demo_x = combined_vectors[:, :attention_demo_length, :]

            query_layer = nn.Linear(CONFIG["n_embed"], CONFIG["n_embed"], bias=False).to(DEVICE)
            key_layer = nn.Linear(CONFIG["n_embed"], CONFIG["n_embed"], bias=False).to(DEVICE)
            value_layer = nn.Linear(CONFIG["n_embed"], CONFIG["n_embed"], bias=False).to(DEVICE)

            queries = query_layer(demo_x)
            keys = key_layer(demo_x)
            values = value_layer(demo_x)

            raw_scores = queries @ keys.transpose(-2, -1)
            scaled_scores = raw_scores / math.sqrt(CONFIG["n_embed"])
            causal_mask = torch.tril(torch.ones(attention_demo_length, attention_demo_length, device=DEVICE))
            masked_scores = scaled_scores.masked_fill(causal_mask == 0, float("-inf"))
            attention_weights = torch.softmax(masked_scores, dim=-1)
            context_vectors = attention_weights @ values

            plot_tokens = [
                itos[int(token)] if itos[int(token)] != "\\n" else "\\\\n"
                for token in xb[0, :attention_demo_length].cpu()
            ]

            fig, axes = plt.subplots(1, 4, figsize=(24, 5))
            sns.heatmap(raw_scores[0].detach().cpu(), cmap="icefire", ax=axes[0])
            axes[0].set_title("Raw QK^T scores")
            axes[0].set_xlabel("Key positions")
            axes[0].set_ylabel("Query positions")

            sns.heatmap(causal_mask.detach().cpu(), cmap="Greys", cbar=False, ax=axes[1])
            axes[1].set_title("Causal mask")
            axes[1].set_xlabel("Visible key positions")
            axes[1].set_ylabel("Query positions")

            sns.heatmap(
                attention_weights[0].detach().cpu(),
                cmap="viridis",
                xticklabels=plot_tokens,
                yticklabels=plot_tokens,
                ax=axes[2],
            )
            axes[2].set_title("Softmax attention weights")
            axes[2].set_xlabel("Keys")
            axes[2].set_ylabel("Queries")

            sns.heatmap(context_vectors[0, :, :24].detach().cpu(), cmap="mako", ax=axes[3])
            axes[3].set_title("Weighted value vectors")
            axes[3].set_xlabel("Embedding dimension")
            axes[3].set_ylabel("Token position")
            plt.tight_layout()
            plt.show()

            row_sums = attention_weights[0].sum(dim=-1).detach().cpu().numpy()
            print("Each attention row sums to:", np.round(row_sums, 3))
            """
        ),
        md(
            """
            ## Step 7: Build the Mini GPT Model

            Now we assemble the actual network.
            A decoder-only transformer block repeats three ideas:

            - layer normalization to stabilize training
            - masked multi-head self-attention to mix information across positions
            - a feed-forward network to transform each position independently

            In the code cell below we:

            - define multi-head self-attention
            - define a transformer block
            - define the full decoder-only language model
            - keep the code commented so students can map every line to the theory
            """
        ),
        code(
            """
            class MultiHeadSelfAttention(nn.Module):
                def __init__(self, n_embed, n_head, block_size, dropout):
                    super().__init__()
                    assert n_embed % n_head == 0, "Embedding dimension must divide evenly across heads."
                    self.n_head = n_head
                    self.head_dim = n_embed // n_head
                    self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
                    self.proj = nn.Linear(n_embed, n_embed)
                    self.dropout = nn.Dropout(dropout)
                    self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

                def forward(self, x, return_attention=False):
                    batch_size, time_steps, channels = x.shape
                    qkv = self.qkv(x)
                    qkv = qkv.view(batch_size, time_steps, 3, self.n_head, self.head_dim)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    queries, keys, values = qkv[0], qkv[1], qkv[2]

                    scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    scores = scores.masked_fill(self.mask[:time_steps, :time_steps] == 0, float("-inf"))
                    attention = torch.softmax(scores, dim=-1)
                    attention = self.dropout(attention)

                    mixed = attention @ values
                    mixed = mixed.transpose(1, 2).contiguous().view(batch_size, time_steps, channels)
                    output = self.proj(mixed)
                    output = self.dropout(output)

                    if return_attention:
                        return output, attention
                    return output


            class FeedForward(nn.Module):
                def __init__(self, n_embed, dropout):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(n_embed, 4 * n_embed),
                        nn.GELU(),
                        nn.Linear(4 * n_embed, n_embed),
                        nn.Dropout(dropout),
                    )

                def forward(self, x):
                    return self.net(x)


            class Block(nn.Module):
                def __init__(self, n_embed, n_head, block_size, dropout):
                    super().__init__()
                    self.ln1 = nn.LayerNorm(n_embed)
                    self.ln2 = nn.LayerNorm(n_embed)
                    self.attn = MultiHeadSelfAttention(n_embed, n_head, block_size, dropout)
                    self.ff = FeedForward(n_embed, dropout)

                def forward(self, x, return_attention=False):
                    if return_attention:
                        attn_output, attention = self.attn(self.ln1(x), return_attention=True)
                        x = x + attn_output
                        x = x + self.ff(self.ln2(x))
                        return x, attention

                    x = x + self.attn(self.ln1(x))
                    x = x + self.ff(self.ln2(x))
                    return x


            class MiniGPT(nn.Module):
                def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer, dropout):
                    super().__init__()
                    self.block_size = block_size
                    self.token_embedding = nn.Embedding(vocab_size, n_embed)
                    self.position_embedding = nn.Embedding(block_size, n_embed)
                    self.blocks = nn.ModuleList(
                        [Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)]
                    )
                    self.final_norm = nn.LayerNorm(n_embed)
                    self.lm_head = nn.Linear(n_embed, vocab_size)

                def forward(self, idx, targets=None, return_attentions=False):
                    batch_size, time_steps = idx.shape
                    positions = torch.arange(time_steps, device=idx.device)
                    x = self.token_embedding(idx) + self.position_embedding(positions)

                    first_block_attention = None
                    for block_index, block in enumerate(self.blocks):
                        if return_attentions and block_index == 0:
                            x, first_block_attention = block(x, return_attention=True)
                        else:
                            x = block(x)

                    x = self.final_norm(x)
                    logits = self.lm_head(x)

                    loss = None
                    if targets is not None:
                        batch_size, time_steps, vocab_size = logits.shape
                        loss = nn.functional.cross_entropy(
                            logits.view(batch_size * time_steps, vocab_size),
                            targets.view(batch_size * time_steps),
                        )

                    if return_attentions:
                        return logits, loss, first_block_attention
                    return logits, loss
            """
        ),
        md(
            """
            ## Step 8: Instantiate the Model and Inspect Its Size

            It is useful to check model size before training so students can connect architecture choices
            to parameter count, memory cost, and runtime.

            In the code cell below we:

            - instantiate the mini GPT model
            - run one forward pass on a random batch
            - count the parameters by top-level component
            - visualize where the learnable capacity lives
            """
        ),
        code(
            """
            model = MiniGPT(
                vocab_size=vocab_size,
                block_size=CONFIG["block_size"],
                n_embed=CONFIG["n_embed"],
                n_head=CONFIG["n_head"],
                n_layer=CONFIG["n_layer"],
                dropout=CONFIG["dropout"],
            ).to(DEVICE)

            test_logits, test_loss = model(xb, yb)
            print(f"Logits shape: {tuple(test_logits.shape)}")
            print(f"Initial random-weight loss: {test_loss.item():.4f}")

            component_totals = {}
            for name, parameter in model.named_parameters():
                component_name = name.split(".")[0]
                component_totals.setdefault(component_name, 0)
                component_totals[component_name] += parameter.numel()

            parameter_df = pd.DataFrame(
                [{"component": key, "parameters": value} for key, value in component_totals.items()]
            ).sort_values("parameters", ascending=False)

            total_parameters = int(parameter_df["parameters"].sum())
            print(f"Total parameters: {total_parameters:,}")
            display(parameter_df)

            plt.figure(figsize=(10, 5))
            sns.barplot(data=parameter_df, x="parameters", y="component", palette="flare")
            plt.title("Where the model parameters live")
            plt.xlabel("Number of parameters")
            plt.ylabel("Component")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 9: Train or Load a Checkpoint

            The model learns by comparing its predicted next-token distribution to the true next token,
            then using gradient descent to reduce the cross-entropy loss.

            In the code cell below we:

            - create an optimizer
            - define a small validation routine
            - either load an existing checkpoint or run training
            - visualize the training and validation loss over time
            """
        ),
        code(
            """
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

            @torch.no_grad()
            def estimate_loss():
                model.eval()
                losses = {}
                for split in ["train", "val"]:
                    split_losses = []
                    for _ in range(CONFIG["eval_batches"]):
                        batch_x, batch_y = get_batch(split)
                        _, loss = model(batch_x, batch_y)
                        split_losses.append(loss.item())
                    losses[split] = float(np.mean(split_losses))
                model.train()
                return losses

            history = []

            if CHECKPOINT_PATH.exists() and USE_EXISTING_CHECKPOINT_IF_AVAILABLE:
                checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                history = checkpoint.get("history", [])
                print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
            else:
                print("Starting training from scratch...")
                model.train()
                for step in tqdm(range(CONFIG["max_iters"]), desc="Training mini GPT"):
                    batch_x, batch_y = get_batch("train")
                    _, loss = model(batch_x, batch_y)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    should_log = step % CONFIG["eval_interval"] == 0 or step == CONFIG["max_iters"] - 1
                    if should_log:
                        loss_snapshot = estimate_loss()
                        history.append(
                            {
                                "step": step,
                                "train_loss": loss_snapshot["train"],
                                "val_loss": loss_snapshot["val"],
                            }
                        )

                        clear_output(wait=True)
                        history_df = pd.DataFrame(history)
                        print(
                            f"Step {step:>4} | train loss {loss_snapshot['train']:.4f} | "
                            f"val loss {loss_snapshot['val']:.4f}"
                        )
                        plt.figure(figsize=(10, 5))
                        plt.plot(history_df["step"], history_df["train_loss"], label="train")
                        plt.plot(history_df["step"], history_df["val_loss"], label="validation")
                        plt.title("Learning curve for the from-scratch transformer")
                        plt.xlabel("Training step")
                        plt.ylabel("Cross-entropy loss")
                        plt.legend()
                        plt.tight_layout()
                        plt.show()

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": CONFIG,
                        "history": history,
                    },
                    CHECKPOINT_PATH,
                )
                print(f"Saved checkpoint to {CHECKPOINT_PATH}")

            if history:
                history_df = pd.DataFrame(history)
                plt.figure(figsize=(10, 5))
                plt.plot(history_df["step"], history_df["train_loss"], label="train")
                plt.plot(history_df["step"], history_df["val_loss"], label="validation")
                plt.title("Final loss curves")
                plt.xlabel("Training step")
                plt.ylabel("Cross-entropy loss")
                plt.legend()
                plt.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            ## Step 10: Visualize Attention and LLM Memory

            When people say an LLM "remembers" context, they do not mean memory in the human sense.
            In a decoder-only transformer, memory means:

            - the model only sees tokens inside the current context window
            - each token attends back to earlier visible tokens
            - the attention weights decide which earlier positions matter most

            In the code cell below we:

            - run the model on one context window
            - visualize one block's average attention map
            - inspect what the last token is attending to
            """
        ),
        code(
            """
            model.eval()

            attention_slice = 24
            context_window = train_data[: CONFIG["block_size"]].unsqueeze(0).to(DEVICE)
            _, _, attention_map = model(context_window, return_attentions=True)

            average_attention = attention_map[0].mean(dim=0).detach().cpu().numpy()
            display_tokens = [
                itos[int(token)] if itos[int(token)] != "\\n" else "\\\\n"
                for token in context_window[0, :attention_slice].cpu()
            ]

            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            sns.heatmap(
                average_attention[:attention_slice, :attention_slice],
                cmap="viridis",
                xticklabels=display_tokens,
                yticklabels=display_tokens,
                ax=axes[0],
            )
            axes[0].set_title("Average attention across heads in the first transformer block")
            axes[0].set_xlabel("Keys: earlier tokens the model can look at")
            axes[0].set_ylabel("Queries: current token positions")

            last_token_attention = average_attention[attention_slice - 1, :attention_slice]
            axes[1].bar(range(attention_slice), last_token_attention, color="darkorange")
            axes[1].set_title("Where the last visible token puts its attention mass")
            axes[1].set_xlabel("Earlier position")
            axes[1].set_ylabel("Attention weight")
            axes[1].set_xticks(range(attention_slice))
            axes[1].set_xticklabels(display_tokens, rotation=90)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 11: Generate Text With Different Decoding Strategies

            The model outputs logits, not words.
            To turn logits into generated text we:

            - convert logits to probabilities with softmax
            - optionally scale them with temperature
            - optionally keep only the top-k candidates
            - sample or take the argmax

            In the code cell below we:

            - define a small text-generation helper
            - compare multiple decoding settings from the same prompt
            - display the resulting text so students can see how randomness changes behavior
            """
        ),
        code(
            """
            @torch.no_grad()
            def generate_text(prompt, max_new_tokens=140, temperature=1.0, top_k=None):
                model.eval()
                running_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

                for _ in range(max_new_tokens):
                    idx_cond = running_ids[:, -CONFIG["block_size"] :]
                    logits, _ = model(idx_cond)
                    next_token_logits = logits[:, -1, :]

                    if temperature == 0:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    else:
                        next_token_logits = next_token_logits / temperature
                        if top_k is not None:
                            top_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                            cutoff = top_values[:, [-1]]
                            next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff, float("-inf"))
                        probabilities = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probabilities, num_samples=1)

                    running_ids = torch.cat([running_ids, next_token], dim=1)

                return decode(running_ids[0].tolist())

            generation_prompt = "ROMEO:\\n"
            generations = [
                {
                    "strategy": "Greedy-like (temperature 0)",
                    "text": generate_text(generation_prompt, temperature=0, top_k=None),
                },
                {
                    "strategy": "Warm sampling (temperature 0.8, top_k 20)",
                    "text": generate_text(generation_prompt, temperature=0.8, top_k=20),
                },
                {
                    "strategy": "Looser sampling (temperature 1.1, top_k 40)",
                    "text": generate_text(generation_prompt, temperature=1.1, top_k=40),
                },
            ]

            generation_df = pd.DataFrame(generations)
            display(generation_df)
            """
        ),
        md(
            """
            ## Step 12: Visualize the Context Window as a Memory Budget

            A decoder-only transformer does not have unlimited memory.
            It can only directly use the last `block_size` tokens.

            In the code cell below we:

            - simulate generation over time
            - show how many tokens are visible at each step
            - draw a visibility map that makes the fixed memory budget concrete
            """
        ),
        code(
            """
            prompt_ids = encode("ROMEO:\\n")
            simulated_steps = 80

            visible_history = [min(len(prompt_ids) + step, CONFIG["block_size"]) for step in range(simulated_steps)]
            visibility = np.zeros((simulated_steps, len(prompt_ids) + simulated_steps))

            for step in range(simulated_steps):
                total_length = len(prompt_ids) + step
                visible_start = max(0, total_length - CONFIG["block_size"])
                visibility[step, visible_start:total_length] = 1

            fig, axes = plt.subplots(1, 2, figsize=(18, 5))
            axes[0].plot(visible_history, linewidth=3, color="forestgreen")
            axes[0].axhline(CONFIG["block_size"], linestyle="--", color="firebrick", label="context limit")
            axes[0].set_title("How much history the model can see while generating")
            axes[0].set_xlabel("Generation step")
            axes[0].set_ylabel("Visible tokens")
            axes[0].legend()

            sns.heatmap(
                visibility[:, :120],
                cmap="Greens",
                cbar=False,
                ax=axes[1],
            )
            axes[1].set_title("Memory visibility map")
            axes[1].set_xlabel("Token position in the growing sequence")
            axes[1].set_ylabel("Generation step")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Wrap-Up

            This notebook built a real miniature transformer from raw text to generated output.
            The model is tiny compared with production LLMs, but the pipeline is the same:

            - represent text as tokens
            - learn embeddings
            - mix information with masked self-attention
            - train with next-token prediction
            - decode with sampling

            The next notebook extends this story to encoder-decoder models and sequence-to-sequence tasks.
            """
        ),
    ]
    return notebook("Mini Transformer From Scratch", cells)


def notebook_two():
    cells = [
        md(
            """
            # Attention, Encoder-Decoder Transformers, and Seq2Seq

            ## What We Are Going To Do

            The first notebook focused on a decoder-only model that predicts the next token.
            This notebook answers a different question:

            **What if the model needs to read one sequence and generate a different sequence?**

            That is the sequence-to-sequence setting used in tasks like:

            - translation
            - summarization
            - paraphrasing
            - structured question answering

            ## Big Ideas

            We will connect theory to practice in three layers:

            1. show why a fixed bottleneck is weak for long sequences
            2. show how attention solves that problem by letting the decoder look back at the source
            3. fine-tune a real encoder-decoder model on SAMSum summarization

            ## Learning Outcomes

            By the end of the notebook, students should be able to explain:

            - why attention is needed in seq2seq models
            - the difference between encoder self-attention and decoder cross-attention
            - how encoder and decoder hidden states interact
            - how a pretrained T5 model is fine-tuned for summarization
            - how cross-attention acts like task-specific memory retrieval
            """
        ),
        md(COMMON_INSTALL_MARKDOWN),
        code(COMMON_INSTALL_CODE),
        md(
            """
            ## Imports and Shared Helpers

            We use the Hugging Face stack for the real seq2seq part of this notebook.
            The theory is unchanged: we still tokenize, batch, optimize, and visualize.
            The difference is that the encoder-decoder architecture is already implemented and pretrained.

            In the code cell below we:

            - import plotting, datasets, and Hugging Face utilities
            - detect the compute device
            - create a checkpoint folder for the summarization model
            """
        ),
        code(
            """
            import inspect
            import random
            from pathlib import Path

            import evaluate
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import torch
            from datasets import DatasetDict, load_dataset
            from IPython.display import display
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                DataCollatorForSeq2Seq,
                Seq2SeqTrainer,
                Seq2SeqTrainingArguments,
            )

            sns.set_theme(style="whitegrid", context="talk")
            plt.rcParams["figure.figsize"] = (10, 6)

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)

            PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
            CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Using device: {DEVICE}")
            """
        ),
        md(
            """
            ## Configuration

            We keep this notebook flexible for class time.
            The fine-tuning section can either load a local checkpoint or run a short live fine-tuning pass.

            In the code cell below we:

            - choose the pretrained checkpoint
            - choose how many SAMSum examples to use in class
            - configure the output directory for the fine-tuned model
            - display the active settings
            """
        ),
        code(
            """
            USE_EXISTING_CHECKPOINT_IF_AVAILABLE = True
            MODEL_CHECKPOINT = "google-t5/t5-small"
            TRAIN_SAMPLES = 2200
            VALIDATION_SAMPLES = 300
            TEST_SAMPLES = 120
            MAX_SOURCE_LENGTH = 256
            MAX_TARGET_LENGTH = 64
            MAX_STEPS = 220

            LOCAL_OUTPUT_DIR = CHECKPOINT_DIR / "t5_samsum_demo"

            seq2seq_config_df = pd.DataFrame(
                [
                    ("model_checkpoint", MODEL_CHECKPOINT),
                    ("train_samples", TRAIN_SAMPLES),
                    ("validation_samples", VALIDATION_SAMPLES),
                    ("test_samples", TEST_SAMPLES),
                    ("max_source_length", MAX_SOURCE_LENGTH),
                    ("max_target_length", MAX_TARGET_LENGTH),
                    ("max_steps", MAX_STEPS),
                    ("local_output_dir", str(LOCAL_OUTPUT_DIR)),
                ],
                columns=["setting", "value"],
            )
            display(seq2seq_config_df)
            """
        ),
        md(
            """
            ## Why Attention Is Needed in Seq2Seq Models

            Early encoder-decoder models often tried to compress the whole input sequence into one fixed vector.
            That creates a bottleneck: every part of the output has to rely on the same compressed summary.

            Attention fixes that by letting each decoder step build a fresh weighted lookup over the source tokens.

            In the code cell below we:

            - build a toy source sequence and a toy target sequence
            - compare a fixed bottleneck representation with attention-based retrieval
            - visualize how different target tokens focus on different source positions
            """
        ),
        code(
            """
            source_tokens = ["alice", "ordered", "pizza", "for", "bob", "in", "chicago"]
            target_tokens = ["alice", "pizza", "bob", "chicago"]

            vocab = sorted(set(source_tokens + target_tokens))
            token_to_id = {token: idx for idx, token in enumerate(vocab)}

            source_matrix = np.eye(len(vocab))[ [token_to_id[token] for token in source_tokens] ]
            query_matrix = np.eye(len(vocab))[ [token_to_id[token] for token in target_tokens] ]

            raw_scores = query_matrix @ source_matrix.T
            scaled_scores = raw_scores * 6.0
            attention_weights = np.exp(scaled_scores) / np.exp(scaled_scores).sum(axis=1, keepdims=True)

            fixed_bottleneck = source_matrix.mean(axis=0, keepdims=True)
            repeated_bottleneck = np.repeat(fixed_bottleneck, len(target_tokens), axis=0)
            attention_context = attention_weights @ source_matrix

            fig, axes = plt.subplots(1, 3, figsize=(21, 5))
            sns.heatmap(
                raw_scores,
                annot=True,
                fmt=".0f",
                cmap="Blues",
                xticklabels=source_tokens,
                yticklabels=target_tokens,
                ax=axes[0],
            )
            axes[0].set_title("Raw match scores between decoder queries and source tokens")
            axes[0].set_xlabel("Source positions")
            axes[0].set_ylabel("Target positions")

            sns.heatmap(
                attention_weights,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                xticklabels=source_tokens,
                yticklabels=target_tokens,
                ax=axes[1],
            )
            axes[1].set_title("Attention turns scores into a lookup distribution")
            axes[1].set_xlabel("Source positions")
            axes[1].set_ylabel("Target positions")

            sns.heatmap(
                np.concatenate([repeated_bottleneck, attention_context], axis=1),
                cmap="magma",
                yticklabels=target_tokens,
                ax=axes[2],
            )
            axes[2].set_title("Left: fixed bottleneck | Right: attention-based context")
            axes[2].set_xlabel("Feature dimension")
            axes[2].set_ylabel("Target positions")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 1: Load and Inspect SAMSum

            SAMSum contains everyday messenger-style conversations paired with summaries.
            That makes it ideal for teaching because students can quickly understand both the input and the target.

            In the code cell below we:

            - load the dataset from Hugging Face
            - create small classroom subsets
            - preview one dialogue-summary pair
            - visualize conversation length and summary length
            """
        ),
        code(
            """
            samsum = load_dataset("knkarthick/samsum")

            if "validation" not in samsum:
                split = samsum["train"].train_test_split(test_size=0.1, seed=SEED)
                samsum = DatasetDict(
                    train=split["train"],
                    validation=split["test"],
                    test=samsum["test"] if "test" in samsum else split["test"],
                )

            dialogue_column = "dialogue" if "dialogue" in samsum["train"].column_names else "dialog"
            summary_column = "summary"

            small_samsum = DatasetDict(
                train=samsum["train"].shuffle(seed=SEED).select(range(min(TRAIN_SAMPLES, len(samsum["train"])))),
                validation=samsum["validation"].shuffle(seed=SEED).select(range(min(VALIDATION_SAMPLES, len(samsum["validation"])))),
                test=samsum["test"].shuffle(seed=SEED).select(range(min(TEST_SAMPLES, len(samsum["test"])))),
            )

            example = small_samsum["train"][0]
            print("Dialogue sample:")
            print(example[dialogue_column])
            print()
            print("Reference summary:")
            print(example[summary_column])

            dialogue_lengths = pd.Series([len(text.split()) for text in small_samsum["train"][dialogue_column]])
            summary_lengths = pd.Series([len(text.split()) for text in small_samsum["train"][summary_column]])
            turn_counts = pd.Series([text.count("\\n") + 1 for text in small_samsum["train"][dialogue_column]])

            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            sns.histplot(dialogue_lengths, bins=30, color="steelblue", ax=axes[0])
            axes[0].set_title("Dialogue length in words")
            axes[0].set_xlabel("Words")

            sns.histplot(summary_lengths, bins=25, color="darkorange", ax=axes[1])
            axes[1].set_title("Summary length in words")
            axes[1].set_xlabel("Words")

            sns.histplot(turn_counts, bins=20, color="seagreen", ax=axes[2])
            axes[2].set_title("Number of turns per conversation")
            axes[2].set_xlabel("Turns")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 2: Tokenize Inputs and Targets

            In encoder-decoder summarization we tokenize two different things:

            - the source dialogue
            - the target summary

            T5 is also a prompted multitask model, so we prepend `summarize:` to the source text.

            In the code cell below we:

            - load the T5 tokenizer
            - preprocess the dialogue-summary pairs
            - create tokenized train, validation, and test sets
            - visualize source and target length distributions
            """
        ),
        code(
            """
            tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
            prefix = "summarize: "

            def preprocess_batch(examples):
                model_inputs = tokenizer(
                    [prefix + dialogue for dialogue in examples[dialogue_column]],
                    max_length=MAX_SOURCE_LENGTH,
                    truncation=True,
                )
                labels = tokenizer(
                    text_target=examples[summary_column],
                    max_length=MAX_TARGET_LENGTH,
                    truncation=True,
                )
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            tokenized_samsum = small_samsum.map(
                preprocess_batch,
                batched=True,
                remove_columns=small_samsum["train"].column_names,
            )

            source_lengths = pd.Series([len(row) for row in tokenized_samsum["train"]["input_ids"]])
            target_lengths = pd.Series([len(row) for row in tokenized_samsum["train"]["labels"]])

            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            sns.histplot(source_lengths, bins=30, color="mediumpurple", ax=axes[0])
            axes[0].set_title("Tokenized source lengths")
            axes[0].set_xlabel("Tokens")

            sns.histplot(target_lengths, bins=25, color="tomato", ax=axes[1])
            axes[1].set_title("Tokenized target lengths")
            axes[1].set_xlabel("Tokens")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 3: Run One Forward Pass and Inspect Encoder/Decoder Shapes

            The encoder reads the dialogue and produces hidden states.
            The decoder reads the partly generated summary and uses cross-attention to look back into those encoder states.

            In the code cell below we:

            - load the pretrained T5 model
            - run a forward pass on one example
            - inspect the shape of logits, encoder hidden states, and attention tensors
            """
        ),
        code(
            """
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT).to(DEVICE)
            model.eval()

            sample_dialogue = example[dialogue_column]
            sample_summary = example[summary_column]

            model_inputs = tokenizer(
                prefix + sample_dialogue,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SOURCE_LENGTH,
            ).to(DEVICE)
            label_inputs = tokenizer(
                text_target=sample_summary,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TARGET_LENGTH,
            ).input_ids.to(DEVICE)

            with torch.no_grad():
                outputs = model(
                    **model_inputs,
                    labels=label_inputs,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            shape_df = pd.DataFrame(
                [
                    ("encoder hidden state", tuple(outputs.encoder_last_hidden_state.shape)),
                    ("logits", tuple(outputs.logits.shape)),
                    ("encoder attention layer 0", tuple(outputs.encoder_attentions[0].shape)),
                    ("decoder attention layer 0", tuple(outputs.decoder_attentions[0].shape)),
                    ("cross attention layer 0", tuple(outputs.cross_attentions[0].shape)),
                ],
                columns=["tensor", "shape"],
            )
            display(shape_df)
            """
        ),
        md(
            """
            ## Step 4: Visualize Encoder Self-Attention and Decoder Cross-Attention

            This is the heart of the encoder-decoder story.

            - encoder self-attention mixes information inside the source dialogue
            - decoder self-attention mixes information inside the partially generated summary
            - cross-attention lets each summary token decide which source tokens matter right now

            In the code cell below we:

            - decode tokens back into readable pieces
            - visualize encoder self-attention from one layer and head
            - visualize decoder cross-attention from one layer and head
            """
        ),
        code(
            """
            encoder_tokens = tokenizer.convert_ids_to_tokens(model_inputs["input_ids"][0])
            decoder_input_ids = model._shift_right(label_inputs)
            decoder_tokens = tokenizer.convert_ids_to_tokens(decoder_input_ids[0])

            with torch.no_grad():
                attention_outputs = model(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    labels=label_inputs,
                    output_attentions=True,
                    return_dict=True,
                )

            encoder_attention = attention_outputs.encoder_attentions[0][0, 0].detach().cpu().numpy()
            cross_attention = attention_outputs.cross_attentions[0][0, 0].detach().cpu().numpy()

            encoder_plot_length = min(24, len(encoder_tokens))
            decoder_plot_length = min(16, len(decoder_tokens))

            fig, axes = plt.subplots(1, 2, figsize=(20, 7))
            sns.heatmap(
                encoder_attention[:encoder_plot_length, :encoder_plot_length],
                cmap="viridis",
                xticklabels=encoder_tokens[:encoder_plot_length],
                yticklabels=encoder_tokens[:encoder_plot_length],
                ax=axes[0],
            )
            axes[0].set_title("Encoder self-attention")
            axes[0].set_xlabel("Source keys")
            axes[0].set_ylabel("Source queries")

            sns.heatmap(
                cross_attention[:decoder_plot_length, :encoder_plot_length],
                cmap="magma",
                xticklabels=encoder_tokens[:encoder_plot_length],
                yticklabels=decoder_tokens[:decoder_plot_length],
                ax=axes[1],
            )
            axes[1].set_title("Decoder cross-attention")
            axes[1].set_xlabel("Source tokens the decoder can look at")
            axes[1].set_ylabel("Current summary tokens")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 5: Prepare the Fine-Tuning Pipeline

            We now move from inspection to learning.
            Fine-tuning updates pretrained weights so the model adapts to our summarization dataset.

            In the code cell below we:

            - build a compatibility helper for current or slightly older Hugging Face versions
            - create the seq2seq data collator
            - define ROUGE-based evaluation
            """
        ),
        code(
            """
            rouge = evaluate.load("rouge")
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL_CHECKPOINT)

            def make_seq2seq_training_args(**kwargs):
                parameters = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
                if "eval_strategy" not in parameters and "eval_strategy" in kwargs:
                    kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
                return Seq2SeqTrainingArguments(**kwargs)

            def processor_argument(processor):
                parameters = inspect.signature(Seq2SeqTrainer.__init__).parameters
                if "processing_class" in parameters:
                    return {"processing_class": processor}
                return {"tokenizer": processor}

            def compute_metrics(eval_prediction):
                predictions, labels = eval_prediction
                if isinstance(predictions, tuple):
                    predictions = predictions[0]

                decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                result = rouge.compute(
                    predictions=decoded_predictions,
                    references=decoded_labels,
                    use_stemmer=True,
                )
                prediction_lengths = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
                result["generated_length"] = float(np.mean(prediction_lengths))
                return {key: round(value, 4) for key, value in result.items()}

            print("Seq2Seq fine-tuning helpers are ready.")
            """
        ),
        md(
            """
            ## Step 6: Fine-Tune T5 or Load an Existing Checkpoint

            This is the live training section of the notebook.
            If a checkpoint is already present, we load it to save class time.
            Otherwise we run a short fine-tuning pass and then save the result locally.

            In the code cell below we:

            - load or fine-tune the summarization model
            - track training and evaluation metrics
            - visualize the learning curve
            """
        ),
        code(
            """
            training_history = None

            if LOCAL_OUTPUT_DIR.exists() and USE_EXISTING_CHECKPOINT_IF_AVAILABLE:
                seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_OUTPUT_DIR).to(DEVICE)
                tokenizer = AutoTokenizer.from_pretrained(LOCAL_OUTPUT_DIR)
                print(f"Loaded fine-tuned checkpoint from {LOCAL_OUTPUT_DIR}")
            else:
                seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT).to(DEVICE)

                training_args = make_seq2seq_training_args(
                    output_dir=str(LOCAL_OUTPUT_DIR),
                    max_steps=MAX_STEPS,
                    per_device_train_batch_size=4,
                    per_device_eval_batch_size=4,
                    learning_rate=2e-4,
                    weight_decay=0.01,
                    eval_strategy="steps",
                    eval_steps=50,
                    save_steps=50,
                    logging_steps=25,
                    save_total_limit=2,
                    predict_with_generate=True,
                    generation_max_length=MAX_TARGET_LENGTH,
                    fp16=(DEVICE == "cuda"),
                    report_to="none",
                )

                trainer = Seq2SeqTrainer(
                    model=seq2seq_model,
                    args=training_args,
                    train_dataset=tokenized_samsum["train"],
                    eval_dataset=tokenized_samsum["validation"],
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    **processor_argument(tokenizer),
                )

                trainer.train()
                trainer.save_model()
                tokenizer.save_pretrained(LOCAL_OUTPUT_DIR)
                print(f"Saved fine-tuned checkpoint to {LOCAL_OUTPUT_DIR}")

                training_history = pd.DataFrame(trainer.state.log_history)

            if training_history is not None and not training_history.empty:
                plt.figure(figsize=(10, 5))
                if "loss" in training_history:
                    loss_rows = training_history.dropna(subset=["loss"])
                    plt.plot(loss_rows["step"], loss_rows["loss"], label="train loss")
                if "eval_loss" in training_history:
                    eval_rows = training_history.dropna(subset=["eval_loss"])
                    plt.plot(eval_rows["step"], eval_rows["eval_loss"], label="eval loss")
                plt.title("T5 fine-tuning learning curve")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.legend()
                plt.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            ## Step 7: Summarize Unseen Conversations

            Training only matters if the model produces useful outputs on new examples.
            For summarization, the easiest classroom check is to compare generated summaries against references.

            In the code cell below we:

            - run generation on held-out conversations
            - decode predictions back into text
            - compare predictions with reference summaries
            """
        ),
        code(
            """
            seq2seq_model.eval()

            held_out_examples = [small_samsum["test"][idx] for idx in range(min(3, len(small_samsum["test"])))]
            predictions = []

            for held_out in held_out_examples:
                inputs = tokenizer(
                    prefix + held_out[dialogue_column],
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SOURCE_LENGTH,
                ).to(DEVICE)
                generated_ids = seq2seq_model.generate(
                    **inputs,
                    max_new_tokens=MAX_TARGET_LENGTH,
                    do_sample=False,
                )
                prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                predictions.append(
                    {
                        "dialogue_preview": held_out[dialogue_column][:180] + "...",
                        "reference_summary": held_out[summary_column],
                        "model_summary": prediction,
                    }
                )

            prediction_df = pd.DataFrame(predictions)
            display(prediction_df)
            """
        ),
        md(
            """
            ## Step 8: Visualize Cross-Attention as Seq2Seq Memory

            In decoder-only models, memory means the visible left context.
            In encoder-decoder models, memory is more structured:

            - the encoder stores the source sequence as hidden states
            - the decoder retrieves from that memory through cross-attention

            In the code cell below we:

            - select one generated example
            - inspect one decoder token's cross-attention distribution
            - visualize which source tokens the summary token relied on
            """
        ),
        code(
            """
            attention_example = held_out_examples[0]
            attention_inputs = tokenizer(
                prefix + attention_example[dialogue_column],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SOURCE_LENGTH,
            ).to(DEVICE)
            attention_labels = tokenizer(
                text_target=attention_example[summary_column],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TARGET_LENGTH,
            ).input_ids.to(DEVICE)

            with torch.no_grad():
                attention_result = seq2seq_model(
                    **attention_inputs,
                    labels=attention_labels,
                    output_attentions=True,
                    return_dict=True,
                )

            cross_attention_map = attention_result.cross_attentions[0][0, 0].detach().cpu().numpy()
            source_tokens = tokenizer.convert_ids_to_tokens(attention_inputs["input_ids"][0])
            target_tokens = tokenizer.convert_ids_to_tokens(seq2seq_model._shift_right(attention_labels)[0])

            last_target_index = min(len(target_tokens) - 1, 10)
            source_plot_length = min(28, len(source_tokens))

            plt.figure(figsize=(14, 5))
            plt.bar(
                range(source_plot_length),
                cross_attention_map[last_target_index, :source_plot_length],
                color="crimson",
            )
            plt.xticks(range(source_plot_length), source_tokens[:source_plot_length], rotation=90)
            plt.title(
                f"Cross-attention for target token '{target_tokens[last_target_index]}'"
            )
            plt.xlabel("Source token position")
            plt.ylabel("Attention weight")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Wrap-Up

            This notebook made the encoder-decoder pattern concrete.
            Students should now be able to separate three attention roles:

            - encoder self-attention: mix information within the input
            - decoder self-attention: mix information within the generated output prefix
            - cross-attention: retrieve what matters from the encoded input

            The final notebook moves from architecture understanding to applications:
            BERT-style classification, GPT-style generation, and Hugging Face workflows.
            """
        ),
    ]
    return notebook("Attention, Encoder-Decoder Transformers, and Seq2Seq", cells)


def notebook_three():
    cells = [
        md(
            """
            # Fine-Tuning and Applications With Hugging Face

            ## What We Are Going To Do

            The first notebook built a tiny GPT from scratch.
            The second notebook explained encoder-decoder transformers for seq2seq tasks.
            This notebook shifts to the modern workflow students will use most often in practice:

            - start from a pretrained transformer
            - adapt it to a downstream task
            - evaluate it
            - turn it into an application

            ## Tasks Covered

            - BERT-family sequence classification on IMDb
            - GPT-style dialogue generation on DailyDialog
            - summarization and chatbot inference using Hugging Face models

            ## Learning Outcomes

            By the end of the notebook, students should be able to explain:

            - the difference between bidirectional and causal attention
            - how fine-tuning differs from training from scratch
            - how Hugging Face datasets, tokenizers, data collators, and trainers fit together
            - how to interpret logits, probabilities, and generated text in downstream applications
            """
        ),
        md(COMMON_INSTALL_MARKDOWN),
        code(COMMON_INSTALL_CODE),
        md(
            """
            ## Imports and Shared Helpers

            We are now using the higher-level Hugging Face training stack for multiple tasks.
            The theory remains the same, but the abstractions let us focus on task design instead of boilerplate.

            In the code cell below we:

            - import the libraries needed for classification and generation
            - set reproducible seeds
            - prepare the checkpoint directory for task-specific models
            """
        ),
        code(
            """
            import inspect
            import random
            from pathlib import Path

            import evaluate
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import torch
            from datasets import DatasetDict, load_dataset
            from IPython.display import display
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                DataCollatorForLanguageModeling,
                DataCollatorWithPadding,
                Trainer,
                TrainingArguments,
            )

            sns.set_theme(style="whitegrid", context="talk")
            plt.rcParams["figure.figsize"] = (10, 6)

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)

            PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
            CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Using device: {DEVICE}")
            """
        ),
        md(
            """
            ## Configuration

            This notebook includes two fine-tuning sections.
            The BERT-style classifier is intended to run in class.
            The GPT-style dialogue model is included as a teaching extension and can be skipped or loaded from checkpoint.

            In the code cell below we:

            - choose pretrained checkpoints for each task
            - choose classroom subset sizes
            - define checkpoint output folders
            - display the active plan
            """
        ),
        code(
            """
            BERT_CHECKPOINT = "distilbert/distilbert-base-uncased"
            GPT_CHECKPOINT = "distilbert/distilgpt2"
            T5_FALLBACK_CHECKPOINT = "google-t5/t5-small"

            IMDB_TRAIN_SAMPLES = 3500
            IMDB_VALIDATION_SAMPLES = 1000
            DAILYDIALOG_TRAIN_SAMPLES = 2200
            DAILYDIALOG_VALIDATION_SAMPLES = 400

            RUN_GPT_FINE_TUNING = False
            USE_EXISTING_CHECKPOINT_IF_AVAILABLE = True

            IMDB_OUTPUT_DIR = CHECKPOINT_DIR / "distilbert_imdb_demo"
            GPT_OUTPUT_DIR = CHECKPOINT_DIR / "distilgpt2_dailydialog_demo"

            config_table = pd.DataFrame(
                [
                    ("bert_checkpoint", BERT_CHECKPOINT),
                    ("gpt_checkpoint", GPT_CHECKPOINT),
                    ("imdb_train_samples", IMDB_TRAIN_SAMPLES),
                    ("imdb_validation_samples", IMDB_VALIDATION_SAMPLES),
                    ("dailydialog_train_samples", DAILYDIALOG_TRAIN_SAMPLES),
                    ("dailydialog_validation_samples", DAILYDIALOG_VALIDATION_SAMPLES),
                    ("run_gpt_fine_tuning", RUN_GPT_FINE_TUNING),
                    ("imdb_output_dir", str(IMDB_OUTPUT_DIR)),
                    ("gpt_output_dir", str(GPT_OUTPUT_DIR)),
                ],
                columns=["setting", "value"],
            )
            display(config_table)
            """
        ),
        md(
            """
            ## BERT vs GPT: Bidirectional vs Causal Attention

            Before we fine-tune anything, it helps to compare the attention patterns conceptually.

            - BERT-style encoders can attend in both directions
            - GPT-style decoders can only attend to the left

            In the code cell below we:

            - build a small token sequence
            - draw a bidirectional attention mask
            - draw a causal attention mask
            - summarize the architectural differences in a table
            """
        ),
        code(
            """
            comparison_tokens = ["The", "movie", "was", "surprisingly", "good"]
            bidirectional_mask = np.ones((len(comparison_tokens), len(comparison_tokens)))
            causal_mask = np.tril(np.ones((len(comparison_tokens), len(comparison_tokens))))

            architecture_df = pd.DataFrame(
                [
                    ("BERT family", "encoder", "bidirectional", "classification, retrieval, embeddings"),
                    ("GPT family", "decoder", "causal", "generation, chat, completion"),
                    ("T5 family", "encoder-decoder", "both + cross-attention", "summarization, translation"),
                ],
                columns=["model family", "core block", "attention pattern", "common tasks"],
            )
            display(architecture_df)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.heatmap(
                bidirectional_mask,
                annot=True,
                cmap="Blues",
                cbar=False,
                xticklabels=comparison_tokens,
                yticklabels=comparison_tokens,
                ax=axes[0],
            )
            axes[0].set_title("Bidirectional mask (BERT-style)")
            axes[0].set_xlabel("Visible tokens")
            axes[0].set_ylabel("Current token")

            sns.heatmap(
                causal_mask,
                annot=True,
                cmap="Reds",
                cbar=False,
                xticklabels=comparison_tokens,
                yticklabels=comparison_tokens,
                ax=axes[1],
            )
            axes[1].set_title("Causal mask (GPT-style)")
            axes[1].set_xlabel("Visible tokens")
            axes[1].set_ylabel("Current token")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 1: Load and Inspect IMDb for Classification

            IMDb is a classic sentiment dataset with movie reviews labeled as positive or negative.
            It is a good fine-tuning dataset because the task is intuitive and the labels are easy to interpret.

            In the code cell below we:

            - load the dataset
            - create classroom-sized train and validation subsets
            - visualize label balance and review length
            """
        ),
        code(
            """
            imdb = load_dataset("stanfordnlp/imdb")

            small_imdb = DatasetDict(
                train=imdb["train"].shuffle(seed=SEED).select(range(min(IMDB_TRAIN_SAMPLES, len(imdb["train"])))),
                validation=imdb["test"].shuffle(seed=SEED).select(range(min(IMDB_VALIDATION_SAMPLES, len(imdb["test"])))),
            )

            label_names = {0: "negative", 1: "positive"}
            label_counts = pd.Series(small_imdb["train"]["label"]).map(label_names).value_counts()
            review_lengths = pd.Series([len(text.split()) for text in small_imdb["train"]["text"]])

            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            label_counts.plot(kind="bar", color=["firebrick", "seagreen"], ax=axes[0])
            axes[0].set_title("IMDb label balance")
            axes[0].set_xlabel("Class")
            axes[0].set_ylabel("Count")
            axes[0].tick_params(axis="x", rotation=0)

            sns.histplot(review_lengths, bins=35, color="steelblue", ax=axes[1])
            axes[1].set_title("Review length distribution")
            axes[1].set_xlabel("Words")
            axes[1].set_ylabel("Number of reviews")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 2: Tokenize the Reviews

            Fine-tuning a pretrained model means using the tokenizer that matches the checkpoint.
            For sequence classification, the tokenizer creates:

            - `input_ids`
            - `attention_mask`

            In the code cell below we:

            - load the DistilBERT tokenizer
            - tokenize the train and validation subsets
            - visualize sequence lengths and one sample attention mask
            """
        ),
        code(
            """
            bert_tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT)

            def tokenize_reviews(batch):
                return bert_tokenizer(batch["text"], truncation=True, max_length=256)

            tokenized_imdb = small_imdb.map(tokenize_reviews, batched=True)
            bert_lengths = pd.Series([len(ids) for ids in tokenized_imdb["train"]["input_ids"]])

            sample_review_encoding = bert_tokenizer(
                small_imdb["train"][0]["text"],
                truncation=True,
                max_length=48,
            )

            fig, axes = plt.subplots(1, 2, figsize=(16, 4))
            sns.histplot(bert_lengths, bins=30, color="purple", ax=axes[0])
            axes[0].set_title("BERT tokenized review lengths")
            axes[0].set_xlabel("Tokens")

            sns.heatmap(
                np.array(sample_review_encoding["attention_mask"]).reshape(1, -1),
                cmap="Greens",
                cbar=False,
                ax=axes[1],
            )
            axes[1].set_title("Attention mask for one padded/truncated example")
            axes[1].set_xlabel("Token position")
            axes[1].set_yticks([])
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 3: Look at Real BERT Attention

            The conceptual mask earlier was hand-made.
            Here we inspect actual attention weights from a pretrained classifier backbone.

            In the code cell below we:

            - load DistilBERT with attention outputs enabled
            - run one short example through the model
            - visualize one attention head so students can see bidirectional connectivity
            """
        ),
        code(
            """
            bert_attention_model = AutoModelForSequenceClassification.from_pretrained(
                BERT_CHECKPOINT,
                num_labels=2,
                output_attentions=True,
            ).to(DEVICE)
            bert_attention_model.eval()

            short_text = "This movie was surprisingly thoughtful, funny, and well acted."
            short_inputs = bert_tokenizer(
                short_text,
                return_tensors="pt",
                truncation=True,
                max_length=24,
            ).to(DEVICE)

            with torch.no_grad():
                bert_outputs = bert_attention_model(**short_inputs)

            bert_tokens = bert_tokenizer.convert_ids_to_tokens(short_inputs["input_ids"][0])
            bert_attention = bert_outputs.attentions[0][0, 0].detach().cpu().numpy()
            plot_len = min(18, len(bert_tokens))

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                bert_attention[:plot_len, :plot_len],
                cmap="viridis",
                xticklabels=bert_tokens[:plot_len],
                yticklabels=bert_tokens[:plot_len],
            )
            plt.title("One DistilBERT attention head")
            plt.xlabel("Visible tokens")
            plt.ylabel("Current token")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 4: Prepare the IMDb Fine-Tuning Pipeline

            This section builds the training loop around the pretrained classifier.
            The underlying theory is the same as before: logits go into cross-entropy loss,
            gradients update the weights, and validation metrics tell us whether the model improves.

            In the code cell below we:

            - define compatibility helpers for `TrainingArguments` and `Trainer`
            - create a dynamic padding collator
            - define the accuracy metric
            """
        ),
        code(
            """
            imdb_accuracy = evaluate.load("accuracy")
            classification_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)

            def make_training_args(**kwargs):
                parameters = inspect.signature(TrainingArguments.__init__).parameters
                if "eval_strategy" not in parameters and "eval_strategy" in kwargs:
                    kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
                return TrainingArguments(**kwargs)

            def trainer_processor_argument(processor):
                parameters = inspect.signature(Trainer.__init__).parameters
                if "processing_class" in parameters:
                    return {"processing_class": processor}
                return {"tokenizer": processor}

            def compute_classification_metrics(eval_prediction):
                logits, labels = eval_prediction
                predictions = np.argmax(logits, axis=-1)
                return imdb_accuracy.compute(predictions=predictions, references=labels)

            print("Classification fine-tuning helpers are ready.")
            """
        ),
        md(
            """
            ## Step 5: Fine-Tune DistilBERT or Load a Saved Checkpoint

            Fine-tuning is much faster than training from scratch because we start from useful pretrained weights.

            In the code cell below we:

            - load or fine-tune the classifier
            - save the best local checkpoint
            - plot the training and evaluation curves
            """
        ),
        code(
            """
            bert_history = None

            if IMDB_OUTPUT_DIR.exists() and USE_EXISTING_CHECKPOINT_IF_AVAILABLE:
                classifier_model = AutoModelForSequenceClassification.from_pretrained(IMDB_OUTPUT_DIR).to(DEVICE)
                bert_tokenizer = AutoTokenizer.from_pretrained(IMDB_OUTPUT_DIR)
                print(f"Loaded fine-tuned classifier from {IMDB_OUTPUT_DIR}")
            else:
                classifier_model = AutoModelForSequenceClassification.from_pretrained(
                    BERT_CHECKPOINT,
                    num_labels=2,
                    id2label={0: "NEGATIVE", 1: "POSITIVE"},
                    label2id={"NEGATIVE": 0, "POSITIVE": 1},
                ).to(DEVICE)

                training_args = make_training_args(
                    output_dir=str(IMDB_OUTPUT_DIR),
                    max_steps=250,
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    learning_rate=2e-5,
                    weight_decay=0.01,
                    eval_strategy="steps",
                    eval_steps=50,
                    save_steps=50,
                    logging_steps=25,
                    save_total_limit=2,
                    fp16=(DEVICE == "cuda"),
                    report_to="none",
                )

                trainer = Trainer(
                    model=classifier_model,
                    args=training_args,
                    train_dataset=tokenized_imdb["train"],
                    eval_dataset=tokenized_imdb["validation"],
                    data_collator=classification_collator,
                    compute_metrics=compute_classification_metrics,
                    **trainer_processor_argument(bert_tokenizer),
                )

                trainer.train()
                trainer.save_model()
                bert_tokenizer.save_pretrained(IMDB_OUTPUT_DIR)
                bert_history = pd.DataFrame(trainer.state.log_history)
                print(f"Saved fine-tuned classifier to {IMDB_OUTPUT_DIR}")

            if bert_history is not None and not bert_history.empty:
                plt.figure(figsize=(10, 5))
                if "loss" in bert_history:
                    loss_rows = bert_history.dropna(subset=["loss"])
                    plt.plot(loss_rows["step"], loss_rows["loss"], label="train loss")
                if "eval_loss" in bert_history:
                    eval_rows = bert_history.dropna(subset=["eval_loss"])
                    plt.plot(eval_rows["step"], eval_rows["eval_loss"], label="eval loss")
                plt.title("DistilBERT fine-tuning curve on IMDb")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.legend()
                plt.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            ## Step 6: Run Classification Inference and Inspect Probabilities

            A classification model outputs logits for each class.
            Applying softmax turns those logits into probabilities.

            In the code cell below we:

            - run the fine-tuned classifier on custom reviews
            - compute softmax probabilities
            - visualize how confident the model is about each label
            """
        ),
        code(
            """
            classifier_model.eval()

            demo_reviews = [
                "This was a sharp, well paced film with excellent performances.",
                "The plot was messy, the acting was flat, and I wanted it to end.",
                "I expected nonsense, but it was weirdly charming and sincere.",
            ]

            encoded_batch = bert_tokenizer(
                demo_reviews,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(DEVICE)

            with torch.no_grad():
                logits = classifier_model(**encoded_batch).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

            probability_df = pd.DataFrame(
                probabilities,
                columns=["negative_probability", "positive_probability"],
            )
            probability_df.insert(0, "review", demo_reviews)
            display(probability_df)

            plt.figure(figsize=(10, 5))
            x_positions = np.arange(len(demo_reviews))
            plt.bar(x_positions - 0.15, probabilities[:, 0], width=0.3, label="negative")
            plt.bar(x_positions + 0.15, probabilities[:, 1], width=0.3, label="positive")
            plt.xticks(x_positions, [f"Review {idx + 1}" for idx in range(len(demo_reviews))])
            plt.title("Classifier confidence on custom examples")
            plt.ylabel("Probability")
            plt.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 7: Load and Format DailyDialog for GPT-Style Generation

            For a chatbot-style causal language model, we flatten dialogues into a single text stream.
            The model learns to predict the next token in that stream, which means it implicitly learns reply patterns.

            In the code cell below we:

            - load DailyDialog
            - format each conversation as alternating speaker turns
            - create classroom-sized train and validation subsets
            - visualize dialogue length statistics
            """
        ),
        code(
            """
            dailydialog = load_dataset("li2017dailydialog/daily_dialog")
            dialog_column = "dialog" if "dialog" in dailydialog["train"].column_names else "dialogue"

            def format_dialogue(example):
                turns = example[dialog_column]
                speaker_turns = [f"Speaker {(idx % 2) + 1}: {turn}" for idx, turn in enumerate(turns)]
                formatted = " <eos> ".join(speaker_turns)
                prompt = " <eos> ".join(speaker_turns[:-1]) if len(speaker_turns) > 1 else speaker_turns[0]
                target_reply = speaker_turns[-1]
                example["formatted_dialogue"] = formatted
                example["prompt_text"] = prompt
                example["target_reply"] = target_reply
                return example

            formatted_dailydialog = dailydialog.map(format_dialogue)
            small_dialog = DatasetDict(
                train=formatted_dailydialog["train"].shuffle(seed=SEED).select(
                    range(min(DAILYDIALOG_TRAIN_SAMPLES, len(formatted_dailydialog["train"])))
                ),
                validation=formatted_dailydialog["validation"].shuffle(seed=SEED).select(
                    range(min(DAILYDIALOG_VALIDATION_SAMPLES, len(formatted_dailydialog["validation"])))
                ),
            )

            turn_counts = pd.Series([text.count("<eos>") + 1 for text in small_dialog["train"]["formatted_dialogue"]])
            dialogue_word_lengths = pd.Series([len(text.split()) for text in small_dialog["train"]["formatted_dialogue"]])

            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            sns.histplot(turn_counts, bins=25, color="teal", ax=axes[0])
            axes[0].set_title("Turns per dialogue")
            axes[0].set_xlabel("Turns")

            sns.histplot(dialogue_word_lengths, bins=30, color="darkgoldenrod", ax=axes[1])
            axes[1].set_title("Dialogue length in words")
            axes[1].set_xlabel("Words")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 8: Tokenize Dialogue Data for a GPT-Style Model

            GPT models are causal language models.
            During fine-tuning we feed full dialogue strings and ask the model to predict every next token.

            In the code cell below we:

            - load the GPT tokenizer
            - tokenize the formatted dialogues
            - inspect token lengths
            - show one tokenized example
            """
        ),
        code(
            """
            gpt_tokenizer = AutoTokenizer.from_pretrained(GPT_CHECKPOINT)
            if gpt_tokenizer.pad_token is None:
                gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

            def tokenize_dialogues(batch):
                return gpt_tokenizer(
                    batch["formatted_dialogue"],
                    truncation=True,
                    max_length=256,
                )

            tokenized_dialog = small_dialog.map(
                tokenize_dialogues,
                batched=True,
                remove_columns=small_dialog["train"].column_names,
            )

            dialog_token_lengths = pd.Series([len(ids) for ids in tokenized_dialog["train"]["input_ids"]])
            sample_dialog_tokens = gpt_tokenizer.convert_ids_to_tokens(tokenized_dialog["train"][0]["input_ids"][:40])

            print("Sample tokenized dialogue pieces:")
            print(sample_dialog_tokens)

            plt.figure(figsize=(10, 5))
            sns.histplot(dialog_token_lengths, bins=30, color="slateblue")
            plt.title("Token lengths for GPT dialogue fine-tuning samples")
            plt.xlabel("Tokens")
            plt.ylabel("Number of dialogues")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 9: Visualize Real GPT Attention

            This section mirrors the BERT attention inspection, but now for a causal model.
            The goal is to make the left-to-right constraint visible in real model outputs.

            In the code cell below we:

            - load DistilGPT2 with attention outputs enabled
            - run a short prompt through the model
            - visualize one attention head
            """
        ),
        code(
            """
            gpt_attention_model = AutoModelForCausalLM.from_pretrained(
                GPT_CHECKPOINT,
                output_attentions=True,
            ).to(DEVICE)
            gpt_attention_model.eval()

            prompt_for_attention = "Speaker 1: Are you free tonight? <eos> Speaker 2:"
            prompt_inputs = gpt_tokenizer(
                prompt_for_attention,
                return_tensors="pt",
                truncation=True,
                max_length=32,
            ).to(DEVICE)

            with torch.no_grad():
                gpt_outputs = gpt_attention_model(**prompt_inputs)

            gpt_tokens = gpt_tokenizer.convert_ids_to_tokens(prompt_inputs["input_ids"][0])
            gpt_attention = gpt_outputs.attentions[0][0, 0].detach().cpu().numpy()
            plot_len = min(20, len(gpt_tokens))

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                gpt_attention[:plot_len, :plot_len],
                cmap="magma",
                xticklabels=gpt_tokens[:plot_len],
                yticklabels=gpt_tokens[:plot_len],
            )
            plt.title("One DistilGPT2 attention head")
            plt.xlabel("Visible earlier tokens")
            plt.ylabel("Current token")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Step 10: Optionally Fine-Tune a GPT-Style Dialogue Model

            This section is deliberately optional because it is the first thing to skip if class time is short.
            It is still pedagogically useful because it shows that the same Trainer workflow also works for causal LM tasks.

            In the code cell below we:

            - define a language-modeling data collator
            - either load a saved GPT checkpoint or run a short fine-tuning session
            - plot the learning curve when training happens live
            """
        ),
        code(
            """
            gpt_history = None
            lm_collator = DataCollatorForLanguageModeling(tokenizer=gpt_tokenizer, mlm=False)

            if GPT_OUTPUT_DIR.exists() and USE_EXISTING_CHECKPOINT_IF_AVAILABLE:
                dialogue_model = AutoModelForCausalLM.from_pretrained(GPT_OUTPUT_DIR).to(DEVICE)
                gpt_tokenizer = AutoTokenizer.from_pretrained(GPT_OUTPUT_DIR)
                print(f"Loaded GPT dialogue checkpoint from {GPT_OUTPUT_DIR}")
            elif RUN_GPT_FINE_TUNING:
                dialogue_model = AutoModelForCausalLM.from_pretrained(GPT_CHECKPOINT).to(DEVICE)

                gpt_training_args = make_training_args(
                    output_dir=str(GPT_OUTPUT_DIR),
                    max_steps=180,
                    per_device_train_batch_size=4,
                    per_device_eval_batch_size=4,
                    learning_rate=5e-5,
                    weight_decay=0.01,
                    eval_strategy="steps",
                    eval_steps=45,
                    save_steps=45,
                    logging_steps=20,
                    save_total_limit=2,
                    fp16=(DEVICE == "cuda"),
                    report_to="none",
                )

                gpt_trainer = Trainer(
                    model=dialogue_model,
                    args=gpt_training_args,
                    train_dataset=tokenized_dialog["train"],
                    eval_dataset=tokenized_dialog["validation"],
                    data_collator=lm_collator,
                    **trainer_processor_argument(gpt_tokenizer),
                )

                gpt_trainer.train()
                gpt_trainer.save_model()
                gpt_tokenizer.save_pretrained(GPT_OUTPUT_DIR)
                gpt_history = pd.DataFrame(gpt_trainer.state.log_history)
                print(f"Saved GPT dialogue checkpoint to {GPT_OUTPUT_DIR}")
            else:
                dialogue_model = AutoModelForCausalLM.from_pretrained(GPT_CHECKPOINT).to(DEVICE)
                print("Skipping GPT fine-tuning in this run. Using the pretrained checkpoint for generation demos.")

            if gpt_history is not None and not gpt_history.empty:
                plt.figure(figsize=(10, 5))
                if "loss" in gpt_history:
                    loss_rows = gpt_history.dropna(subset=["loss"])
                    plt.plot(loss_rows["step"], loss_rows["loss"], label="train loss")
                if "eval_loss" in gpt_history:
                    eval_rows = gpt_history.dropna(subset=["eval_loss"])
                    plt.plot(eval_rows["step"], eval_rows["eval_loss"], label="eval loss")
                plt.title("GPT-style dialogue fine-tuning curve")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.legend()
                plt.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            ## Step 11: Build Simple Applications

            A notebook on LLMs should end with applications students recognize.
            We will do two:

            - a chatbot-style next-reply generator
            - a summarizer for a conversation

            In the code cell below we:

            - generate a reply from a dialogue prompt using GPT-style decoding
            - summarize a conversation with a T5 model
            - compare decoding settings for the chatbot output
            """
        ),
        code(
            """
            dialogue_model.eval()

            chat_prompt = "Speaker 1: Are you still coming to dinner? <eos> Speaker 2:"
            chat_inputs = gpt_tokenizer(chat_prompt, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                greedy_reply_ids = dialogue_model.generate(
                    **chat_inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=gpt_tokenizer.eos_token_id,
                )
                sampled_reply_ids = dialogue_model.generate(
                    **chat_inputs,
                    max_new_tokens=40,
                    do_sample=True,
                    top_k=40,
                    top_p=0.95,
                    temperature=0.9,
                    pad_token_id=gpt_tokenizer.eos_token_id,
                )

            greedy_reply = gpt_tokenizer.decode(greedy_reply_ids[0], skip_special_tokens=True)
            sampled_reply = gpt_tokenizer.decode(sampled_reply_ids[0], skip_special_tokens=True)

            summarizer_source = CHECKPOINT_DIR / "t5_samsum_demo"
            summarizer_checkpoint = str(summarizer_source if summarizer_source.exists() else T5_FALLBACK_CHECKPOINT)
            summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_checkpoint)
            summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_checkpoint).to(DEVICE)
            summarizer_model.eval()

            conversation_for_summary = (
                "Ava: We moved the project meeting to 3 PM.\\n"
                "Liam: That works for me.\\n"
                "Ava: Please bring the updated slides.\\n"
                "Liam: I will also send the draft before lunch."
            )

            summary_inputs = summarizer_tokenizer(
                "summarize: " + conversation_for_summary,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(DEVICE)

            with torch.no_grad():
                summary_ids = summarizer_model.generate(
                    **summary_inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )

            summary_text = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            application_df = pd.DataFrame(
                [
                    ("Chatbot reply - greedy", greedy_reply),
                    ("Chatbot reply - sampled", sampled_reply),
                    ("Conversation summary", summary_text),
                ],
                columns=["application", "output"],
            )
            display(application_df)
            """
        ),
        md(
            """
            ## Wrap-Up

            Across the three notebooks, students have now seen the whole story:

            - how a decoder-only transformer is built from scratch
            - why attention is necessary for seq2seq tasks
            - how encoder-decoder models use cross-attention as memory retrieval
            - how pretrained transformers are fine-tuned for classification, summarization, and dialogue

            That covers the conceptual pipeline from raw text and attention math all the way to modern Hugging Face applications.
            """
        ),
    ]
    return notebook("Fine-Tuning and Applications With Hugging Face", cells)


def write_notebook(filename: str, content):
    path = NOTEBOOK_DIR / filename
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def main():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook("01_mini_transformer_from_scratch.ipynb", notebook_one())
    write_notebook("02_attention_encoder_decoder_seq2seq.ipynb", notebook_two())
    write_notebook("03_finetuning_and_applications_with_hf.ipynb", notebook_three())


if __name__ == "__main__":
    main()
