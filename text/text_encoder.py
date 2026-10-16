import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Adds information about the position of each token in the sequence.

    Input:
        x: Tensor of shape [batch_size, sequence_length, hidden_size]

    Output:
        Tensor of shape [batch_size, sequence_length, hidden_size]
    """

    def __init__(self, hidden_size, max_length=128):
        super().__init__()

        # Create a tensor containing the position of each token.
        position = torch.arange(max_length).unsqueeze(1)

        # Compute the frequencies used by the sinusoidal functions.
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2)
            * (-math.log(10000.0) / hidden_size)
        )

        # Store the positional encoding as a non-trainable buffer.
        positional_encoding = torch.zeros(
            max_length,
            hidden_size
        )

        positional_encoding[:, 0::2] = torch.sin(
            position * div_term
        )

        positional_encoding[:, 1::2] = torch.cos(
            position * div_term[:hidden_size // 2]
        )

        # Add batch dimension:
        # [max_length, hidden_size]
        #        ↓
        # [1, max_length, hidden_size]
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer(
            "positional_encoding",
            positional_encoding
        )

    def forward(self, x):
        """
        Add positional information to the input embeddings.

        Args:
            x: Tensor of shape [B, L, D]

        Returns:
            Tensor of shape [B, L, D]
        """

        sequence_length = x.size(1)

        if sequence_length > self.positional_encoding.size(1):
            raise ValueError(
                "Sequence length is larger than the maximum "
                "supported positional encoding length."
            )

        return x + self.positional_encoding[:, :sequence_length, :]

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder block.

    Structure:

        Multi-Head Self-Attention
        -> residual connection + LayerNorm
        -> Feed Forward Network
        -> residual connection + LayerNorm

    Input:
        x: Tensor of shape [B, L, D]

    Output:
        Tensor of shape [B, L, D]

    where:
        B = batch size
        L = sequence length
        D = hidden size
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        feed_forward_size=None,
        dropout=0.0
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads."
            )

        # If no FFN size is specified, use the standard
        # 4x expansion commonly used in Transformer blocks.
        if feed_forward_size is None:
            feed_forward_size = 4 * hidden_size

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(hidden_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, feed_forward_size),
            nn.ReLU(),
            nn.Linear(feed_forward_size, hidden_size)
        )

        self.feed_forward_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x:
                Token representations of shape [B, L, D].

            attention_mask:
                Tensor of shape [B, L].
                1 = valid token
                0 = padding token

        Returns:
            Tensor of shape [B, L, D].
        """

        # PyTorch MultiheadAttention expects True for positions
        # that must be ignored.
        key_padding_mask = None

        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        # ---------------------------------------------------------
        # 1. Multi-Head Self-Attention
        # ---------------------------------------------------------

        attention_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        # Residual connection + LayerNorm
        x = self.attention_norm(
            x + self.dropout(attention_output)
        )

        # ---------------------------------------------------------
        # 2. Feed Forward Network
        # ---------------------------------------------------------

        feed_forward_output = self.feed_forward(x)

        # Residual connection + LayerNorm
        x = self.feed_forward_norm(
            x + self.dropout(feed_forward_output)
        )

        return x

class TextEncoder(nn.Module):
    """
    Transformer-based text encoder.

    Converts token IDs into contextualized token representations.

    Input:
        input_ids:      [B, L]
        attention_mask: [B, L]

    Output:
        text_features:  [B, L, D]

    where:
        B = batch size
        L = sequence length
        D = hidden size
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        max_length=128,
        feed_forward_size=None,
        dropout=0.0,
        pad_token_id=0
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than zero."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        # ---------------------------------------------------------
        # Token embedding
        # ---------------------------------------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=pad_token_id
        )

        # ---------------------------------------------------------
        # Positional encoding
        # ---------------------------------------------------------

        self.positional_encoding = PositionalEncoding(
            hidden_size=hidden_size,
            max_length=max_length
        )

        # ---------------------------------------------------------
        # Transformer Encoder
        # ---------------------------------------------------------

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    feed_forward_size=feed_forward_size,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input_ids, attention_mask=None):
        """
        Encode a batch of tokenized captions.

        Args:
            input_ids:
                Tensor of token IDs with shape [B, L].

            attention_mask:
                Tensor with shape [B, L].
                1 = valid token
                0 = padding token.

                If None, the mask is generated automatically
                using pad_token_id.

        Returns:
            Contextualized token representations
            with shape [B, L, D].
        """

        if input_ids.dim() != 2:
            raise ValueError(
                "input_ids must have shape [B, L]."
            )

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_length:
            raise ValueError(
                "Sequence length is larger than max_length."
            )

        # ---------------------------------------------------------
        # Attention mask
        # ---------------------------------------------------------

        if attention_mask is None:
            attention_mask = (
                input_ids != self.pad_token_id
            ).long()

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape "
                "as input_ids."
            )

        # ---------------------------------------------------------
        # Token embedding
        # ---------------------------------------------------------

        x = self.embedding(input_ids)

        # ---------------------------------------------------------
        # Positional encoding
        # ---------------------------------------------------------

        x = self.positional_encoding(x)

        # ---------------------------------------------------------
        # Transformer blocks
        # ---------------------------------------------------------

        for layer in self.layers:
            x = layer(
                x,
                attention_mask=attention_mask
            )

        return x