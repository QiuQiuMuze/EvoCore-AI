import torch
import os

class TextEnvironment:
    """
    A simple text-based environment for token prediction tasks.
    You can initialize with:
      - a single text string,
      - a path to a text file (.txt),
      - or a path to a folder containing .txt files.
    Splits input text into whitespace tokens and rewards correct next-token predictions.
    """

    def __init__(self, source):
        # Determine source type and load text
        if os.path.isdir(source):
            texts = []
            for fname in sorted(os.listdir(source)):
                if fname.lower().endswith('.txt'):
                    path = os.path.join(source, fname)
                    with open(path, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
            text = "\n".join(texts)
        elif os.path.isfile(source):
            with open(source, 'r', encoding='utf-8') as f:
                text = f.read()
        elif isinstance(source, str):
            text = source
        else:
            raise ValueError(f"Unsupported source type: {source}")

        # Tokenize by whitespace
        self.tokens = text.split()
        # Build vocabulary mapping token -> index
        unique_tokens = list(dict.fromkeys(self.tokens))
        self.vocab = {tok: idx for idx, tok in enumerate(unique_tokens)}
        self.inverse_vocab = {idx: tok for tok, idx in self.vocab.items()}
        self.reset()

    def reset(self):
        # Start before first token
        self.index = 0
        return self._get_state()

    def _get_state(self):
        # One-hot vector of current token
        vec = torch.zeros(len(self.vocab), dtype=torch.float32)
        tok = self.tokens[self.index]
        vec[self.vocab[tok]] = 1.0
        return vec.unsqueeze(0)  # shape [1, vocab_size]

    def step(self, action: int):
        """
        Takes an action (predicted next-token index), returns:
          next_state (1 x vocab_size tensor) or None if done,
          reward (float),
          done (bool)
        Reward is +1.0 if action equals correct next token index, else -0.5.
        """
        # Determine correct index for the next token
        if self.index + 1 < len(self.tokens):
            correct_idx = self.vocab[self.tokens[self.index + 1]]
        else:
            correct_idx = None

        # Compute reward
        if correct_idx is not None and action == correct_idx:
            reward = 0.2
        else:
            reward = -0.1

        # Advance pointer
        self.index += 1
        done = self.index >= len(self.tokens) - 1

        # Prepare next state
        next_state = self._get_state() if not done else None
        return next_state, reward, done

    def vocab_size(self):
        return len(self.vocab)
