
import json
import sys
from model import build_vocab

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_vocab.py <path_to_data>")
        sys.exit(1)

    data_path = sys.argv[1]
    vocab_size, char_to_idx, _ = build_vocab(data_path)

    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump({
            "vocab_size": vocab_size,
            "char_to_idx": char_to_idx
        }, f)

    print(f"Vocabulary built with size {vocab_size} and saved to vocab.json")
