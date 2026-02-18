import argparse
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle


class NetworkVisualizer24Word:
    def __init__(self, weights_file):
        data = np.load(weights_file)
        self.I = data["i_weights"]
        self.J = data["j_weights"]
        self.L4 = data["l4_biases"]
        self.L5 = data["l5_biases"]

        self.categories = [
            "BACK",
            "DARK",
            "EACH",
            "FROM",
            "JUST",
            "BEEN",
            "GOOD",
            "MUCH",
            "SOME",
            "TIME",
            "LIKE",
            "ONLY",
            "WORK",
            "WAVE",
            "ZERO",
            "ZONE",
            "VAIN",
            "VAST",
            "QUIZ",
            "HELP",
            "FIND",
            "PLUS",
            "YAWN",
            "STOP",
        ]
        self.binary_patterns = self._create_binary_patterns()

    def _create_binary_patterns(self):
        patterns = {}
        for i in range(len(self.categories)):
            patterns[i] = np.array([(i >> bit) & 1 for bit in range(12)], dtype=float)
        return patterns

    def encode_word(self, word):
        values = np.zeros(30)
        word = word.upper()
        unique_letters = sorted(set(c for c in word if "A" <= c <= "Z"))

        for i, char in enumerate(unique_letters[:4]):
            values[i] = (ord(char) - ord("A") + 1) / 26.0

        for char in unique_letters:
            values[4 + ord(char) - ord("A")] = 1.0
        return values

    def forward(self, word):
        x = self.encode_word(word)
        hidden = 1 / (1 + np.exp(-(np.dot(self.I, x) + self.L4)))
        outputs = 1 / (1 + np.exp(-(np.dot(self.J, hidden) + self.L5)))
        return outputs

    def decode(self, outputs):
        rounded = np.round(np.clip(outputs, 0, 1))
        best_idx = 0
        min_hamming = 1e9
        for idx, pattern in self.binary_patterns.items():
            hamming = int(np.sum(np.abs(rounded - pattern)))
            if hamming < min_hamming:
                min_hamming = hamming
                best_idx = idx
        confidence = int(((12 - min_hamming) / 12) * 100)
        return best_idx, confidence, rounded, min_hamming

    def predict(self, word):
        outputs = self.forward(word)
        idx, confidence, rounded, hamming = self.decode(outputs)
        return self.categories[idx], confidence, outputs, rounded, hamming

    def _scramble(self, word):
        letters = list(word.upper())
        random.shuffle(letters)
        return "".join(letters)

    def _adjacent_typo(self, word):
        word = word.upper()
        if not word:
            return word
        idx = random.randint(0, len(word) - 1)
        char = word[idx]
        if char == "A":
            repl = "B"
        elif char == "Z":
            repl = "Y"
        else:
            repl = chr(ord(char) + random.choice([-1, 1]))
        return word[:idx] + repl + word[idx + 1 :]

    def _drop_letter(self, word):
        word = word.upper()
        if len(word) <= 1:
            return word
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx + 1 :]

    def _multi_adjacent(self, word):
        letters = list(word.upper())
        if len(letters) < 2:
            return "".join(letters)
        count = random.randint(2, min(4, len(letters)))
        for pos in random.sample(range(len(letters)), count):
            char = letters[pos]
            if char == "A":
                letters[pos] = "B"
            elif char == "Z":
                letters[pos] = "Y"
            else:
                letters[pos] = chr(ord(char) + random.choice([-1, 1]))
        return "".join(letters)

    def visualize_encoding_comparison(self, word1, word2):
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.28, top=0.92, bottom=0.08)
        fig.suptitle(
            f"V3 Encoding Comparison: {word1.upper()} vs {word2.upper()}",
            fontsize=18,
            fontweight="bold",
            y=0.96,
        )

        row1 = [(word1, "Original"), (self._scramble(word1), "Scrambled"), (self._multi_adjacent(word1), "Multi-Adjacent")]
        row2 = [(word2, "Original"), (self._scramble(word2), "Scrambled"), (self._multi_adjacent(word2), "Multi-Adjacent")]

        for col, (variant, label) in enumerate(row1):
            ax = fig.add_subplot(gs[0, col])
            self._draw_encoding_cell(ax, variant, label, word1)
        for col, (variant, label) in enumerate(row2):
            ax = fig.add_subplot(gs[1, col])
            self._draw_encoding_cell(ax, variant, label, word2)

        legend_ax = fig.add_axes([0.16, 0.02, 0.68, 0.04])
        legend_ax.axis("off")
        red_patch = mpatches.Patch(color="#FF6B6B", label="Sorted Positional (4)")
        cyan_patch = mpatches.Patch(color="#4ECDC4", label="Binary Presence (A-Z)")
        legend_ax.legend(handles=[red_patch, cyan_patch], loc="center", ncol=2, fontsize=11, frameon=True)

        plt.savefig("encoding_comparison_24word.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _draw_encoding_cell(self, ax, word, label, expected_word):
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 4.5)
        ax.axis("off")

        encoding = self.encode_word(word)
        pred_word, confidence, _, _, hamming = self.predict(word)
        is_correct = pred_word == expected_word.upper()

        ax.text(16, 4.1, f"{word.upper()}", fontsize=13, fontweight="bold", ha="center", family="monospace")
        ax.text(16, 3.8, label, fontsize=10, style="italic", ha="center", color="#666")

        for i in range(4):
            x = 1 + i * 2.2
            color = "#FF6B6B" if encoding[i] > 0 else "#FFE5E5"
            rect = Rectangle((x, 2.2), 1.8, 1, facecolor=color, edgecolor="#CC0000", linewidth=2)
            ax.add_patch(rect)
            if encoding[i] > 0:
                ax.text(x + 0.9, 2.7, f"{encoding[i]:.2f}", fontsize=8, ha="center", va="center", fontweight="bold", color="#333")
        ax.text(0.5, 2.7, "Pos:", fontsize=9, ha="right", va="center", fontweight="bold")

        for i in range(26):
            col = i % 13
            row = 0 if i < 13 else 1
            x = 11 + col * 1.5
            y = 1.8 - row * 1.1
            color = "#4ECDC4" if encoding[4 + i] > 0 else "#E5F9F7"
            rect = Rectangle((x, y), 1.3, 0.95, facecolor=color, edgecolor="#009999", linewidth=1)
            ax.add_patch(rect)
            if encoding[4 + i] > 0:
                ax.text(x + 0.65, y + 0.475, chr(ord("A") + i), fontsize=8, ha="center", va="center", fontweight="bold", color="#003333")
        ax.text(10.5, 1.35, "Letters:", fontsize=9, ha="right", va="center", fontweight="bold")

        color = "#C8E6C9" if is_correct else "#FFCDD2"
        border_color = "#2E7D32" if is_correct else "#C62828"
        pred_box = FancyBboxPatch((0.5, 0.1), 30, 0.7, boxstyle="round,pad=0.05", facecolor=color, edgecolor=border_color, linewidth=2)
        ax.add_patch(pred_box)
        ax.text(16, 0.45, f"-> {pred_word} ({confidence}%, d={hamming})", fontsize=11, ha="center", va="center", fontweight="bold")

    def visualize_encoding_heatmap(self, word1, word2):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(f"V3 Encoding Heatmap: {word1.upper()} vs {word2.upper()}", fontsize=16, fontweight="bold", y=0.98)

        row1 = [(word1, "Original"), (self._scramble(word1), "Scrambled"), (self._multi_adjacent(word1), "Multi-Adjacent")]
        row2 = [(word2, "Original"), (self._scramble(word2), "Scrambled"), (self._multi_adjacent(word2), "Multi-Adjacent")]

        for col, (variant, label) in enumerate(row1):
            self._draw_heatmap_cell(axes[0, col], variant, label)
        for col, (variant, label) in enumerate(row2):
            self._draw_heatmap_cell(axes[1, col], variant, label)

        plt.tight_layout()
        plt.savefig("encoding_heatmap_24word.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _draw_heatmap_cell(self, ax, word, label):
        encoding = self.encode_word(word)
        row0 = np.concatenate([encoding[:4], np.zeros(22)])
        row1 = encoding[4:30]
        data = np.array([row0, row1])
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"{word.upper()}\n{label}", fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(range(26))
        ax.set_xticklabels([chr(ord("A") + i) for i in range(26)], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Sorted (4)", "Binary (26)"], fontsize=9, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
        cbar.set_label("Value", fontsize=8)

    def visualize_prediction_comparison(self, word1, word2):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"V3 Prediction Comparison: {word1.upper()} vs {word2.upper()}", fontsize=15, fontweight="bold", y=0.98)

        for word_idx, base_word in enumerate([word1, word2]):
            ax = axes[word_idx]
            variants = [
                (base_word, "Original"),
                (self._scramble(base_word), "Scrambled"),
                (self._adjacent_typo(base_word), "Adjacent"),
                (self._drop_letter(base_word), "Dropped"),
                (self._multi_adjacent(base_word), "Multi-Adj"),
            ]
            predictions = []
            confidences = []
            labels = []

            for variant_word, label in variants:
                pred_word, confidence, _, _, _ = self.predict(variant_word)
                predictions.append(pred_word)
                confidences.append(confidence)
                labels.append(f"{variant_word[:4]}\n({label})")

            colors = ["#FF6B6B" if pred == base_word.upper() else "#FFD93D" for pred in predictions]
            bars = ax.bar(range(len(labels)), confidences, color=colors, edgecolor="black", linewidth=2, alpha=0.7, width=0.65)
            ax.set_ylim(0, 105)
            ax.set_ylabel("Bit-Match Confidence (%)", fontsize=11, fontweight="bold")
            ax.set_title(f"{base_word.upper()}", fontsize=12, fontweight="bold", pad=15)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)
            ax.grid(alpha=0.3, axis="y", linestyle="--")
            ax.set_axisbelow(True)

            for bar, conf, pred in zip(bars, confidences, predictions):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{conf}%\n{pred}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig("prediction_comparison_24word.png", dpi=150, bbox_inches="tight")
        plt.close()

    def visualize_codebook_heatmap(self):
        fig, ax = plt.subplots(figsize=(14, 8))
        matrix = np.array([self.binary_patterns[i] for i in range(len(self.categories))], dtype=float)
        im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_title("V3 12-Bit Codebook for 24 Words", fontsize=14, fontweight="bold")
        ax.set_xlabel("Output Bit Index (0-11)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Word Category (24 total)", fontsize=11, fontweight="bold")
        ax.set_xticks(range(12))
        ax.set_xticklabels([f"B{i}" for i in range(12)], fontsize=9)
        ax.set_yticks(range(len(self.categories)))
        ax.set_yticklabels(self.categories, fontsize=8)
        plt.colorbar(im, ax=ax, label="Bit Value")
        plt.tight_layout()
        plt.savefig("binary_codebook_24word.png", dpi=150, bbox_inches="tight")
        plt.close()

    def visualize_weight_distribution(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("V3 Weight Distribution Analysis (24-Word Binary)", fontsize=14, fontweight="bold")

        ax = axes[0, 0]
        ax.hist(self.I.flatten(), bins=50, color="#FF6B6B", alpha=0.7, edgecolor="black")
        ax.set_title("Input->Hidden Weights [I] (50x30)", fontweight="bold")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.hist(self.J.flatten(), bins=50, color="#4ECDC4", alpha=0.7, edgecolor="black")
        ax.set_title("Hidden->Output Weights [J] (12x50)", fontweight="bold")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.bar(range(50), self.L4, color="#95E1D3", alpha=0.7, edgecolor="black")
        ax.set_title("Hidden Biases [L4] (50)", fontweight="bold")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Bias Value")
        ax.grid(alpha=0.3, axis="y")

        ax = axes[1, 1]
        ax.bar(range(12), self.L5, color=plt.cm.Set3(np.linspace(0, 1, 12)), alpha=0.8, edgecolor="black")
        ax.set_title("Output Bit Biases [L5] (12 bits)", fontweight="bold")
        ax.set_xlabel("Bit Index")
        ax.set_ylabel("Bias Value")
        ax.set_xticks(range(12))
        ax.set_xticklabels([f"B{i}" for i in range(12)], fontsize=9)
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("weight_distribution_24word.png", dpi=150, bbox_inches="tight")
        plt.close()

    def visualize_weight_heatmaps(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("V3 Weight Matrix Heatmaps (24-Word Binary)", fontsize=14, fontweight="bold")

        ax = axes[0]
        im = ax.imshow(self.I, aspect="auto", cmap="RdBu_r", interpolation="nearest")
        ax.set_title("Input->Hidden [I] (50x30)", fontweight="bold")
        ax.set_xlabel("Input Neuron (30)")
        ax.set_ylabel("Hidden Neuron (50)")
        plt.colorbar(im, ax=ax, label="Weight Value")

        ax = axes[1]
        im = ax.imshow(self.J, aspect="auto", cmap="RdBu_r", interpolation="nearest")
        ax.set_title("Hidden->Output [J] (12x50)", fontweight="bold")
        ax.set_xlabel("Hidden Neuron (50)")
        ax.set_yticks(range(12))
        ax.set_yticklabels([f"B{i}" for i in range(12)], fontsize=9)
        plt.colorbar(im, ax=ax, label="Weight Value")

        plt.tight_layout()
        plt.savefig("weight_heatmaps_24word.png", dpi=150, bbox_inches="tight")
        plt.close()

    def visualize_architecture(self):
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        ax.text(5, 9.5, "HERMES V3 - 30-50-12 with 24-Word Binary Codebook", fontsize=16, fontweight="bold", ha="center")

        input_y = 7.5
        ax.text(1.5, input_y + 1, "INPUT LAYER", fontsize=12, fontweight="bold", color="#333")
        sorted_box = FancyBboxPatch((0.3, input_y - 0.6), 1.3, 1, boxstyle="round,pad=0.1", edgecolor="#FF6B6B", facecolor="#FFE5E5", linewidth=2.5)
        binary_box = FancyBboxPatch((1.8, input_y - 0.6), 1.3, 1, boxstyle="round,pad=0.1", edgecolor="#4ECDC4", facecolor="#E5F9F7", linewidth=2.5)
        ax.add_patch(sorted_box)
        ax.add_patch(binary_box)
        ax.text(0.95, input_y + 0.05, "Sorted\nPositional", fontsize=9, ha="center", va="center", fontweight="bold")
        ax.text(0.95, input_y - 0.5, "(4)", fontsize=8, ha="center", style="italic")
        ax.text(2.45, input_y + 0.05, "Binary\nPresence", fontsize=9, ha="center", va="center", fontweight="bold")
        ax.text(2.45, input_y - 0.5, "(26)", fontsize=8, ha="center", style="italic")

        hidden_y = 5
        ax.text(5, hidden_y + 1.2, "HIDDEN LAYER", fontsize=12, fontweight="bold", color="#333")
        hidden_box = FancyBboxPatch((4.6, hidden_y - 0.4), 0.8, 0.8, boxstyle="round,pad=0.1", edgecolor="#95E1D3", facecolor="#E8F8F5", linewidth=2.5)
        ax.add_patch(hidden_box)
        ax.text(5, hidden_y + 0.15, "50", fontsize=11, ha="center", va="center", fontweight="bold")
        ax.text(5, hidden_y - 0.3, "neurons", fontsize=8, ha="center", style="italic")

        output_y = 2.5
        ax.text(8.5, output_y + 1.2, "OUTPUT LAYER", fontsize=12, fontweight="bold", color="#333")
        output_box = FancyBboxPatch((8.1, output_y - 0.4), 0.8, 0.8, boxstyle="round,pad=0.1", edgecolor="#F38181", facecolor="#FFE5E5", linewidth=2.5)
        ax.add_patch(output_box)
        ax.text(8.5, output_y + 0.15, "12", fontsize=11, ha="center", va="center", fontweight="bold")
        ax.text(8.5, output_y - 0.35, "bits", fontsize=8, ha="center", style="italic")

        ax.arrow(2.2, input_y, 2.3, hidden_y - input_y + 0.2, head_width=0.12, head_length=0.15, fc="#999", ec="#999", alpha=0.6, linewidth=2)
        ax.arrow(5.4, hidden_y - 0.4, 2.8, output_y - hidden_y + 0.2, head_width=0.12, head_length=0.15, fc="#999", ec="#999", alpha=0.6, linewidth=2)
        ax.text(3.3, (input_y + hidden_y) / 2 + 0.4, "1,500 weights", fontsize=9, ha="center", fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, pad=0.3))
        ax.text(6.8, (hidden_y + output_y) / 2, "600 weights", fontsize=9, ha="center", fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, pad=0.3))
        ax.text(
            5,
            0.5,
            "12 output bits map to 24 word IDs via nearest binary pattern (Hamming decode).",
            fontsize=10,
            ha="center",
            style="italic",
            bbox=dict(boxstyle="round", facecolor="#FFFACD", alpha=0.9, pad=0.5, linewidth=1.5),
        )

        plt.tight_layout()
        plt.savefig("network_architecture_24word.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="V3 visualization for 24-word binary model")
    parser.add_argument("weights_file", type=str, help="Path to .npz weights file")
    parser.add_argument("--words", type=str, nargs=2, default=["JUST", "WORK"], help="Two words to compare")
    args = parser.parse_args()

    viz = NetworkVisualizer24Word(args.weights_file)
    viz.visualize_encoding_comparison(args.words[0], args.words[1])
    viz.visualize_encoding_heatmap(args.words[0], args.words[1])
    viz.visualize_architecture()
    viz.visualize_prediction_comparison(args.words[0], args.words[1])
    viz.visualize_codebook_heatmap()
    viz.visualize_weight_distribution()
    viz.visualize_weight_heatmaps()
    print("Generated 7 V3 visualization files for 24-word model.")


if __name__ == "__main__":
    main()
