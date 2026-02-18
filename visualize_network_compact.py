import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import argparse

class NetworkVisualizer:
    def __init__(self, weights_file):
        """Load weights from npz file"""
        try:
            data = np.load(weights_file)
            self.I = data['i_weights']
            self.J = data['j_weights']
            self.L4 = data['l4_biases']
            self.L5 = data['l5_biases']
            print(f"Loaded weights from {weights_file}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise

    def encode_word(self, word):
        """Encode a word using hybrid encoding"""
        input_values = np.zeros(30)
        word = word.upper()
        unique_letters = sorted(set(c for c in word if 'A' <= c <= 'Z'))

        for i, char in enumerate(unique_letters[:4]):
            input_values[i] = (ord(char) - ord('A') + 1) / 26.0

        for char in unique_letters:
            char_index = ord(char) - ord('A')
            input_values[4 + char_index] = 1.0

        return input_values

    def predict(self, word):
        """Get prediction for a word"""
        categories = ['BACK', 'DARK', 'EACH', 'FROM', 'JUST', 'BEEN',
                     'GOOD', 'MUCH', 'SOME', 'TIME', 'LIKE', 'ONLY']
        encoding = self.encode_word(word)
        hidden = 1 / (1 + np.exp(-(np.dot(self.I, encoding) + self.L4)))
        outputs = 1 / (1 + np.exp(-(np.dot(self.J, hidden) + self.L5)))
        predicted_idx = np.argmax(outputs)
        confidence = int((outputs[predicted_idx] / np.sum(outputs)) * 100)
        return categories[predicted_idx], confidence

    def visualize_encoding_clean(self, word1, word2):
        """Clean, large encoding visualization for 2 words"""
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.28, top=0.92, bottom=0.08)

        fig.suptitle(f'Encoding Comparison: {word1.upper()} vs {word2.upper()}',
                    fontsize=18, fontweight='bold', y=0.96)

        variations = [
            (word1, 'Original'),
            (self._scramble(word1), 'Scrambled'),
            (self._multi_adjacent(word1), 'Multi-Adjacent'),
        ]

        # Word 1 row
        for col, (variant, label) in enumerate(variations):
            ax = fig.add_subplot(gs[0, col])
            self._draw_encoding_cell(ax, variant, label, word1)

        # Word 2 row
        variations2 = [
            (word2, 'Original'),
            (self._scramble(word2), 'Scrambled'),
            (self._multi_adjacent(word2), 'Multi-Adjacent'),
        ]

        for col, (variant, label) in enumerate(variations2):
            ax = fig.add_subplot(gs[1, col])
            self._draw_encoding_cell(ax, variant, label, word2)

        # Legend
        legend_ax = fig.add_axes([0.2, 0.02, 0.6, 0.04])
        legend_ax.axis('off')
        red_patch = mpatches.Patch(color='#FF6B6B', label='Sorted Positional (4 values)')
        cyan_patch = mpatches.Patch(color='#4ECDC4', label='Binary Presence (A-Z)')
        legend_ax.legend(handles=[red_patch, cyan_patch], loc='center', ncol=2, fontsize=11, frameon=True)

        plt.savefig('encoding_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved: encoding_comparison.png")
        plt.close()

    def _draw_encoding_cell(self, ax, word, label, expected_word):
        """Draw a single encoding visualization cell"""
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 4.5)
        ax.axis('off')

        encoding = self.encode_word(word)
        pred_word, confidence = self.predict(word)
        is_correct = pred_word == expected_word.upper()

        # Title with word
        ax.text(16, 4.1, f'{word.upper()}', fontsize=13, fontweight='bold', ha='center', family='monospace')
        ax.text(16, 3.8, label, fontsize=10, style='italic', ha='center', color='#666')

        # Sorted positional (red) - larger, better spaced
        for i in range(4):
            x = 1 + i * 2.2
            color = '#FF6B6B' if encoding[i] > 0 else '#FFE5E5'
            rect = Rectangle((x, 2.2), 1.8, 1, facecolor=color, edgecolor='#CC0000', linewidth=2)
            ax.add_patch(rect)
            if encoding[i] > 0:
                ax.text(x + 0.9, 2.7, f'{encoding[i]:.2f}', fontsize=8, ha='center', va='center',
                       fontweight='bold', color='#333')

        ax.text(0.5, 2.7, 'Pos:', fontsize=9, ha='right', va='center', fontweight='bold')

        # Binary presence (cyan) - in grid, better spaced
        for i in range(26):
            col = i % 13
            row = 0 if i < 13 else 1
            x = 11 + col * 1.5
            y = 1.8 - row * 1.1

            color = '#4ECDC4' if encoding[4 + i] > 0 else '#E5F9F7'
            rect = Rectangle((x, y), 1.3, 0.95, facecolor=color, edgecolor='#009999', linewidth=1)
            ax.add_patch(rect)

            if encoding[4 + i] > 0:
                ax.text(x + 0.65, y + 0.475, chr(ord('A') + i), fontsize=8,
                       ha='center', va='center', fontweight='bold', color='#003333')

        ax.text(10.5, 1.35, 'Letters:', fontsize=9, ha='right', va='center', fontweight='bold')

        # Prediction box
        color = '#C8E6C9' if is_correct else '#FFCDD2'
        border_color = '#2E7D32' if is_correct else '#C62828'
        pred_box = FancyBboxPatch((0.5, 0.1), 30, 0.7, boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor=border_color, linewidth=2)
        ax.add_patch(pred_box)
        ax.text(16, 0.45, f'→ {pred_word} ({confidence}%)', fontsize=11,
               ha='center', va='center', fontweight='bold')

    def visualize_encoding_heatmap_clean(self, word1, word2):
        """Clean heatmap comparison with reduced overlap"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(f'Encoding Heatmap: {word1.upper()} vs {word2.upper()}',
                    fontsize=16, fontweight='bold', y=0.98)

        variations = [
            (word1, 'Original'),
            (self._scramble(word1), 'Scrambled'),
            (self._multi_adjacent(word1), 'Multi-Adjacent'),
        ]

        # Word 1
        for col, (variant, label) in enumerate(variations):
            ax = axes[0, col]
            self._draw_heatmap_cell(ax, variant, label, is_top=True, word_label=word1.upper())

        # Word 2
        variations2 = [
            (word2, 'Original'),
            (self._scramble(word2), 'Scrambled'),
            (self._multi_adjacent(word2), 'Multi-Adjacent'),
        ]

        for col, (variant, label) in enumerate(variations2):
            ax = axes[1, col]
            self._draw_heatmap_cell(ax, variant, label, is_top=False, word_label=word2.upper())

        plt.tight_layout()
        plt.savefig('encoding_heatmap.png', dpi=150, bbox_inches='tight')
        print("Saved: encoding_heatmap.png")
        plt.close()

    def _draw_heatmap_cell(self, ax, word, label, is_top, word_label):
        """Draw a single heatmap cell"""
        encoding = self.encode_word(word)

        # Create visualization data
        row0 = np.concatenate([encoding[:4], np.zeros(22)])
        row1 = encoding[4:30]
        encoding_2d = np.array([row0, row1])

        im = ax.imshow(encoding_2d, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Labels
        ax.set_title(f'{word.upper()}\n{label}', fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks(range(26))
        ax.set_xticklabels([chr(ord('A') + i) for i in range(26)], fontsize=8, rotation=0)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Sorted (4)', 'Binary (26)'], fontsize=9, fontweight='bold')

        if is_top:
            ax.set_ylabel(f'{word_label}', fontsize=10, fontweight='bold', labelpad=10)

        # Colorbar with reduced overlap
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
        cbar.set_label('Value', fontsize=8)

    def visualize_prediction_comparison(self, word1, word2):
        """Original style prediction comparison with actual word predictions"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Prediction Comparison: {word1.upper()} vs {word2.upper()}',
                    fontsize=15, fontweight='bold', y=0.98)

        categories = ['BACK', 'DARK', 'EACH', 'FROM', 'JUST', 'BEEN',
                     'GOOD', 'MUCH', 'SOME', 'TIME', 'LIKE', 'ONLY']

        variations_list = [
            (f'{word1}', 'Original'),
            (self._scramble(word1), 'Scrambled'),
            (self._adjacent_typo(word1), 'Adjacent'),
            (self._drop_letter(word1), 'Dropped'),
            (self._multi_adjacent(word1), 'Multi-Adj'),
        ]

        for word_idx, base_word in enumerate([word1, word2]):
            ax = axes[word_idx]

            if word_idx == 1:
                variations_list = [
                    (f'{word2}', 'Original'),
                    (self._scramble(word2), 'Scrambled'),
                    (self._adjacent_typo(word2), 'Adjacent'),
                    (self._drop_letter(word2), 'Dropped'),
                    (self._multi_adjacent(word2), 'Multi-Adj'),
                ]

            predictions = []
            confidences = []
            labels = []

            for variant_word, label in variations_list:
                pred_word, confidence = self.predict(variant_word)
                predictions.append(pred_word)
                confidences.append(confidence)
                labels.append(f'{variant_word[:4]}\n({label})')

            colors = ['#FF6B6B' if pred == base_word.upper() else '#FFD93D'
                     for pred in predictions]

            bars = ax.bar(range(len(labels)), confidences, color=colors,
                         edgecolor='black', linewidth=2, alpha=0.7, width=0.65)

            ax.set_ylim(0, 105)
            ax.set_ylabel('Confidence (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'{base_word.upper()}', fontsize=12, fontweight='bold', pad=15)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)
            ax.grid(alpha=0.3, axis='y', linestyle='--')
            ax.set_axisbelow(True)

            # Add percentage and prediction labels on bars
            for bar, conf, pred in zip(bars, confidences, predictions):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{conf}%\n{pred}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig('prediction_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved: prediction_comparison.png")
        plt.close()

    def visualize_weight_distribution(self):
        """Visualize weight distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Weight Distribution Analysis', fontsize=14, fontweight='bold')

        # I weights distribution
        ax = axes[0, 0]
        ax.hist(self.I.flatten(), bins=50, color='#FF6B6B', alpha=0.7, edgecolor='black')
        ax.set_title('Input→Hidden Weights [I] (30×50=1500)', fontweight='bold')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
        ax.text(0.98, 0.97, f'μ={self.I.mean():.4f}\nσ={self.I.std():.4f}',
               transform=ax.transAxes, fontsize=9, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # J weights distribution
        ax = axes[0, 1]
        ax.hist(self.J.flatten(), bins=50, color='#4ECDC4', alpha=0.7, edgecolor='black')
        ax.set_title('Hidden→Output Weights [J] (50×12=600)', fontweight='bold')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
        ax.text(0.98, 0.97, f'μ={self.J.mean():.4f}\nσ={self.J.std():.4f}',
               transform=ax.transAxes, fontsize=9, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # L4 biases (hidden)
        ax = axes[1, 0]
        ax.bar(range(50), self.L4, color='#95E1D3', alpha=0.7, edgecolor='black')
        ax.set_title('Hidden Layer Biases [L4] (50 values)', fontweight='bold')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Bias Value')
        ax.grid(alpha=0.3, axis='y')

        # L5 biases (output)
        ax = axes[1, 1]
        categories = ['BACK', 'DARK', 'EACH', 'FROM', 'JUST', 'BEEN',
                     'GOOD', 'MUCH', 'SOME', 'TIME', 'LIKE', 'ONLY']
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        ax.bar(categories, self.L5, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Output Layer Biases [L5] (12 categories)', fontweight='bold')
        ax.set_ylabel('Bias Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('weight_distribution.png', dpi=150, bbox_inches='tight')
        print("Saved: weight_distribution.png")
        plt.close()

    def visualize_weight_heatmaps(self):
        """Visualize weight matrices as heatmaps"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Weight Matrix Heatmaps', fontsize=14, fontweight='bold')

        # I weights heatmap
        ax = axes[0]
        im = ax.imshow(self.I, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax.set_title('Input→Hidden Weights [I] (50×30)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Input Neuron (30 total)')
        ax.set_ylabel('Hidden Neuron (50 total)')
        for i in range(0, 50, 5):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for i in range(0, 30, 5):
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        plt.colorbar(im, ax=ax, label='Weight Value')

        # J weights heatmap
        ax = axes[1]
        im = ax.imshow(self.J, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax.set_title('Hidden→Output Weights [J] (12×50)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Hidden Neuron (50 total)')
        categories = ['BACK', 'DARK', 'EACH', 'FROM', 'JUST', 'BEEN',
                     'GOOD', 'MUCH', 'SOME', 'TIME', 'LIKE', 'ONLY']
        ax.set_yticks(range(12))
        ax.set_yticklabels(categories, fontsize=9)
        for i in range(0, 50, 5):
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        plt.colorbar(im, ax=ax, label='Weight Value')

        plt.tight_layout()
        plt.savefig('weight_heatmaps.png', dpi=150, bbox_inches='tight')
        print("Saved: weight_heatmaps.png")
        plt.close()

    def visualize_architecture(self):
        """Professional architecture diagram"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Title
        ax.text(5, 9.5, 'HERMES OPTIMUS - 30-50-12 Neural Network Architecture',
                fontsize=16, fontweight='bold', ha='center')

        # Input layer
        input_y = 7.5
        ax.text(1.5, input_y + 1, 'INPUT LAYER', fontsize=12, fontweight='bold', color='#333')

        sorted_box = FancyBboxPatch((0.3, input_y - 0.6), 1.3, 1,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='#FF6B6B', facecolor='#FFE5E5', linewidth=2.5)
        ax.add_patch(sorted_box)
        ax.text(0.95, input_y + 0.05, 'Sorted\nPositional', fontsize=9, ha='center', va='center', fontweight='bold')
        ax.text(0.95, input_y - 0.5, '(4)', fontsize=8, ha='center', style='italic')

        binary_box = FancyBboxPatch((1.8, input_y - 0.6), 1.3, 1,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='#4ECDC4', facecolor='#E5F9F7', linewidth=2.5)
        ax.add_patch(binary_box)
        ax.text(2.45, input_y + 0.05, 'Binary\nPresence', fontsize=9, ha='center', va='center', fontweight='bold')
        ax.text(2.45, input_y - 0.5, '(26)', fontsize=8, ha='center', style='italic')

        # Hidden layer
        hidden_y = 5
        ax.text(5, hidden_y + 1.2, 'HIDDEN LAYER', fontsize=12, fontweight='bold', color='#333')
        hidden_box = FancyBboxPatch((4.6, hidden_y - 0.4), 0.8, 0.8,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='#95E1D3', facecolor='#E8F8F5', linewidth=2.5)
        ax.add_patch(hidden_box)
        ax.text(5, hidden_y + 0.15, '50', fontsize=11, ha='center', va='center', fontweight='bold')
        ax.text(5, hidden_y - 0.3, 'neurons', fontsize=8, ha='center', style='italic')

        # Output layer
        output_y = 2.5
        ax.text(8.5, output_y + 1.2, 'OUTPUT LAYER', fontsize=12, fontweight='bold', color='#333')
        output_box = FancyBboxPatch((8.1, output_y - 0.4), 0.8, 0.8,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='#F38181', facecolor='#FFE5E5', linewidth=2.5)
        ax.add_patch(output_box)
        ax.text(8.5, output_y + 0.15, '12', fontsize=11, ha='center', va='center', fontweight='bold')
        ax.text(8.5, output_y - 0.3, 'words', fontsize=8, ha='center', style='italic')

        # Connections
        ax.arrow(2.2, input_y, 2.3, hidden_y - input_y + 0.2,
                head_width=0.12, head_length=0.15, fc='#999', ec='#999', alpha=0.6, linewidth=2)
        ax.text(3.3, (input_y + hidden_y) / 2 + 0.4, '1,500 weights',
                fontsize=9, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.3))

        ax.arrow(5.4, hidden_y - 0.4, 2.8, output_y - hidden_y + 0.2,
                head_width=0.12, head_length=0.15, fc='#999', ec='#999', alpha=0.6, linewidth=2)
        ax.text(6.8, (hidden_y + output_y) / 2, '600 weights',
                fontsize=9, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.3))

        # Legend
        info_text = "Hybrid Encoding combines letter proximity (sorted positional)\nwith exact letter identification (binary presence flags)"
        ax.text(5, 0.5, info_text, fontsize=10, ha='center', style='italic',
               bbox=dict(boxstyle='round', facecolor='#FFFACD', alpha=0.9, pad=0.5, linewidth=1.5))

        plt.tight_layout()
        plt.savefig('network_architecture.png', dpi=150, bbox_inches='tight')
        print("Saved: network_architecture.png")
        plt.close()

    def _scramble(self, word):
        """Scramble word letters"""
        import random
        letters = list(word.upper())
        random.shuffle(letters)
        return ''.join(letters)

    def _adjacent_typo(self, word):
        """Apply single adjacent-key typo"""
        import random
        word = word.upper()
        if len(word) == 0:
            return word
        idx = random.randint(0, len(word) - 1)
        char = word[idx]
        adj_char = chr(ord(char) + random.choice([-1, 1]))
        if 'A' <= adj_char <= 'Z':
            return word[:idx] + adj_char + word[idx+1:]
        return word

    def _drop_letter(self, word):
        """Drop one letter"""
        import random
        word = word.upper()
        if len(word) <= 1:
            return word
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx+1:]

    def _multi_adjacent(self, word):
        """Apply multi-letter adjacent substitution"""
        import random
        word = list(word.upper())
        num_to_replace = random.randint(2, min(4, len(word)))
        positions = random.sample(range(len(word)), num_to_replace)
        for pos in positions:
            char = word[pos]
            adj_char = chr(ord(char) + random.choice([-1, 1]))
            if 'A' <= adj_char <= 'Z':
                word[pos] = adj_char
        return ''.join(word)

def main():
    parser = argparse.ArgumentParser(description='Professional network visualization')
    parser.add_argument('weights_file', type=str, help='Path to .npz weights file')
    parser.add_argument('--words', type=str, nargs=2, default=['JUST', 'MUCH'],
                       help='Two words to compare (default: JUST MUCH)')
    args = parser.parse_args()

    print(f"Comparing: {args.words[0]} vs {args.words[1]}\n")

    viz = NetworkVisualizer(args.weights_file)

    print("Generating visualizations...")
    viz.visualize_encoding_clean(args.words[0], args.words[1])
    viz.visualize_encoding_heatmap_clean(args.words[0], args.words[1])
    viz.visualize_architecture()
    viz.visualize_prediction_comparison(args.words[0], args.words[1])
    viz.visualize_weight_distribution()
    viz.visualize_weight_heatmaps()

    print("\nComplete! Generated 6 visualizations.")

if __name__ == "__main__":
    main()
