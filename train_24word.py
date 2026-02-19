import numpy as np
import random
import argparse
import os

class TICalcNeuralNetwork:
    def __init__(self, learning_rate=0.01, epochs=30000, hidden_size=50, output_size=12, input_size=30, lr_decay_factor=1.0):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lr_decay_factor = lr_decay_factor

        self.input_size = input_size  # 30 = 4 sorted positional + 26 binary presence
        self.hidden_size = hidden_size
        self.output_size = output_size  # 12 outputs for binary encoding of 24 words

        self.I = (np.random.rand(self.hidden_size, self.input_size) - 0.5) / 2
        self.J = (np.random.rand(self.output_size, self.hidden_size) - 0.5) / 4
        self.L4 = np.zeros(self.hidden_size)
        self.L5 = np.zeros(self.output_size)

        # 24-word dictionary (12 original + 12 new)
        self.categories = [
            "BACK", "DARK", "EACH", "FROM", "JUST", "BEEN", "GOOD", "MUCH",
            "SOME", "TIME", "LIKE", "ONLY", "WORK", "WAVE", "ZERO", "ZONE",
            "VAIN", "VAST", "QUIZ", "HELP", "FIND", "PLUS", "YAWN", "STOP"
        ]

        if len(self.categories) != 24:
            raise ValueError(f"Must have exactly 24 categories, got {len(self.categories)}")

        # Create binary encoding patterns (24 unique patterns from 12 bits)
        self.binary_patterns = self._create_binary_patterns()

    def _create_binary_patterns(self):
        """Create 24 unique 12-bit binary patterns for the 24 words"""
        patterns = {}
        for i in range(len(self.categories)):
            # Convert index to 12-bit binary pattern
            pattern = np.array([(i >> bit) & 1 for bit in range(self.output_size)], dtype=float)
            patterns[i] = pattern
        return patterns

    def _index_to_binary(self, index):
        """Convert word index to binary target pattern"""
        return self.binary_patterns[index]

    def _binary_to_index(self, outputs):
        """Find closest binary pattern to output and return word index"""
        # Round outputs to 0 or 1
        binary_output = np.round(np.clip(outputs, 0, 1))

        # Find closest pattern (minimum hamming distance)
        min_distance = float('inf')
        best_index = 0

        for idx, pattern in self.binary_patterns.items():
            distance = np.sum(np.abs(binary_output - pattern))
            if distance < min_distance:
                min_distance = distance
                best_index = idx

        return best_index

    def sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1 / (1 + np.exp(-x))

    def get_adjacent_char(self, char):
        """Get a character adjacent to the given char in the alphabet"""
        original_ord = ord(char)
        if original_ord == ord('A'):
            return 'B'
        elif original_ord == ord('Z'):
            return 'Y'
        else:
            return chr(original_ord + random.choice([-1, 1]))

    def encode_word(self, word):
        # Hybrid encoding: Sorted positional + binary presence flags
        input_values = np.zeros(self.input_size)
        word = word.upper()

        unique_letters = sorted(set(char for char in word if 'A' <= char <= 'Z'))

        # Positions 0-3: Sorted positional encoding
        for i, char in enumerate(unique_letters[:4]):
            input_values[i] = (ord(char) - ord('A') + 1) / 26.0

        # Positions 4-29: Binary flags for each letter A-Z
        for char in unique_letters:
            char_index = ord(char) - ord('A')
            input_values[4 + char_index] = 1.0

        return input_values

    def forward(self, inputs):
        hidden_inputs_raw = np.dot(self.I, inputs) + self.L4
        hidden_outputs = self.sigmoid(hidden_inputs_raw)
        final_inputs_raw = np.dot(self.J, hidden_outputs) + self.L5
        final_outputs = self.sigmoid(final_inputs_raw)
        return hidden_outputs, final_outputs

    def train(self, word_str, category_index, augment=True):
        # Use binary pattern as target instead of one-hot
        targets = self._index_to_binary(category_index)
        repeats = 5 if augment else 1

        for i_repeat in range(repeats):
            current_word_str = word_str

            # Repeats 0: Original word (no modification)
            # Repeats 1-2: Character drop
            if augment and (i_repeat == 1 or i_repeat == 2):
                word_list = list(current_word_str)
                if len(word_list) > 1:
                    drop_index = random.randint(0, len(word_list) - 1)
                    word_list.pop(drop_index)
                    current_word_str = "".join(word_list)

            inputs = self.encode_word(current_word_str)

            # Repeats 3-4: Multi-letter adjacent substitution
            if augment and (i_repeat == 3 or i_repeat == 4):
                word_list = list(current_word_str)
                if len(word_list) > 0:
                    if i_repeat == 3:
                        num_to_replace = random.choice([1, 2])
                    else:
                        num_to_replace = random.randint(2, min(4, len(word_list)))

                    positions = random.sample(range(len(word_list)), min(num_to_replace, len(word_list)))
                    for pos in positions:
                        if random.random() < 0.8:
                            word_list[pos] = self.get_adjacent_char(word_list[pos])
                        else:
                            new_char = chr(random.randint(ord('A'), ord('Z')))
                            while new_char == word_list[pos]:
                                new_char = chr(random.randint(ord('A'), ord('Z')))
                            word_list[pos] = new_char
                    current_word_str = "".join(word_list)

            inputs = self.encode_word(current_word_str)

            hidden_outputs, final_outputs = self.forward(inputs)

            # Backpropagation with binary targets
            delta_output = (final_outputs - targets) * final_outputs * (1.0 - final_outputs)
            error_propagated_to_hidden = np.dot(self.J.T, delta_output)
            delta_hidden = error_propagated_to_hidden * hidden_outputs * (1.0 - hidden_outputs)

            self.J -= self.learning_rate * np.outer(delta_output, hidden_outputs)
            self.L5 -= self.learning_rate * delta_output
            self.I -= self.learning_rate * np.outer(delta_hidden, inputs)
            self.L4 -= self.learning_rate * delta_hidden

    def train_model(self, verbose=True):
        training_data = []
        for i, category_name in enumerate(self.categories):
            word_to_train = category_name[:self.input_size]
            training_data.append((word_to_train, i))

        decay_info = f", LR decay factor={self.lr_decay_factor}" if self.lr_decay_factor < 1.0 else ""
        if verbose:
            print(f"Starting training for {self.epochs} epochs with LR={self.initial_learning_rate}, Arch: {self.input_size}-{self.hidden_size}-{self.output_size}{decay_info}")
            print(f"24-word binary encoding (12-bit patterns)")

        for epoch in range(self.epochs):
            if self.lr_decay_factor < 1.0:
                self.learning_rate = self.initial_learning_rate * (self.lr_decay_factor ** (epoch / self.epochs))

            random.shuffle(training_data)
            for word, category_idx in training_data:
                self.train(word, category_idx, augment=True)

            if verbose and (epoch + 1) % (self.epochs // 20 if self.epochs >= 20 else 1) == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} completed (LR={self.learning_rate:.6f})")

        if verbose:
            print("Training complete!")

    def predict(self, word):
        inputs = self.encode_word(word)
        _, outputs = self.forward(inputs)

        # Decode binary pattern to category index
        category_index = self._binary_to_index(outputs)

        # Calculate confidence based on distance to target pattern
        target_pattern = self.binary_patterns[category_index]
        binary_output = np.round(np.clip(outputs, 0, 1))
        matches = np.sum(binary_output == target_pattern)
        confidence = int((matches / self.output_size) * 100)

        if 0 <= category_index < len(self.categories):
            return self.categories[category_index], confidence
        else:
            return "ERR_PREDICT_IDX", 0

    def test_accuracy(self, test_words_map=None, num_scrambled_target=48, test_modes=None):
        if test_modes is None:
            test_modes = ["original", "scrambled", "substitution", "drop"]

        if test_words_map is None:
            test_words_map = {}

            if "original" in test_modes:
                for word_cat in self.categories:
                    test_words_map[word_cat] = word_cat

            if "scrambled" in test_modes:
                scrambled_words_generated = 0
                num_categories = len(self.categories)
                scrambles_per_cat_target = (num_scrambled_target // num_categories if num_categories > 0 else 0) + 1
                attempts_per_category_heuristic = scrambles_per_cat_target * 3 + 5

                for word_cat in self.categories:
                    if scrambled_words_generated >= num_scrambled_target: break
                    unique_scrambles_for_this_cat = 0
                    if len(word_cat) <= 1: continue
                    for _ in range(attempts_per_category_heuristic):
                        char_list = list(word_cat)
                        random.shuffle(char_list)
                        scrambled = "".join(char_list)
                        if scrambled != word_cat and scrambled not in test_words_map:
                            test_words_map[scrambled] = word_cat
                            scrambled_words_generated += 1
                            unique_scrambles_for_this_cat += 1
                            if unique_scrambles_for_this_cat >= scrambles_per_cat_target or \
                               scrambled_words_generated >= num_scrambled_target:
                                break

            if "substitution" in test_modes:
                subs_per_cat = 3
                for word_cat in self.categories:
                    unique_subs = 0
                    attempts = subs_per_cat * 5
                    for _ in range(attempts):
                        if unique_subs >= subs_per_cat: break
                        word_list = list(word_cat)
                        sub_index = random.randint(0, len(word_list) - 1)
                        original_char = word_list[sub_index]
                        new_char = chr(random.randint(ord('A'), ord('Z')))
                        if new_char != original_char:
                            word_list[sub_index] = new_char
                            substituted = "".join(word_list)
                            if substituted not in test_words_map:
                                test_words_map[substituted] = word_cat
                                unique_subs += 1

            if "drop" in test_modes:
                drops_per_cat = 2
                for word_cat in self.categories:
                    unique_drops = 0
                    if len(word_cat) <= 1: continue
                    for _ in range(drops_per_cat):
                        word_list = list(word_cat)
                        drop_index = random.randint(0, len(word_list) - 1)
                        word_list.pop(drop_index)
                        dropped = "".join(word_list)
                        if dropped not in test_words_map:
                            test_words_map[dropped] = word_cat
                            unique_drops += 1

            if "adjacency" in test_modes:
                def get_adjacent_char(char):
                    original_ord = ord(char)
                    if original_ord == ord('A'):
                        return 'B'
                    elif original_ord == ord('Z'):
                        return 'Y'
                    else:
                        return chr(original_ord + random.choice([-1, 1]))

                for word_cat in self.categories:
                    word_len = len(word_cat)
                    for intensity in range(1, min(word_len + 1, 5)):
                        variants_created = 0
                        attempts = 10
                        for _ in range(attempts):
                            if variants_created >= 2: break
                            word_list = list(word_cat)
                            positions_to_replace = random.sample(range(word_len), intensity)
                            for pos in positions_to_replace:
                                word_list[pos] = get_adjacent_char(word_list[pos])
                            adjacent_variant = "".join(word_list)
                            if adjacent_variant != word_cat and adjacent_variant not in test_words_map:
                                test_words_map[adjacent_variant] = word_cat
                                variants_created += 1

            total_entries = len(test_words_map)
            test_breakdown = " + ".join([f"{mode}" for mode in test_modes if mode in ["original", "scrambled", "substitution", "drop", "adjacency"]])
            print(f"Generated test map with {total_entries} total entries ({test_breakdown} patterns).")

        correct_predictions = 0
        total_testable_words = len(test_words_map)
        if total_testable_words == 0:
            print("\nNo test words for accuracy calculation.")
            return []

        results_log = []
        print("\n--- Testing Model Robustness (24-Word Binary Encoding) ---")
        for word_input, expected_category_word in test_words_map.items():
            predicted_category_word, confidence = self.predict(word_input)
            is_match = (predicted_category_word == expected_category_word)
            if is_match: correct_predictions += 1
            result_str = f"Input: '{word_input}' → Predicted: '{predicted_category_word}' ({confidence}%) - Expected: '{expected_category_word}' [{ 'MATCH' if is_match else 'MISS' }]"
            results_log.append(result_str)
            print(result_str)
        accuracy = (correct_predictions / total_testable_words) * 100 if total_testable_words > 0 else 0
        print(f"\nRobustness Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_testable_words})")
        print("--- End of Test ---")
        return results_log

    def generate_verbose_ti_basic_weights(self):
        l4_zeros = "{" + ",".join(["0"]*self.hidden_size) + "}"
        l5_zeros = "{" + ",".join(["0"]*self.output_size) + "}"
        ti_basic_code = f""": Verbose Weights for TI-BASIC (24-Word Binary Encoding)
: Network Architecture: {self.input_size}-{self.hidden_size}-{self.output_size}
: 24 Words encoded as 12-bit binary patterns
{{{self.hidden_size},{self.input_size}}}→dim([I])
{{{self.output_size},{self.hidden_size}}}→dim([J])
{l4_zeros}→L₄
{l5_zeros}→L₅
"""
        for r in range(self.hidden_size):
            for c_idx in range(self.input_size):
                val_str = f"{self.I[r, c_idx]:.6f}".replace("-", "⁻")
                ti_basic_code += f"{val_str}→[I]({r+1},{c_idx+1})\n"
        for r in range(self.output_size):
            for c_idx in range(self.hidden_size):
                val_str = f"{self.J[r, c_idx]:.6f}".replace("-", "⁻")
                ti_basic_code += f"{val_str}→[J]({r+1},{c_idx+1})\n"
        for i, val in enumerate(self.L4):
            val_str = f"{val:.6f}".replace("-", "⁻")
            ti_basic_code += f"{val_str}→L₄({i+1})\n"
        for i, val in enumerate(self.L5):
            val_str = f"{val:.6f}".replace("-", "⁻")
            ti_basic_code += f"{val_str}→L₅({i+1})\n"
        return ti_basic_code

    def save_verbose_weights_to_file(self, filename="nn_weights_24word_verbose.txt"):
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.generate_verbose_ti_basic_weights())
        print(f"Verbose TI-BASIC weights saved to {filename}")

    def save_numpy_weights(self, filename="nn_weights_24word.npz"):
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savez_compressed(filename, i_weights=self.I, j_weights=self.J, l4_biases=self.L4, l5_biases=self.L5)
        print(f"NumPy weights saved to {filename}")

    def load_numpy_weights(self, filename="nn_weights_24word.npz"):
        if not os.path.exists(filename):
            print(f"Error: Weight file '{filename}' not found.")
            return False
        try:
            data = np.load(filename)
            if not ('i_weights' in data and 'j_weights' in data and 'l4_biases' in data and 'l5_biases' in data):
                print(f"Error: Weight file '{filename}' missing required arrays.")
                return False
            if data['i_weights'].shape != (self.hidden_size, self.input_size) or \
               data['j_weights'].shape != (self.output_size, self.hidden_size) or \
               data['l4_biases'].shape != (self.hidden_size,) or \
               data['l5_biases'].shape != (self.output_size,):
                print(f"Error: Weight dimensions mismatch.")
                return False
            self.I = data['i_weights']
            self.J = data['j_weights']
            self.L4 = data['l4_biases']
            self.L5 = data['l5_biases']
            print(f"NumPy weights loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Train 24-word neural network with binary output encoding')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--load_weights', type=str, metavar='<npz_file>', help='Load .npz weights, skip training.')
    parser.add_argument('--epochs', type=int, default=500000, help='Number of training epochs (default: 500000)')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate (default: 0.02)')
    parser.add_argument('--lr-decay', type=float, default=0.3, help='Learning rate decay factor (default: 0.3)')
    parser.add_argument('--hidden-size', type=int, default=50, help='Hidden layer size (default: 50)')
    parser.add_argument('--test', action='store_true', help='Run test suite after training')

    args = parser.parse_args()

    network = TICalcNeuralNetwork(
        learning_rate=args.lr,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        output_size=12,
        input_size=30,
        lr_decay_factor=args.lr_decay
    )

    if args.load_weights:
        if not network.load_numpy_weights(args.load_weights):
            return
    else:
        network.train_model(verbose=True)
        network.save_numpy_weights()
        network.save_verbose_weights_to_file()

    if args.test:
        network.test_accuracy(test_modes=["original", "scrambled", "substitution", "drop", "adjacency"])

if __name__ == "__main__":
    main()
