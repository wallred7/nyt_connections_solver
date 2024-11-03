class Evaluator:
    def __init__(self, words: list, labels: list):
        self.words = words
        self.labels = labels
        self.word_label_dict = self._create_word_label_dict()
        self.groups = self._create_groups()

    def _create_word_label_dict(self):
        """Creates a dictionary mapping words to their labels."""
        word_label_dict = {}
        for i, label in enumerate(self.labels):
            word_label_dict[self.words[i]] = label
        print(word_label_dict)
        return word_label_dict

    def _create_groups(self):
        """Creates groups of words based on their labels."""
        groups = {}
        for word, label in self.word_label_dict.items():
            if label not in groups:
                groups[label] = []
            groups[label].append(word)
        return groups

    def _check_group_match(self, group1, group2):
        """Checks if two groups are equivalent, regardless of order."""
        return set(group1) == set(group2)

    def evaluate(self, true_labels):
        """Evaluates the submission against the true labels."""
        # Assuming true_labels is a list of lists, where each inner list represents a group
        true_groups = {}
        for i, group in enumerate(true_labels):
            true_groups[i] = group

        # Check if the number of groups matches
        if len(self.groups) != len(true_groups):
            raise ValueError('Mismatched group and label sizes')

        # Check if each group matches a true group
        for label, group in self.groups.items():
            found_match = False
            for true_label, true_group in true_groups.items():
                if self._check_group_match(group, true_group):
                    found_match = True
                    break
            if not found_match:
                return False
        return True
