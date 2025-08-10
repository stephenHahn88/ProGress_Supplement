from abc import ABC, abstractmethod
from music21 import interval, note
import pickle
from collections import Counter
import torch

SCALE_DEGREES = [
    'A2', 'm2', 'P8', 'A6', 'A1', 'A5',
    'm7', 'M2', 'm6', 'M7',
    'm3', 'M3', 'P5', 'd7',
    'P4', 'M6', 'A4', 'd5'
]
SCALE_DEGREE_TO_CLASS = {sd: i for i, sd in enumerate(SCALE_DEGREES)}
CLASS_TO_SCALE_DEGREE = {i: sd for i, sd in enumerate(SCALE_DEGREES)}

SCALE_DEGREE_TO_C = {
    'A2': "D#",
    'm2': "Db",
    'P8': "C",
    'A6': "A#",
    'A1': "C#",
    'A5': "G#",
    'm7': "Bb",
    'M2': "D",
    'm6': "Ab",
    'M7': "B",
    'm3': "Eb",
    'M3': "E",
    "P5": "G",
    "d7": "Bbb",
    "P4": "F",
    "M6": "A",
    "A4": "F#",
    "d5": "Gb"
}

C_BASED_0 = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E-": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G-": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A-": 8,
    "A": 9,
    "Bbb": 9,
    "B--": 9,
    "A#": 10,
    "Bb": 10,
    "B-": 10,
    "B": 11
}


def parse_multiple_graphs(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    graphs = []
    idx = 0
    while idx < len(lines):
        graph = {}

        # Start of a new graph
        if lines[idx].startswith("N="):
            graph["N"] = int(lines[idx].split("=")[1])
            idx += 1

            # Parse X
            if lines[idx] == "X:":
                idx += 1
                graph["X"] = torch.tensor(list(map(int, lines[idx].split())))
                idx += 1

            # Parse E
            if lines[idx] == "E:":
                idx += 1
                E = []
                for _ in range(graph["N"]):
                    E.append(list(map(int, lines[idx].split())))
                    idx += 1
                graph["E"] = torch.tensor(E)

            # Parse R
            if lines[idx] == "R:":
                idx += 1
                R = []
                # Parse until we hit something that looks like a path or next N=
                while idx < len(lines) and not lines[idx].startswith("N=") and not lines[idx].startswith("X:") and not \
                        lines[idx].startswith("E:") and not lines[idx].startswith("R:"):
                    try:
                        R.append(list(map(float, lines[idx].split())))
                        idx += 1
                    except ValueError:
                        break
                graph["R"] = torch.tensor(R)

            # Assume next line is filename
            if idx < len(lines) and not lines[idx].startswith("N="):
                graph["filename"] = lines[idx]
                idx += 1

            graphs.append(graph)
        else:
            idx += 1  # Skip unrecognized lines

    return graphs


def retrieve_offsets_indices_notes(X, R):
    offset_idx = 7
    is_treble_idx = 8
    is_bass_idx = 9

    batch_size, num_notes, _ = R.shape

    treble_offsets_all = []
    treble_indices_all = []
    treble_notes_all = []

    bass_offsets_all = []
    bass_indices_all = []
    bass_notes_all = []

    for b in range(batch_size):
        R_b = R[b]
        X_b = X[b]

        offsets = R_b[:, offset_idx]
        is_treble = R_b[:, is_treble_idx].bool()
        is_bass = R_b[:, is_bass_idx].bool()

        treble_indices = torch.where(is_treble)[0]
        bass_indices = torch.where(is_bass)[0]

        treble_offsets = offsets[treble_indices]
        bass_offsets = offsets[bass_indices]

        treble_note_classes = X_b[treble_indices]
        treble_note_classes = torch.argmax(treble_note_classes, dim=-1)
        bass_note_classes = X_b[bass_indices]
        bass_note_classes = torch.argmax(bass_note_classes, dim=-1)

        # Convert classes to notes, assuming C Major is tonic
        treble_notes = [SCALE_DEGREE_TO_C[CLASS_TO_SCALE_DEGREE[n.item()]] for n in treble_note_classes]
        bass_notes = [SCALE_DEGREE_TO_C[CLASS_TO_SCALE_DEGREE[n.item()]] for n in bass_note_classes]

        treble_offsets_all.append(treble_offsets.tolist())
        treble_indices_all.append(treble_indices.tolist())
        treble_notes_all.append(treble_notes)

        bass_offsets_all.append(bass_offsets.tolist())
        bass_indices_all.append(bass_indices.tolist())
        bass_notes_all.append(bass_notes)

    return treble_offsets_all, treble_indices_all, treble_notes_all, \
        bass_offsets_all, bass_indices_all, bass_notes_all


class Rule(ABC):
    @abstractmethod
    def __init__(self, X, E, R, num_X_classes, num_E_classes, scg_kwargs):
        self.X = X
        self.E = E
        self.R = R
        self.num_X_classes = num_X_classes
        self.num_E_classes = num_E_classes
        self.scg_kwargs = scg_kwargs

    @abstractmethod
    def calculate_score(self) -> float:
        pass


class DissonanceChecker(Rule):
    def __init__(self, X, E, R, num_X_classes, num_E_classes, scg_kwargs):
        super().__init__(X, E, R, num_X_classes, num_E_classes, scg_kwargs)

    def calculate_score(self) -> float:
        disallowed_intervals = self.scg_kwargs['disallowed_intervals']
        (treble_offsets_all, treble_indices_all, treble_notes_all,
         bass_offsets_all, bass_indices_all, bass_notes_all) = \
            retrieve_offsets_indices_notes(self.X, self.R)

        num_dissonant = 0
        for (treble_offsets, treble_indices, treble_notes,
             bass_offsets, bass_indices, bass_notes) in zip(
            treble_offsets_all, treble_indices_all, treble_notes_all,
            bass_offsets_all, bass_indices_all, bass_notes_all):

            offset_counter = Counter(treble_offsets + bass_offsets)
            offset_dict_treble = {offset: (idx, note)
                                  for offset, idx, note in zip(treble_offsets, treble_indices, treble_notes)}
            offset_dict_bass = {offset: (idx, note)
                                for offset, idx, note in zip(bass_offsets, bass_indices, bass_notes)}

            # Gather all instances where counterpoint needs to be checked
            dissonances = 0
            for offset, (curr_treble_idx, curr_treble_note) in offset_dict_treble.items():
                if offset_counter[offset] < 2:
                    continue
                curr_bass_idx, curr_bass_note = offset_dict_bass[offset]
                if (C_BASED_0[curr_treble_note] - C_BASED_0[curr_bass_note]) % 12 in disallowed_intervals:
                    dissonances += 1
            return -dissonances


class ParallelChecker(Rule):
    def __init__(self, X, E, R, num_X_classes, num_E_classes, scg_kwargs):
        super().__init__(X, E, R, num_X_classes, num_E_classes, scg_kwargs)

    def check_is_parallel(self, counterpoint):
        if self.scg_kwargs['disallowed_parallels'] is None:
            # Avoid parallel 5ths and 8ves
            interval_half_steps = [0, 7]
        else:
            interval_half_steps = self.scg_kwargs['disallowed_parallels']

        # Convert pitches to numbers 0-11
        from_bass = C_BASED_0[counterpoint["from_bass"]]
        from_treble = C_BASED_0[counterpoint["from_treble"]]
        to_bass = C_BASED_0[counterpoint["to_bass"]]
        to_treble = C_BASED_0[counterpoint["to_treble"]]
        if from_bass == to_bass and from_treble == to_treble:
            return 0

        from_interval = (from_treble - from_bass) % 12
        to_interval = (to_treble - to_bass) % 12
        for interval_i in interval_half_steps:
            if from_interval == interval_i and to_interval == interval_i:
                return 1
        return 0

    def calculate_score(self):
        (treble_offsets_all, treble_indices_all, treble_notes_all,
         bass_offsets_all, bass_indices_all, bass_notes_all) = \
            retrieve_offsets_indices_notes(self.X, self.R)

        num_parallels = 0
        for (treble_offsets, treble_indices, treble_notes,
             bass_offsets, bass_indices, bass_notes) in zip(
            treble_offsets_all, treble_indices_all, treble_notes_all,
            bass_offsets_all, bass_indices_all, bass_notes_all):
            offset_counter = Counter(treble_offsets + bass_offsets)
            offset_dict_treble = {offset: (idx, note)
                                  for offset, idx, note in zip(treble_offsets, treble_indices, treble_notes)}
            offset_dict_bass = {offset: (idx, note)
                                for offset, idx, note in zip(bass_offsets, bass_indices, bass_notes)}

            # Gather all instances where counterpoint needs to be checked
            counterpoints = []
            for offset, (curr_treble_idx, curr_treble_note) in offset_dict_treble.items():
                if offset_counter[offset] < 2 or offset <= 0:
                    continue
                curr_bass_idx, curr_bass_note = offset_dict_bass[offset]

                prev_offset_treble = treble_offsets[treble_offsets.index(offset) - 1]
                prev_offset_bass = bass_offsets[bass_offsets.index(offset) - 1]
                pair_prev = (offset_dict_treble[prev_offset_treble][1], offset_dict_bass[prev_offset_bass][1])

                counterpoints.append({
                    "from_treble": pair_prev[0],
                    "from_bass": pair_prev[1],
                    "to_treble": curr_treble_note,
                    "to_bass": curr_bass_note
                })

            num_parallels += sum([self.check_is_parallel(counterpoint) for counterpoint in counterpoints])
        # print(num_parallels)
        return -num_parallels


if __name__ == "__main__":
    from pprint import pprint

    with open("./debug.pkl", "rb") as f:
        data = pickle.load(f)
    X = data[0]
    R = data[1]

    scg_kwargs = {
        "num_samples": 8,
        "rules": [ParallelChecker],
        "disallowed_intervals": [2],
        "disallowed_parallels": [0, 7]
    }

    dc = DissonanceChecker(X, None, R, 18, 4, scg_kwargs)
    dc.calculate_score()

    # pc = ParallelChecker(X, None, R, 18, 4, None)
    # pc.calculate_score()

    # graphs = parse_multiple_graphs("./generated_samples1.txt")
    # first = graphs[0]
    # print(first["X"])
    #
    # pc = ParallelChecker(first["X"], first["E"], first["R"], 18, 4, None)
    # pc.calculate_score()
    # is_parallel = pc.check_is_parallel({
    #             "from_treble": "D",
    #             "from_bass": "D",
    #             "to_treble": "D",
    #             "to_bass": "F#"
    # })
    # print(is_parallel)
