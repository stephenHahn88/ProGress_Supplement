import copy
import math
from collections import defaultdict
from pprint import pprint

import music21.stream
from music21.note import Note, Rest
from music21.stream import Score, Part, Measure
from music21 import converter, note, tempo, stream, key, voiceLeading, chord, interval

from phrase_stitching.config import *

COMMON_ROMAN_NUMERALS_C = {
    "I": ["C", "E", "G"],
    "i": ["C", "E-", "G"],
    "ii": ["D", "F", "A"],
    "iio": ["D", "F", "A-"],
    "iii": ["E", "G", "B"],
    "III": ["E-", "G", "B-"],
    "IV": ["F", "A", "C"],
    "iv": ["F", "A-", "C"],
    "V": ["G", "B", "D", "F"],
    "v": ["G", "B-", "D"],
    "vi": ["A", "C", "E"],
    "VI": ["A-", "C", "E-"],
    "viio": ["B", "D", "F"],
    "VII": ["B-", "D", "F", "A-"]
}

FUNCTION_TO_ROMAN_NUMERAL = {
    "tonic": ["I", "i", "iii", "III", "vi", "VI"],
    "predominant": ["ii", "iio", "IV", "iv", "v", "vi", "VI", "VII"],
    "dominant": ["V", "viio"]
}

MODES_ROMAN_NUMERALS = {
    "major": {"I", "ii", "iii", "IV", "V", "vi", "viio"},
    "minor": {"i", "iio", "III", "iv", "v", "V", "VI", "viio", "VII"}
}

VALID_FUNCTION_PROGRESSION = {
    "tonic": ["tonic", "predominant", "dominant"],
    "predominant": ["predominant", "dominant"],
    "dominant": ["dominant", "tonic"]
}

UNLIKELY_PROGRESSIONS = {
    "I": [],
    "i": [],
    "ii": ["IV", "iv", "vi", "VI", "v"],
    "iio": ["IV", "iv", "vi", "VI", "v"],
    "iii": ["i", "v", "V", "viio"],
    "III": ["I", "v", "V", "viio"],
    "IV": ["iii", "III", "v", "vi", "VI"],
    "iv": ["ii", "iii", "III", "v", "vi", "VI"],
    "v": ["iii", "III", "vi", "VI", "viio"],
    "V": ["ii", "iio", "iii", "III", "iv", "IV", "viio"],
    "vi": ["i"],
    "VI": ["I"],
    "viio": ["ii", "iio", "iii", "III", "iv", "IV"],
    "VII": ["I", "ii", "iio", "IV", "iv", "v", "V", "vi", "VI", "viio"]
}


class InvalidAnalysisException(Exception):
    def __init__(self, filepath):
        super().__init__(f"Invalid analysis for file: {filepath}")


def get_active_note_at(time, voice):
    """Get the note sounding at a specific time in a voice"""
    current_time = 0.0
    for offset, n in voice:
        start = current_time
        end = start + n.quarterLength
        if start <= time < end:
            return n
        current_time = end
    return None


def get_vertical_pairs(score: music21.stream.Score, include_octave=False):
    # Flatten the parts and store notes with absolute times
    voices = []
    for part in score.parts:
        flat_notes = []
        current_offset = 0.0
        for n in part.flat.notes:
            flat_notes.append((current_offset, n))
            current_offset += n.quarterLength
        voices.append(flat_notes)

    # Determine the total length of the piece (in quarterLength)
    total_length = max(
        sum(n.quarterLength for _, n in voice) for voice in voices
    )

    # Collect vertical pairs at each sixteenth note (0.25 quarter note) interval
    sixteenth = 0.25
    time_points = [round(i * sixteenth, 5) for i in range(int(total_length / sixteenth) + 1)]

    vertical_pairs = {}
    # Print vertical note pairs at each sixteenth note
    for t in time_points:
        n1 = get_active_note_at(t, voices[0])
        n2 = get_active_note_at(t, voices[1])
        if isinstance(n1, note.Note) and isinstance(n2, note.Note):
            if include_octave:
                vertical_pairs[f"{t:.2f}"] = (n1.nameWithOctave, n2.nameWithOctave)
            else:
                vertical_pairs[f"{t:.2f}"] = (n1.name, n2.name)
    return vertical_pairs


def determine_possible_Roman_numerals(vertical_pairs):
    potential_RNs = {}
    for offset, (note1, note2) in vertical_pairs.items():
        Roman_numerals = []
        for RN, associated_notes in COMMON_ROMAN_NUMERALS_C.items():
            if note1 in associated_notes and note2 in associated_notes:
                Roman_numerals.append(RN)
        potential_RNs[offset] = Roman_numerals
    return potential_RNs


def get_starting_potential_Romans(potential_Roman_numerals, vertical_pairs):
    return [
               potential_Roman_numerals['0.00'],
               potential_Roman_numerals['1.00'],
               potential_Roman_numerals['2.00']
           ], [
               vertical_pairs['0.00'][1],
               vertical_pairs['1.00'][1],
               vertical_pairs['2.00'][1]
           ]


def get_ending_potential_Romans(potential_Roman_numerals, vertical_pairs):
    last_beat = max([math.floor(float(offset)) for offset in potential_Roman_numerals.keys()])
    return [
               potential_Roman_numerals[f'{last_beat - i:.2f}'] for i in range(2, -1, -1)
           ], [
               vertical_pairs[f'{last_beat - i:.2f}'][1] for i in range(2, -1, -1)
           ]


def get_potential_Romans(keys, potential_Roman_numerals, vertical_pairs):
    return [
               potential_Roman_numerals[k] for k in keys
           ], [
               vertical_pairs[k][1] for k in keys
           ]


def determine_most_likely_Roman_numerals(potential_Roman_numerals):
    """
    potential_Roman_numerals: A list of possible Roman numerals over three beats
    """

    # determine mode possibilities
    major_set = MODES_ROMAN_NUMERALS["major"]
    major_without_dom = major_set.copy()
    major_without_dom.remove("V")
    major_without_dom.remove("viio")

    minor_set = MODES_ROMAN_NUMERALS["minor"]
    minor_without_dom = minor_set.copy()
    minor_without_dom.remove("V")
    minor_without_dom.remove("viio")

    could_be_major = True
    could_be_minor = True
    for potential_RNs in potential_Roman_numerals:
        potential_RNs_set = set(potential_RNs)
        if len(potential_RNs_set.intersection(major_set)) <= 0:
            could_be_major = False
        elif len(potential_RNs_set.intersection(minor_set)) <= 0:
            could_be_minor = False

    # filter Roman numerals that aren't part of the mode
    modal_ordered_strong_beats = []
    for potential_RNs in potential_Roman_numerals:
        pruned_RNs_set = set(potential_RNs)
        if not could_be_major:
            pruned_RNs_set = pruned_RNs_set.difference(major_without_dom)
        if not could_be_minor:
            pruned_RNs_set = pruned_RNs_set.difference(minor_without_dom)
        modal_ordered_strong_beats.append(list(pruned_RNs_set))

    valid_RN_sequences = []
    for rn_1 in modal_ordered_strong_beats[0]:
        for rn_2 in modal_ordered_strong_beats[1]:
            for rn_3 in modal_ordered_strong_beats[2]:
                rn_1_functions = [function for function, RNs in FUNCTION_TO_ROMAN_NUMERAL.items() if rn_1 in RNs]
                rn_2_functions = [function for function, RNs in FUNCTION_TO_ROMAN_NUMERAL.items() if rn_2 in RNs]
                rn_3_functions = [function for function, RNs in FUNCTION_TO_ROMAN_NUMERAL.items() if rn_3 in RNs]
                if is_valid_function_progression(rn_1_functions, rn_2_functions) and is_valid_function_progression(
                        rn_2_functions, rn_3_functions):
                    valid_RN_sequences.append((rn_1, rn_2, rn_3))

    # pprint(potential_Roman_numerals)
    # pprint(modal_ordered_strong_beats)
    # pprint(valid_RN_sequences)
    return valid_RN_sequences


def is_valid_function_progression(from_functions, to_functions):
    for from_function in from_functions:
        for to_function in to_functions:
            if to_function in VALID_FUNCTION_PROGRESSION[from_function]:
                return True
    return False


def prune_unlikely_progressions(valid_RN_sequences):
    pruned_progressions = []
    for progression in valid_RN_sequences:
        first_part_unlikely = progression[1] in UNLIKELY_PROGRESSIONS[progression[0]]
        second_part_unlikely = progression[2] in UNLIKELY_PROGRESSIONS[progression[1]]
        # print(progression, first_part_unlikely, second_part_unlikely)
        if not first_part_unlikely and not second_part_unlikely:
            pruned_progressions.append(progression)
    return pruned_progressions


def score_valid_progressions(valid_progressions, associated_bass_notes):
    scores = defaultdict(float)
    for progression in valid_progressions:
        # avoid progressions that mix major and minor
        has_major = False
        has_minor = False
        for RN in progression:
            if RN in ["V", "viio"]: continue
            if RN in MODES_ROMAN_NUMERALS["major"]:
                has_major = True
            if RN in MODES_ROMAN_NUMERALS["minor"]:
                has_minor = True
        if has_major and has_minor:
            scores[progression] += SCORE_MAJOR_AND_MINOR

        for RN, bass_note in zip(progression, associated_bass_notes):
            # root position
            if COMMON_ROMAN_NUMERALS_C[RN][0] == bass_note:
                scores[progression] += SCORE_ROOT_POSITION
            # first inversion
            if COMMON_ROMAN_NUMERALS_C[RN][1] == bass_note:
                scores[progression] += SCORE_FIRST_INVERSION
            # second inversion
            if COMMON_ROMAN_NUMERALS_C[RN][2] == bass_note:
                scores[progression] += SCORE_SECOND_INVERSION
            # III and v are relatively rare in reality
            if RN == "iii" or RN == "III":
                scores[progression] += SCORE_INCLUDES_III
            if RN == "v":
                scores[progression] += SCORE_INCLUDES_v
    return scores


def choose_highest_scoring_progression(scores):
    highest_scoring_progression = max(scores, key=scores.get)
    if scores[highest_scoring_progression] < 0:
        raise InvalidAnalysisException("All progressions have a bad score")
    return highest_scoring_progression


def rank_progressions_highest_score_first(scores):
    choose_highest_scoring_progression(scores)
    progressions = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return progressions


def find_file_start_and_end(filepath):
    score = converter.parse(filepath)
    vertical_pairs = get_vertical_pairs(score)

    has_problematic_start = False
    has_problematic_end = False

    potential_Roman_numerals = determine_possible_Roman_numerals(vertical_pairs)

    starting_numerals, starting_bass = get_starting_potential_Romans(potential_Roman_numerals, vertical_pairs)
    valid_starting_RN_sequences = determine_most_likely_Roman_numerals(starting_numerals)
    pruned_valid_starting_RN_sequences = prune_unlikely_progressions(valid_starting_RN_sequences)
    start_scores = score_valid_progressions(pruned_valid_starting_RN_sequences, starting_bass)
    try:
        best_start = choose_highest_scoring_progression(start_scores)
    except InvalidAnalysisException as e:
        has_problematic_start = True

    ending_numerals, ending_bass = get_ending_potential_Romans(potential_Roman_numerals, vertical_pairs)
    valid_ending_RN_sequences = determine_most_likely_Roman_numerals(ending_numerals)
    pruned_valid_ending_RN_sequences = prune_unlikely_progressions(valid_ending_RN_sequences)
    end_scores = score_valid_progressions(pruned_valid_ending_RN_sequences, ending_bass)
    try:
        best_end = choose_highest_scoring_progression(end_scores)
    except InvalidAnalysisException as e:
        has_problematic_end = True

    if has_problematic_start or has_problematic_end:
        print("Problematic start") if has_problematic_start else ''
        print("Problematic end") if has_problematic_end else ''
        return None, None, None
    return score, best_start, best_end


def get_ranked_triplets_from_beat(filepath, curr_beat, potential_Roman_numerals, vertical_pairs):
    next_beat = f"{(float(curr_beat) + 1):.2f}"
    next_next_beat = f"{(float(curr_beat) + 2):.2f}"

    # Find and score best potential RN triple
    numerals, bass = get_potential_Romans([curr_beat, next_beat, next_next_beat], potential_Roman_numerals,
                                          vertical_pairs)
    valid_RN_sequences = determine_most_likely_Roman_numerals(numerals)
    pruned_valid_sequences = prune_unlikely_progressions(valid_RN_sequences)
    scores = score_valid_progressions(pruned_valid_sequences, bass)
    if len(scores) == 0:
        raise InvalidAnalysisException(filepath)
    return rank_progressions_highest_score_first(scores)


def align_parts(part1, part2):
    """Yield pairs of simultaneous notes from part1 and part2"""
    all_offsets = sorted(
        set(n.offset for n in part1.recurse().notes).union(
            n.offset for n in part2.recurse().notes
        )
    )
    for i in range(len(all_offsets) - 1):
        offset_start = all_offsets[i]
        offset_end = all_offsets[i + 1]

        n1_start = part1.recurse().stream().getElementAtOrBefore(offset_start, [note.Note])
        n1_end = part1.recurse().stream().getElementAtOrBefore(offset_end, [note.Note])
        n2_start = part2.recurse().stream().getElementAtOrBefore(offset_start, [note.Note])
        n2_end = part2.recurse().stream().getElementAtOrBefore(offset_end, [note.Note])

        if None not in (n1_start, n1_end, n2_start, n2_end):
            yield n1_start, n1_end, n2_start, n2_end


def check_bad_counterpoint(score: Score, filepath=""):
    parallels = []
    part1, part2 = score.parts[0], score.parts[1]

    for n1a, n1b, n2a, n2b in align_parts(part1, part2):
        vlq = voiceLeading.VoiceLeadingQuartet(n1a, n1b, n2a, n2b)
        if vlq.parallelFifth():
            interval_type = 'P5'
        elif vlq.parallelOctave():
            interval_type = 'P8'
        else:
            continue

        parallels.append({
            'type': interval_type,
            'start_notes': (n1a.nameWithOctave, n2a.nameWithOctave),
            'end_notes': (n1b.nameWithOctave, n2b.nameWithOctave),
            'offsets': (n1a.offset, n1b.offset)
        })

    if len(parallels) > 0:
        raise InvalidAnalysisException(f"Parallel: {parallels}")


def check_illegal_harmonics_on_integer_beats(score, illegal_intervals=('m2', "M2", "P4", "M7")):
    illegal_instances = []
    part1, part2 = score.parts[0], score.parts[1]

    # Determine maximum duration across both parts
    max_offset = max(
        part1.recurse().stream().notesAndRests.stream().highestOffset,
        part2.recurse().stream().notesAndRests.stream().highestOffset
    )

    # Generate integer beat offsets up to the ceiling of max_offset
    integer_beats = range(0, math.ceil(max_offset) + 1)

    for offset in integer_beats:
        n1 = part1.recurse().stream().getElementAtOrBefore(offset, [note.Note])
        n2 = part2.recurse().stream().getElementAtOrBefore(offset, [note.Note])

        if n1 is None or n2 is None:
            continue

        iv = interval.Interval(n2, n1)
        simple_name = iv.simpleName  # Interval name modulo octave, e.g., 'm2', 'P5', 'A4'

        if simple_name in illegal_intervals:
            illegal_instances.append({
                'offset': offset,
                'note1': n1.nameWithOctave,
                'note2': n2.nameWithOctave,
                'interval': simple_name
            })

    if len(illegal_instances) > 0:
        raise InvalidAnalysisException(simple_name)


def analyze_entire_phrase(filepath):
    score = converter.parse(filepath)
    vertical_pairs = get_vertical_pairs(score)
    potential_Roman_numerals = determine_possible_Roman_numerals(vertical_pairs)
    is_clearly_major = all([n not in ['E-', 'A-', 'B-'] for notes in vertical_pairs.values() for n in notes])
    is_clearly_minor = any([n in ['E-', 'A-', 'B-'] for notes in vertical_pairs.values() for n in notes]) and \
                       all([n != 'E' for notes in vertical_pairs.values() for n in notes])

    curr_beat = '0.00'
    all_scores = []
    while f"{(float(curr_beat) + 2):.2f}" in potential_Roman_numerals.keys():
        ranked_scores = get_ranked_triplets_from_beat(filepath, curr_beat, potential_Roman_numerals, vertical_pairs)
        all_scores.append(ranked_scores)
        curr_beat = f"{(float(curr_beat) + 1):.2f}"
    # pprint(all_scores)

    simplest_route = [beat[0][0][0] for beat in all_scores]
    simplest_route += [all_scores[-1][0][0][1], all_scores[-1][0][0][2]]
    if is_clearly_major:
        simplest_route = [rn.upper() if rn in ['i', 'iv', 'v'] else rn for rn in simplest_route]
        simplest_route = [rn.lower() if rn in ['III', 'VI'] else rn for rn in simplest_route]
    if is_clearly_minor:
        simplest_route = [rn.lower() if rn in ['I', 'IV'] else rn for rn in simplest_route]
        simplest_route = [rn.upper() if rn in ['iii', 'vi'] else rn for rn in simplest_route]
    return score, simplest_route


def get_phrases_with_working_start_and_end(folder_indices=None):
    if folder_indices is None:
        folder_indices = [1, 2, 3, 4]
    working_phrases = {}
    for output_folder in folder_indices:
        for i in range(1, 21):
            filepath = f'./diffusion_output/output_graphs_{output_folder}/output_graph_{i}.xml'
            print(f"-----------------{filepath}-----------------")
            score, best_start, best_end = find_file_start_and_end(filepath)
            if best_start and best_end:
                print(best_start, best_end)
                working_phrases[filepath] = {"score": score, "start": best_start, "end": best_end}

    return working_phrases


def check_bad_mode_mixture(score, illegal_coexistence=None):
    if illegal_coexistence is None:
        illegal_coexistence = [
            ("E", "E-"),
            ("A", "A-"),
            ("E", "A-"),
            ("E", "B-")
        ]

    has_first = False
    has_second = False

    for pair in illegal_coexistence:
        for n in score.recurse().notes:
            if n.pitch.name == pair[0]:
                has_first = True
            elif n.pitch.name == pair[1]:
                has_second = True

        # Early exit if both are found
        if has_first and has_second:
            raise InvalidAnalysisException(f"{pair}")

        has_first = False
        has_second = False


def main():
    invalid_count = 0
    file_count = 0
    files_per_folder = 40
    for f in range(8, 14):
        for i in range(1, files_per_folder + 1):
            filepath = f"./diffusion_output/output_graphs_{f}/output_graph_{i}.xml"
            file_count += 1
            try:
                score, analysis = analyze_entire_phrase(filepath)
                check_bad_mode_mixture(score)
                check_illegal_harmonics_on_integer_beats(score)
                check_bad_counterpoint(score)

                print(i)
                # print(analysis)
            except FileNotFoundError as e:
                file_count -= 1
                continue
            except InvalidAnalysisException as e:
                invalid_count += 1
                print(i, e)
                # print(f"error for {filepath}")
                continue
    # working_phrases = get_phrases_with_working_start_and_end(folder_indices=[1, 2, 3, 4])
    # pprint(working_phrases)

    # score1 = working_phrases["./diffusion_output/output_graphs_2/output_graph_11.xml"]["score"]
    # score2 = working_phrases["./diffusion_output/output_graphs_2/output_graph_2.xml"]["score"]
    # transpose_score(score2, -8)
    # combined_score = combine_two_scores(score1, score2)
    # combined_score.show()
    print()
    print()
    print(invalid_count, file_count)
    print(invalid_count/file_count)


if __name__ == "__main__":
    main()
