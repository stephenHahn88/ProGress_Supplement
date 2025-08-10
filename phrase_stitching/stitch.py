import copy
import pickle
import random
from collections import defaultdict

from music21 import stream, tempo, meter
from music21.stream import Part, Score

from phrase_stitching.RN_analysis import analyze_entire_phrase, get_vertical_pairs, InvalidAnalysisException, \
    check_illegal_harmonics_on_integer_beats, check_bad_counterpoint, check_bad_mode_mixture

from phrase_stitching.write_inner_voices import write_inner_voices

# Key = new tonic relative to previous phrase,
# Value = harmonies that may start a new phrase
POSSIBLE_STARTS_ENDING_FROM_TONIC = {
    "I": ["I", "ii", "iii", "IV", "V", "vi", "viio"],  # C -> C
    "ii": ["VII", "III", "i", "I"],  # C -> d
    "V": ["IV", "V", "ii", "viio", "iio", "iv"],  # C -> G
    "IV": ["V", "I", "vi"],  # C -> F
    "vi": ["III", "iv", "VI", "iio"],  # C -> a
    "iii": ["VI", "iio", "iv", "V", "viio"],  # C -> e
    "i": ["i", "iio", "III", "iv", "v", "V", "VI", "VII", "viio"],  # c -> c
    "v": ["iv", "V", "viio", "iio"],  # c -> g
    "iv": ["v", "V", "iv", "iio"],  # c -> f
    "VI": ["iii", "IV", "vi", "ii"],  # c -> Ab
    "III": ["vi", "ii", "IV", "V", "viio"]  # c -> Eb
}

POSSIBLE_KEY_PROGRESSIONS = [
    ["I", "iii", "V", "I"],
    ["I", "V", "I"],
    ["i", "III", "i"]
]


def extend_last_note_to_fill_measure(score, analysis):
    """
    Extends the last note in each part to complete the final measure up to the full measure length.
    Assumes 4/4 time unless another time signature is specified in the part.
    """
    for part in score.parts:
        ts = part.recurse().getElementsByClass(meter.TimeSignature).first()
        measure_length_quarters = ts.barDuration.quarterLength if ts else 4.0

        last_note = None
        for n in part.recurse().notesAndRests:
            last_note = n

        if last_note is None:
            continue  # skip empty parts

        last_measure = last_note.getContextByClass('Measure')
        if last_measure is None:
            continue

        # Total duration of the last measure
        total_duration = sum(n.duration.quarterLength for n in last_measure.notesAndRests)

        remaining_duration = measure_length_quarters - total_duration

        if remaining_duration > 0 and last_note.isNote:
            last_note.duration.quarterLength += remaining_duration
            for _ in range(int(remaining_duration)):
                analysis.append(analysis[-1])
            # print(f"Extended {last_note} by {remaining_duration} quarter note(s) to complete the measure.")
        # else:
            # print("No extension needed or last item is a rest.")

    return score, analysis


def combine_two_scores(score1, score2):
    """Combine two music21 scores by preserving measure structure"""
    combined_score = Score()

    # Get tempo and time signature from first score
    tempos = score1.recurse().getElementsByClass(tempo.MetronomeMark)
    first_tempo = copy.deepcopy(tempos[0]) if tempos else None

    time_sigs = score1.recurse().getElementsByClass('TimeSignature')
    time_sig = time_sigs[0] if time_sigs else None

    for part1, part2 in zip(score1.parts, score2.parts):
        combined_part = Part()

        # Add score-level elements to the first measure of the combined part
        if first_tempo:
            combined_part.insert(0, copy.deepcopy(first_tempo))
        if time_sig:
            combined_part.insert(0, copy.deepcopy(time_sig))

        # Copy all measures from part1
        measures1 = part1.getElementsByClass(stream.Measure)
        for measure in measures1:
            combined_part.append(copy.deepcopy(measure))

        # Copy all measures from part2, but adjust their measure numbers
        measures2 = part2.getElementsByClass(stream.Measure)
        last_measure_num = len(measures1)

        for i, measure in enumerate(measures2):
            measure_copy = copy.deepcopy(measure)
            measure_copy.number = last_measure_num + i + 1
            combined_part.append(measure_copy)

        # Remove trailing empty measures
        combined_part = remove_trailing_empty_measures(combined_part)

        combined_score.append(combined_part)

    return combined_score


def remove_trailing_empty_measures(part):
    """Remove only trailing empty measures while preserving measure structure"""
    measures = part.getElementsByClass(stream.Measure)

    if not measures:
        return part

    # Find the last measure with actual content
    last_content_index = -1
    for i, measure in enumerate(measures):
        if measure.notesAndRests:
            last_content_index = i

    if last_content_index == -1:
        # No measures with content, return as is
        return part

    # If there are trailing empty measures, remove them
    if last_content_index < len(measures) - 1:
        # Create new part with non-measure elements
        new_part = Part()

        # Copy non-measure elements (like clefs, key signatures, etc.)
        for element in part:
            if not isinstance(element, stream.Measure):
                new_part.append(copy.deepcopy(element))

        # Copy only measures up to the last one with content
        for i in range(last_content_index + 1):
            new_part.append(copy.deepcopy(measures[i]))

        return new_part

    return part


def transpose_score(score, semitones):
    transposed_score = copy.deepcopy(score)
    for n in transposed_score.recurse().notes:
        n.transpose(semitones, inPlace=True)
    return transposed_score


def stitch_i_III_iv_i(score_analysis_starts, score_analysis_ends, score_analysis):
    beginning = None
    middle = None
    middle2 = None
    end = None

    while beginning == middle or beginning == middle2 or beginning == end or \
            middle == middle2 or middle == end or middle2 == end:
        print("sampling possibilities...")
        beginning = random.sample(score_analysis_ends['i'], 1)[0]
        # print(len(score_analysis_ends['i']))

        possible_starts_for_middle = POSSIBLE_STARTS_ENDING_FROM_TONIC["III"]
        possible_scores = [
            score for possible_start in possible_starts_for_middle for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["I"]
        ]
        # print(len(possible_scores))
        middle = random.sample(possible_scores, 1)[0]

        possible_starts_for_middle2 = POSSIBLE_STARTS_ENDING_FROM_TONIC["ii"]
        possible_scores = [
            score for possible_start in possible_starts_for_middle2 for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["i"]
        ]
        # print(len(possible_scores))
        middle2 = random.sample(possible_scores, 1)[0]

        possible_starts_for_end = POSSIBLE_STARTS_ENDING_FROM_TONIC["V"]
        possible_scores = [
            score for possible_start in possible_starts_for_end for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["i"]
        ]
        # print(len(possible_scores))
        end = random.sample(possible_scores, 1)[0]

    beginning, beginning_analysis = extend_last_note_to_fill_measure(beginning, score_analysis[beginning])
    middle, middle_analysis = extend_last_note_to_fill_measure(middle, score_analysis[middle])
    middle2, middle2_analysis = extend_last_note_to_fill_measure(middle2, score_analysis[middle2])
    end, end_analysis = extend_last_note_to_fill_measure(end, score_analysis[end])

    write_inner_voices(beginning, beginning_analysis)
    write_inner_voices(middle, middle_analysis)
    write_inner_voices(middle2, middle2_analysis)
    write_inner_voices(end, end_analysis)

    middle = transpose_score(middle, 3)
    middle2 = transpose_score(middle2, 5)

    final = combine_two_scores(beginning, middle)
    final = combine_two_scores(final, middle2)
    final = combine_two_scores(final, end)
    final.show()


def stitch_i_III_V_i(score_analysis_starts, score_analysis_ends, score_analysis):
    beginning = None
    middle = None
    middle2 = None
    end = None

    while beginning == middle or beginning == middle2 or beginning == end or \
            middle == middle2 or middle == end or middle2 == end:
        print("sampling possibilities...")
        beginning = random.sample(score_analysis_ends['i'], 1)[0]
        # print(len(score_analysis_ends['i']))

        possible_starts_for_middle = POSSIBLE_STARTS_ENDING_FROM_TONIC["III"]
        possible_scores = [
            score for possible_start in possible_starts_for_middle for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["I"]
        ]
        # print(len(possible_scores))
        middle = random.sample(possible_scores, 1)[0]

        possible_starts_for_middle2 = POSSIBLE_STARTS_ENDING_FROM_TONIC["iii"]
        possible_scores = [
            score for possible_start in possible_starts_for_middle2 for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["i"]
        ]
        # print(len(possible_scores))
        middle2 = random.sample(possible_scores, 1)[0]

        possible_starts_for_end = POSSIBLE_STARTS_ENDING_FROM_TONIC["IV"]
        possible_scores = [
            score for possible_start in possible_starts_for_end for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["i"]
        ]
        # print(len(possible_scores))
        end = random.sample(possible_scores, 1)[0]

    beginning, beginning_analysis = extend_last_note_to_fill_measure(beginning, score_analysis[beginning])
    middle, middle_analysis = extend_last_note_to_fill_measure(middle, score_analysis[middle])
    middle2, middle2_analysis = extend_last_note_to_fill_measure(middle2, score_analysis[middle2])
    end, end_analysis = extend_last_note_to_fill_measure(end, score_analysis[end])

    write_inner_voices(beginning, beginning_analysis)
    write_inner_voices(middle, middle_analysis)
    write_inner_voices(middle2, middle2_analysis)
    write_inner_voices(end, end_analysis)

    middle = transpose_score(middle, 3)
    middle2 = transpose_score(middle2, 7)

    final = combine_two_scores(beginning, middle)
    final = combine_two_scores(final, middle2)
    final = combine_two_scores(final, end)
    final.show()


def stitch_I_V_I(score_analysis_starts, score_analysis_ends, score_analysis):
    beginning = None
    middle = None
    middle2 = None
    end = None

    while beginning == middle or beginning == middle2 or beginning == end or \
            middle == middle2 or middle == end or middle2 == end:
        # print("sampling possibilities...")
        beginning = random.sample(score_analysis_ends['I'], 1)[0]

        possible_starts_for_middle = POSSIBLE_STARTS_ENDING_FROM_TONIC["V"]
        possible_scores = [
            score for possible_start in possible_starts_for_middle for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["I"]
        ]
        # print(len(possible_scores))
        middle = random.sample(possible_scores, 1)[0]

        possible_starts_for_middle2 = POSSIBLE_STARTS_ENDING_FROM_TONIC["I"]
        possible_scores = [
            score for possible_start in possible_starts_for_middle2 for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["I"]
        ]
        # print(len(possible_scores))
        middle2 = random.sample(possible_scores, 1)[0]

        possible_starts_for_end = POSSIBLE_STARTS_ENDING_FROM_TONIC["IV"]
        possible_scores = [
            score for possible_start in possible_starts_for_end for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["I"]
        ]
        # print(len(possible_scores))
        end = random.sample(possible_scores, 1)[0]

    beginning, beginning_analysis = extend_last_note_to_fill_measure(beginning, score_analysis[beginning])
    middle, middle_analysis = extend_last_note_to_fill_measure(middle, score_analysis[middle])
    middle2, middle2_analysis = extend_last_note_to_fill_measure(middle2, score_analysis[middle2])
    end, end_analysis = extend_last_note_to_fill_measure(end, score_analysis[end])

    write_inner_voices(beginning, beginning_analysis)
    write_inner_voices(middle, middle_analysis)
    write_inner_voices(middle2, middle2_analysis)
    write_inner_voices(end, end_analysis)

    middle = transpose_score(middle, -5)
    middle2 = transpose_score(middle2, -5)

    final = combine_two_scores(beginning, middle)
    final = combine_two_scores(final, middle2)
    final = combine_two_scores(final, end)
    final.show()

def stitch_I_IV_V_I(score_analysis_starts, score_analysis_ends, score_analysis):
    beginning = None
    middle = None
    middle2 = None
    end = None

    while beginning == middle or beginning == middle2 or beginning == end or \
            middle == middle2 or middle == end or middle2 == end:
        print("sampling possibilities...")
        beginning = random.sample(score_analysis_ends['I'], 1)[0]

        possible_starts_for_middle = POSSIBLE_STARTS_ENDING_FROM_TONIC["IV"]
        possible_scores = [
            score for possible_start in possible_starts_for_middle for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["I"]
        ]
        # print(len(possible_scores))
        middle = random.sample(possible_scores, 1)[0]

        possible_starts_for_middle2 = POSSIBLE_STARTS_ENDING_FROM_TONIC["ii"]
        possible_scores = [
            score for possible_start in possible_starts_for_middle2 for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["i"]
        ]
        # print(len(possible_scores))
        middle2 = random.sample(possible_scores, 1)[0]

        possible_starts_for_end = POSSIBLE_STARTS_ENDING_FROM_TONIC["IV"]
        possible_scores = [
            score for possible_start in possible_starts_for_end for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["I"]
        ]
        # print(len(possible_scores))
        end = random.sample(possible_scores, 1)[0]

    beginning, beginning_analysis = extend_last_note_to_fill_measure(beginning, score_analysis[beginning])
    middle, middle_analysis = extend_last_note_to_fill_measure(middle, score_analysis[middle])
    middle2, middle2_analysis = extend_last_note_to_fill_measure(middle2, score_analysis[middle2])
    end, end_analysis = extend_last_note_to_fill_measure(end, score_analysis[end])

    write_inner_voices(beginning, beginning_analysis)
    write_inner_voices(middle, middle_analysis)
    write_inner_voices(middle2, middle2_analysis)
    write_inner_voices(end, end_analysis)

    middle = transpose_score(middle, 5)
    middle2 = transpose_score(middle2, 7)

    final = combine_two_scores(beginning, middle)
    final = combine_two_scores(final, middle2)
    final = combine_two_scores(final, end)
    final.show()

def stitch_i_VI_iv_i(score_analysis_starts, score_analysis_ends, score_analysis):
    beginning = None
    middle = None
    middle2 = None
    end = None

    while beginning == middle or beginning == middle2 or beginning == end or \
            middle == middle2 or middle == end or middle2 == end:
        print("sampling possibilities...")
        beginning = random.sample(score_analysis_ends['i'], 1)[0]

        possible_starts_for_middle = POSSIBLE_STARTS_ENDING_FROM_TONIC["VI"]
        possible_scores = [
            score for possible_start in possible_starts_for_middle for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["I"]
        ]
        # print(len(possible_scores))
        middle = random.sample(possible_scores, 1)[0]

        possible_starts_for_middle2 = POSSIBLE_STARTS_ENDING_FROM_TONIC["vi"]
        possible_scores = [
            score for possible_start in possible_starts_for_middle2 for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["i"]
        ]
        # print(len(possible_scores))
        middle2 = random.sample(possible_scores, 1)[0]

        possible_starts_for_end = POSSIBLE_STARTS_ENDING_FROM_TONIC["V"]
        possible_scores = [
            score for possible_start in possible_starts_for_end for score in score_analysis_starts[possible_start]
            if score in score_analysis_ends["i"]
        ]
        # print(len(possible_scores))
        end = random.sample(possible_scores, 1)[0]

    beginning, beginning_analysis = extend_last_note_to_fill_measure(beginning, score_analysis[beginning])
    middle, middle_analysis = extend_last_note_to_fill_measure(middle, score_analysis[middle])
    middle2, middle2_analysis = extend_last_note_to_fill_measure(middle2, score_analysis[middle2])
    end, end_analysis = extend_last_note_to_fill_measure(end, score_analysis[end])

    write_inner_voices(beginning, beginning_analysis)
    write_inner_voices(middle, middle_analysis)
    write_inner_voices(middle2, middle2_analysis)
    write_inner_voices(end, end_analysis)

    middle = transpose_score(middle, -4)
    middle2 = transpose_score(middle2, -7)

    final = combine_two_scores(beginning, middle)
    final = combine_two_scores(final, middle2)
    final = combine_two_scores(final, end)
    final.show()


def get_organized_phrases():
    score_analyses = {}
    score_analysis_starts = defaultdict(list)
    score_analysis_ends = defaultdict(list)
    for j in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]:
        # for j in [11]:
        for i in range(1, 101):
            try:
                score, analysis = analyze_entire_phrase(f"phrase_stitching/diffusion_output/output_graphs_{j}/output_graph_{i}.xml")
                check_illegal_harmonics_on_integer_beats(score)
                check_bad_mode_mixture(score)
                check_bad_counterpoint(score)
                # print(analysis)

                score_analyses[score] = analysis
                score_analysis_starts[analysis[0]].append(score)
                score_analysis_ends[analysis[-1]].append(score)
            except (InvalidAnalysisException, FileNotFoundError) as e:
                continue
    return score_analysis_starts, score_analysis_ends, score_analyses


def get_structure():
    possible_structure_functions = [
        stitch_I_V_I,
        stitch_i_III_V_i,
        stitch_I_IV_V_I,
        stitch_i_VI_iv_i,
        stitch_i_III_iv_i
    ]
    return random.sample(possible_structure_functions, 1)[0]


def main():
    from pprint import pprint

    score_analysis_starts, score_analysis_ends, score_analyses = get_organized_phrases()
    # with open("score_info.pkl", "wb") as f:
    #     pickle.dump((score_analysis_starts, score_analysis_ends, score_analyses), f)

    # stitch_I_V_I(score_analysis_starts, score_analysis_ends, score_analyses)
    # stitch_i_III_V_i(score_analysis_starts, score_analysis_ends, score_analyses)
    stitch_i_III_iv_i(score_analysis_starts, score_analysis_ends, score_analyses)
    # stitch_I_IV_V_I(score_analysis_starts, score_analysis_ends, score_analyses)
    # stitch_i_VI_iv_i(score_analysis_starts, score_analysis_ends, score_analyses)
    # success = False
    # while not success:
    #     try:
    #         stitch_I_V_I(score_analysis_starts, score_analysis_ends)
    #         success = True
    #     except ValueError as e:
    #         print(e)
    #         continue


if __name__ == "__main__":
    main()
