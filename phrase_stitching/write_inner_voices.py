import copy

from music21 import converter, interval, note, stream, meter, key
from phrase_stitching.RN_analysis import get_vertical_pairs, analyze_entire_phrase, COMMON_ROMAN_NUMERALS_C


IDEAL_DOUBLING = {
    "I": "root",
    "i": "root",
    "ii": "third",
    "iio": "third",
    "iii": "root",
    "III": "root",
    "IV": "root",
    "iv": "root",
    "V": "root",
    "v": "root",
    "vi": "root",
    "VI": "root",
    "viio": "third",
    "VII": "root"
}


def get_likely_inner_voices(score, analysis):
    vertical_pairs = get_vertical_pairs(score)
    inner_voice_notes = []
    curr_beat = 0
    while curr_beat < max([float(k) for k in vertical_pairs.keys()]):
        curr_RN = analysis[curr_beat]
        harmony_notes = COMMON_ROMAN_NUMERALS_C[curr_RN]
        already_in_score = {note: False for note in harmony_notes}
        notes_in_beat = set([note for time, val in vertical_pairs.items() for note in val
                             if curr_beat - 0.01 <= float(time) < curr_beat + 1])
        for note in notes_in_beat:
            if note in already_in_score.keys():
                already_in_score[note] = True

        inner_notes = [note for note, in_there in already_in_score.items() if not in_there]
        while len(inner_notes) < 2:
            if IDEAL_DOUBLING[curr_RN] == "root":
                inner_notes.append(COMMON_ROMAN_NUMERALS_C[curr_RN][0])
            elif IDEAL_DOUBLING[curr_RN] == "third":
                inner_notes.append(COMMON_ROMAN_NUMERALS_C[curr_RN][1])
        if len(inner_notes) > 2:
            if "F" in inner_notes:
                inner_notes.remove("F")
            elif "B" in inner_notes:
                inner_notes.remove("B")


        inner_voice_notes.append(inner_notes)

        curr_beat += 1
    return inner_voice_notes


def assign_voices(note_pairs, score):
    vertical_pairs = get_vertical_pairs(score, include_octave=True)
    outer_pairs = [pair for k, pair in vertical_pairs.items() if float(k) % 1 == 0]
    voice1_notes = []
    voice2_notes = []

    prev1 = None
    prev2 = None

    possible_octaves = [3, 4]
    for n1_name, n2_name in note_pairs:
        options = [
            (note.Note(f"{n1_name}{octave}"), note.Note(f"{n2_name}{octave}")) for octave in possible_octaves
        ] + [
            (note.Note(f"{n2_name}{octave}"), note.Note(f"{n1_name}{octave}")) for octave in possible_octaves
        ]

        min_total_distance = None
        best_option = None

        for opt1, opt2 in options:
            dist1 = interval.Interval(prev1, opt1).semitones if prev1 else 0
            dist2 = interval.Interval(prev2, opt2).semitones if prev2 else 0
            total_dist = abs(dist1) + abs(dist2)

            if min_total_distance is None or total_dist < min_total_distance:
                min_total_distance = total_dist
                best_option = (opt1, opt2)

        chosen1, chosen2 = best_option
        chosen1.quarterLength = 1
        chosen2.quarterLength = 1

        voice1_notes.append(chosen1)
        voice2_notes.append(chosen2)

        prev1 = chosen1
        prev2 = chosen2

    for v1_note, v2_note, (treble, bass) in zip(voice1_notes, voice2_notes, outer_pairs):
        treble_note = note.Note(treble)
        bass_note = note.Note(bass)
        v1_interval_to_treble = interval.Interval(-1)
        v2_interval_to_treble = interval.Interval(-1)
        v1_interval_to_bass = interval.Interval(1)
        v2_interval_to_bass = interval.Interval(1)

        stuck = 0
        while v1_interval_to_treble.semitones < 0 or v1_interval_to_bass.semitones > 0 or \
                v2_interval_to_treble.semitones < 0 or v2_interval_to_bass.semitones > 0:
            v1_interval_to_treble = interval.Interval(v1_note, treble_note)
            v2_interval_to_treble = interval.Interval(v2_note, treble_note)
            v1_interval_to_bass = interval.Interval(v1_note, bass_note)
            v2_interval_to_bass = interval.Interval(v2_note, bass_note)

            if v1_interval_to_bass.semitones > 0:
                v1_note.octave = v1_note.octave + 1
            if v2_interval_to_bass.semitones > 0:
                v2_note.octave = v2_note.octave + 1
            if v1_interval_to_treble.semitones < 0:
                v1_note.octave = v1_note.octave - 1
            if v2_interval_to_treble.semitones < 0:
                v2_note.octave = v2_note.octave - 1

            stuck += 1
            if stuck > 5:
                break



    return voice1_notes, voice2_notes


def merge_repeats_in_measure(measure):
    new_measure = stream.Measure(number=measure.number)
    last_note = None
    for element in measure.notesAndRests:
        if isinstance(element, note.Note):
            if last_note and element.nameWithOctave == last_note.nameWithOctave:
                last_note.quarterLength += element.quarterLength
            else:
                last_note = copy.deepcopy(element)
                new_measure.append(last_note)
        else:
            last_note = None
            new_measure.append(element)

    return new_measure


def merge_repeats_by_measure(part):
    new_part = stream.Part()
    for el in part.recurse():
        if isinstance(el, stream.Measure):
            merged_measure = merge_repeats_in_measure(el)
            new_part.append(merged_measure)
        elif isinstance(el, (meter.TimeSignature, key.KeySignature)):
            new_part.insert(el.offset, el)
    return new_part


def write_inner_voices(score, analysis):
    inner_voices = get_likely_inner_voices(score, analysis)
    # print(inner_voices)
    voice1, voice2 = assign_voices(inner_voices, score)

    part1 = stream.Part()
    part2 = stream.Part()

    time_sig = meter.TimeSignature('4/4')
    part1.append(time_sig)
    part2.append(time_sig)

    for n in voice1:
        part1.append(n)
    for n in voice2:
        part2.append(n)

    part1 = part1.makeMeasures()
    part2 = part2.makeMeasures()

    part1 = merge_repeats_by_measure(part1)
    part2 = merge_repeats_by_measure(part2)

    score.insert(0, part1)
    score.insert(0, part2)


def main():
    filepath = "./diffusion_output/output_graphs_8/output_graph_98.xml"
    score, analysis = analyze_entire_phrase(filepath)
    write_inner_voices(score, analysis)


if __name__ == "__main__":
    main()




