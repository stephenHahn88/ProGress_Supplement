from music21.stream import Score
from music21 import converter, interval, stream, note, meter

from pprint import pprint
import os


def load_scores(folder_path: str) -> dict[str:Score]:
    musicxml_streams = {}
    for path in os.listdir(folder_path):
        full_path = f"{folder_path}/{path}"

        score = converter.parse(full_path)
        musicxml_streams[full_path] = score

    return musicxml_streams


def get_simultaneous_notes(part):
    """
    Returns a list of (offset, note) tuples for all non-rest notes,
    including tied notes only at the start of the tie.
    """
    notes = []
    for n in part.recurse().notes:
        if isinstance(n, note.Note) and (not n.tie or n.tie.type == 'start'):
            notes.append((n.offset, n))
    return notes


def has_illegal_parallels(score: stream.Score, print_info=False):
    """
    Find parallel fifths and octaves between the top two parts in a score.
    """
    p1_notes = get_simultaneous_notes(score.parts[0])
    p2_notes = get_simultaneous_notes(score.parts[1])

    # Create offset-indexed dicts
    p1_dict = dict(p1_notes)
    p2_dict = dict(p2_notes)

    # Find all offsets where both parts have a note
    common_offsets = sorted(set(p1_dict.keys()).intersection(p2_dict.keys()))

    for i in range(len(common_offsets) - 1):
        off1 = common_offsets[i]
        off2 = common_offsets[i + 1]

        n1a = p1_dict[off1]
        n2a = p2_dict[off1]
        n1b = p1_dict[off2]
        n2b = p2_dict[off2]

        # Skip if either voice doesn't move
        if n1a.pitch == n1b.pitch or n2a.pitch == n2b.pitch:
            continue

        intv1 = interval.Interval(n1a, n2a).simpleName
        intv2 = interval.Interval(n1b, n2b).simpleName

        if intv1 == intv2 and intv1 in ['P5', 'P8', 'P4']:
            if print_info:
                print(f"Parallel {intv1} from offset {off1} -> {off2}:")
                print(f"  {n1a.nameWithOctave} - {n2a.nameWithOctave} -> {n1b.nameWithOctave} - {n2b.nameWithOctave}")
            return True


def check_harmonic_intervals(score, illegal_intervals=None):
    """
    Check for illegal harmonic intervals in two-voice counterpoint.

    Args:
        score: music21.stream.Score object
        illegal_intervals: set of interval names to flag as illegal
                         (default: commonly prohibited intervals in counterpoint)

    Returns:
        list of dictionaries containing information about illegal intervals
    """

    # Default illegal intervals (commonly prohibited in strict counterpoint)
    if illegal_intervals is None:
        illegal_intervals = {
            # 'P1',  # unison
            'A1',  # augmented unison
            'd2',  # diminished second
            'm2',  # minor second
            'M2',  # major second
            'A2',  # augmented second
            'd3',  # diminished third
            'A3',  # augmented third
            'A4',  # augmented fourth (tritone)
            'd5',  # diminished fifth (tritone)
            'A5',  # augmented fifth
            'd6',  # diminished sixth
            'A6',  # augmented sixth
            'd7',  # diminished seventh
            'M7',  # major seventh
            'A7',  # augmented seventh
            'd8',  # diminished octave
            'A8'  # augmented octave
        }

    # Extract the two parts
    parts = score.parts
    if len(parts) != 2:
        raise ValueError(f"Expected exactly 2 parts, got {len(parts)}")

    part1, part2 = parts[0], parts[1]

    # Get all notes with their absolute offsets
    notes1 = []
    notes2 = []

    for note in part1.flat.notes:
        if hasattr(note, 'pitch'):  # Single note
            notes1.append((note.offset, note))
        elif hasattr(note, 'pitches'):  # Chord
            for pitch in note.pitches:
                # Create a temporary note for each pitch in chord
                temp_note = note.Note(pitch, quarterLength=note.quarterLength)
                notes1.append((note.offset, temp_note))

    for note in part2.flat.notes:
        if hasattr(note, 'pitch'):  # Single note
            notes2.append((note.offset, note))
        elif hasattr(note, 'pitches'):  # Chord
            for pitch in note.pitches:
                # Create a temporary note for each pitch in chord
                temp_note = note.Note(pitch, quarterLength=note.quarterLength)
                notes2.append((note.offset, temp_note))

    # Group notes by onset time
    onset_times = set()
    for offset, _ in notes1 + notes2:
        onset_times.add(offset)

    # Find simultaneous note onsets and check intervals
    illegal_intervals_found = []

    for onset_time in sorted(onset_times):
        # Get all notes starting at this onset time
        notes_at_onset_part1 = [note for offset, note in notes1 if offset == onset_time]
        notes_at_onset_part2 = [note for offset, note in notes2 if offset == onset_time]

        # Check intervals between all combinations of simultaneous notes
        for note1 in notes_at_onset_part1:
            for note2 in notes_at_onset_part2:
                # Calculate interval
                try:
                    interval_obj = interval.Interval(note1, note2)
                    interval_name = interval_obj.name

                    # Reduce compound intervals to simple intervals for checking
                    simple_interval_name = get_simple_interval_name(interval_obj)

                    # Check if interval is illegal (using simple interval)
                    if simple_interval_name in illegal_intervals:
                        illegal_intervals_found.append({
                            'onset_time': onset_time,
                            'interval': interval_name,
                            'simple_interval': simple_interval_name,
                            'interval_semitones': interval_obj.semitones,
                            'note1': {
                                'pitch': note1.pitch.name,
                                'octave': note1.pitch.octave,
                                'midi': note1.pitch.midi,
                                'offset': onset_time,
                                'duration': note1.quarterLength
                            },
                            'note2': {
                                'pitch': note2.pitch.name,
                                'octave': note2.pitch.octave,
                                'midi': note2.pitch.midi,
                                'offset': onset_time,
                                'duration': note2.quarterLength
                            },
                            'measure': get_measure_number(score, onset_time)
                        })
                except Exception as e:
                    print(f"Error calculating interval at onset {onset_time}: {e}")

    return illegal_intervals_found


def get_simple_interval_name(interval_obj):
    """
    Convert compound intervals to simple intervals for checking purposes.
    9th -> 2nd, 10th -> 3rd, etc.
    """
    # Get the generic interval number (1, 2, 3, etc.)
    generic_num = interval_obj.generic.undirected

    # Reduce compound intervals to simple (within an octave)
    simple_generic = ((generic_num - 1) % 7) + 1

    # Get the quality from the interval name
    interval_name = interval_obj.name
    quality_str = ""

    # Extract quality characters (everything before the number)
    for char in interval_name:
        if char.isdigit():
            break
        quality_str += char

    return f"{quality_str}{simple_generic}"


def get_measure_number(score, offset):
    """Helper function to get measure number for a given offset"""
    try:
        measure = score.flat.getElementsByOffset(offset,
                                                 classList=[stream.Measure])
        if measure:
            return measure[0].number
        else:
            # Fallback: estimate measure number
            time_sig = score.flat.getTimeSignatures()[0] if score.flat.getTimeSignatures() else meter.TimeSignature(
                '4/4')
            measure_length = time_sig.numerator * (4.0 / time_sig.denominator)
            return int(offset // measure_length) + 1
    except:
        return None


def print_illegal_intervals(illegal_intervals):
    """Pretty print the illegal intervals found"""
    if not illegal_intervals:
        print("No illegal harmonic intervals found!")
        return

    print(f"Found {len(illegal_intervals)} illegal harmonic interval(s):")
    print("-" * 80)

    for violation in illegal_intervals:
        print(f"Measure {violation['measure']}, Onset: {violation['onset_time']}:")
        print(f"  Actual Interval: {violation['interval']} ({violation['interval_semitones']} semitones)")
        print(f"  Simple Interval: {violation['simple_interval']} (flagged as illegal)")
        print(f"  Notes: {violation['note1']['pitch']}{violation['note1']['octave']} - "
              f"{violation['note2']['pitch']}{violation['note2']['octave']}")
        print(f"  MIDI: {violation['note1']['midi']} - {violation['note2']['midi']}")
        print(f"  Note 1: offset={violation['note1']['offset']}, duration={violation['note1']['duration']}")
        print(f"  Note 2: offset={violation['note2']['offset']}, duration={violation['note2']['duration']}")
        print()


# Example usage with your score
def analyze_score(score_file_path=None, score_object=None, print_info=False):
    """
    Analyze a score for illegal harmonic intervals
    """
    if score_file_path:
        score = converter.parse(score_file_path)
    elif score_object:
        score = score_object
    else:
        raise ValueError("Must provide either score_file_path or score_object")

    # Check for illegal intervals
    illegal_intervals = check_harmonic_intervals(score)

    # Print results
    if print_info:
        print_illegal_intervals(illegal_intervals)

    return illegal_intervals


if __name__ == "__main__":
    scores = load_scores("./diffusion_output")
    # pprint(scores)

    all_good = []
    for path, score in scores.items():
        # if path == "./diffusion_output/output_graph_11.xml":
        #     score.show("text")
        #     break
        # print(path)
        illegal_intervals = analyze_score(path)
        illegal_parallels = has_illegal_parallels(score)
        if not (illegal_parallels or illegal_intervals):
            all_good.append(path)
    pprint(all_good)
    print(len(all_good))
    # Example usage:
    # If you have a MusicXML file:

    # illegal_parallels = has_illegal_parallels('./diffusion_output/output_graph_5.xml')

    # If you have a music21 score object:
    # illegal_intervals = analyze_score(score_object=your_score)

    # You can also customize which intervals are considered illegal:
    # custom_illegal = {'P1', 'A4', 'd5'}  # Only unisons and tritones
    # illegal_intervals = check_harmonic_intervals(score, custom_illegal)
