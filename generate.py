from collections import defaultdict

from phrase_stitching.stitch import get_organized_phrases, get_structure


def main():
    score_analysis_starts, score_analysis_ends, score_analyses = get_organized_phrases()
    structure_function = get_structure()
    structure_function(score_analysis_starts, score_analysis_ends, score_analyses)


if __name__ == "__main__":
    main()