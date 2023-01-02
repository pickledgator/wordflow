from pyannote.audio import Pipeline
import argparse
import logging
import os
import re
import whisper

from helpers import replace_numbers, split_wav_segments, destroy_wav_files, mp3_to_wav, replace_exact, replace_maintain_capitalization
from output import Output
from expansions import CONTRACTIONS_MAP, YES_MAP, OK_MAP, ETC_MAP, OK_EXACT_MAP, PUNCTUATION_EXACT_MAP

# Token for access the pyannote model
PYANNOTE_TOKEN="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"

# The number of seconds to subtract from the end of the segment window when trying to figure out which speaker was talking
SPEAKER_LOOKUP_MARGIN_S=1.0

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s.%(msecs)03d UTC: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

class WordFlow:
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger("WordFlow")
        self.logger.info("Initializing the pyannote diarization model pipeline...")
        if self.args.diaritize:
            self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=PYANNOTE_TOKEN)
        self.logger.info("Initializing the whisper model pipeline...")
        self.whisper_model = whisper.load_model(args.model)
        self.finished = False
        # self.output = Output(self.logger)

        if args.speakers:
            self.logger.info("Assuming speaker assignments:")
            for i, name in enumerate(args.speakers):
              self.logger.info("SPEAKER_{:01.0f} -> {}".format(i, name))
        else:
            args.speakers = ["SPEAKER_01"]

    def diaritize(self, input_file, num_speak = 1) -> list:
        self.logger.info("Running diarization...")
        diarization = self.diarization_pipeline(input_file, num_speakers=len(self.args.speakers))
        self.logger.info("Finished diarization")

        # Process the results of the diarization model
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Check to see if we have any speaker names provided
            if self.args.speakers:
                # Get the index of the speaker, using the last two digits of the string
                speaker_id = int(speaker[-1:])
                if speaker_id > len(self.args.speakers)-1:
                    self.logger.warn("Speaker ID {} was larger than the list of provided speakers!".format(speaker_id))
                else:
                    speaker = self.args.speakers[speaker_id]
            segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
            if self.args.verbose:
                print("[{:0.2f}] -> [{:0.2f}]: {}".format(turn.start, turn.end, speaker))
        return segments

    def transcribe(self, segments: list, diaritize: bool):
        self.logger.info("Running transcription...")
        
        options = whisper.DecodingOptions(
            without_timestamps=False,
        )

        output = Output(self.logger)

        for i, segment in enumerate(segments):
            # Run the transcription on the audio file
            # Encourge the model to use punctuation as a prior so it doesn't get stuck in no punctuation mode.
            self.logger.info("Transcribing segment {}".format(i))
            result = self.whisper_model.transcribe(
                audio=segment["file"], 
                initial_prompt="This is a sentence with punctuation."
            )

            # Ignore segments that don't have any output from whisper, or return this junk phrase
            # TODO: Figure out how to reduce these segments coming out of the diarization
            if result["text"] == "" or result["text"] == " " or result["text"] == " This is a sentence with punctuation.":
                continue

            output.add_segment(segment["start"], segment["end"], segment["speaker"], result["text"])

        return output
        
        
        # Build the output object
        # for segment in result["segments"]:
        #     # Convert the timestamp seconds output by the model into hours, minutes and seconds
        #     start_seconds = segment["start"]
        #     start_hours = start_seconds // 3600
        #     start_minutes = (start_seconds % 3600) // 60
        #     start_remaining_seconds = start_seconds % 60
        #     end_seconds = segment["end"]
        #     end_hours = end_seconds // 3600
        #     end_minutes = (end_seconds % 3600) // 60
        #     end_remaining_seconds = end_seconds % 60
        #     text = segment["text"]
        #     speaker = ""

        #     # Find the current speaker from the diarization table, with a bit of margin since the segment times might be slightly different
        #     if diaritize:
        #         speaker = self.lookup_speaker(start_seconds + SPEAKER_LOOKUP_MARGIN_S)
            
        #     # Apply any substitution strategies to apply specific styling to the output
        #     if self.args.expand_contractions:
        #         text = replace_maintain_capitalization(text, CONTRACTIONS_MAP)
        #     if self.args.replace_yes:
        #         text = replace_maintain_capitalization(text, YES_MAP)
        #     if self.args.replace_ok:
        #         text = replace_maintain_capitalization(text, OK_MAP)
        #     if self.args.replace_etc:
        #         text = replace_maintain_capitalization(text, ETC_MAP)
        #     if self.args.replace_ok_exact:
        #         text = replace_exact(text, OK_EXACT_MAP)
        #     if self.args.replace_punctuation_exact:
        #         text = replace_exact(text, PUNCTUATION_EXACT_MAP)
        #     if self.args.replace_numbers:    
        #         text = replace_numbers(text)
            
        #     # Add the compiled data to the output object
        #     self.output.add_line(start_hours, start_minutes, start_remaining_seconds, end_hours, end_minutes, end_remaining_seconds, speaker, text)

        # # Ensure the run-on sentences are combined correctly
        # # self.output.combine_runons()



    # def lookup_speaker(self, time_s):
    #     for segment in self.speaker_segments:
    #         # Check to see if we're in a later segment
    #         if(time_s > segment["end"]):
    #             continue
    #         if(time_s <= segment["end"] and time_s >= segment["start"]):
    #             return segment["speaker"]
    #     # Handle case when there's a gap, check to see if the surrounding speaker is the same, as just assume it was that same person
    #     # TODO: This will still fail if the speaker changes during the gap
    #     if(self.lookup_speaker(time_s - 5) == self.lookup_speaker(time_s + 5)):
    #         if self.args.verbose:
    #             self.logger.warn("Gap in speaker data at time {}, using neighboring speaker".format(time_s))
    #         return self.lookup_speaker(time_s - 5)
    #     return None

    def run(self):
        # Start out with just the main file
        audio_files = [self.args.input]

        if self.args.diaritize:
            wav_filepath = mp3_to_wav(self.args.input)
            self.logger.info("Created {}".format(wav_filepath))

            segments = self.diaritize(wav_filepath)
            self.logger.info("Identified {} segments".format(len(segments)))
            
            # If we're diarizing, overwrite the audio files with the new segment files
            segments = split_wav_segments(wav_filepath, segments)
            self.logger.info("Generated {} new wav files based on segments".format(len(segments)))

        output = self.transcribe(segments, self.args.diaritize)

        # Combine same speaker lines up to the max word count
        if self.args.combine_same_speaker_paragraphs:
            output.combine_same_speaker_sentences(self.args.max_words_same_speaker)

        output.print()

        self.logger.info("Removing all wav files")
        destroy_wav_files()

        self.finished = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="The audio input file")
    parser.add_argument("-m", "--model", default="medium.en", help="OpenAI Whisper model to use (tiny[.en], base[.en], small[.en], medium[.en], large)")
    parser.add_argument("-t", "--timestamps", action=argparse.BooleanOptionalAction, help="Include timestamps in output")
    parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction, help="Show verbose debug information")
    parser.add_argument("--diaritize", action=argparse.BooleanOptionalAction, help="Identify speakers using diaritization")
    parser.add_argument("-s", "--speakers", nargs="*", help="Speaker names, if available. Only applicable if diaritize is used")
    parser.add_argument("--replace-numbers", action=argparse.BooleanOptionalAction, help="Replace 0-9 digits with words")
    parser.add_argument("--expand-contractions", action=argparse.BooleanOptionalAction, help="Expand contractions (eg, gotcha -> got you)")
    parser.add_argument("--replace-yes", action=argparse.BooleanOptionalAction, help="Expand variations of yea/yup -> yes (maintain capitalization)")
    parser.add_argument("--replace-ok", action=argparse.BooleanOptionalAction, help="Expand variations of ok -> okay (maintain capitalization)")
    parser.add_argument("--replace-etc", action=argparse.BooleanOptionalAction, help="Expand variations of etc -> etcetra (maintain capitalization)")
    parser.add_argument("--replace-ok-exact", action=argparse.BooleanOptionalAction, help="Expand variations of OKAY -> Okay (exact match)")
    parser.add_argument("--replace-punctuation-exact", action=argparse.BooleanOptionalAction, help="Expand variations punctuation (exact match)")
    parser.add_argument("--combine-same-speaker-paragraphs", action=argparse.BooleanOptionalAction, help="Expand variations punctuation (exact match)")
    parser.add_argument("--max-words-same-speaker", default="100", type=int, help="When combine-same-speaker-paragraphs is set, this is the maximum number of words to combine into a paragraph")
    args = parser.parse_args()

    word_flow = WordFlow(args)
    word_flow.run()
