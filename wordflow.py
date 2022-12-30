from pyannote.audio import Pipeline
from pydub import AudioSegment
import argparse
import logging
import os
import re
import whisper

from output import Output, OutputLine

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
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=PYANNOTE_TOKEN)
        self.logger.info("Initializing the whisper model pipeline...")
        self.whisper_model = whisper.load_model(args.model)
        self.speaker_segments = []
        self.finished = False
        self.clean_substitution_map = {
            "gotcha": "got you",
            "gonna": "going to",
            "wanna": "want to",
            "cause": "because",
            "yeah": "yes",
            "yep": "yes",
            "yup": "yes",
            "mmhmm": "yes",
            "alright": "all right",
            "ok": "okay",
        }
        self.special_substitution_map = {
            "OKAY": "Okay",
            "OK": "okay",
            "!": ".",
        }
        self.output = Output()

        if args.speakers:
            self.logger.info("Assuming speaker assignments:")
            for i, name in enumerate(args.speakers):
              self.logger.info("SPEAKER_{:01.0f} -> {}".format(i, name))

    def diaritize(self, input_file, num_speak = 1):
        self.logger.info("Running diarization...")
        diarization = self.diarization_pipeline(input_file)
        self.logger.info("Finished diarization")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_seconds = turn.start
            start_hours = start_seconds // 3600
            start_minutes = (start_seconds % 3600) // 60
            start_remaining_seconds = start_seconds % 60
            end_seconds = turn.end
            end_hours = end_seconds // 3600
            end_minutes = (end_seconds % 3600) // 60
            end_remaining_seconds = end_seconds % 60
            # Check to see if we have any speaker names provided
            if self.args.speakers:
                # Get the index of the speaker, using the last two digits of the string
                speaker_id = int(speaker[-1:])
                if speaker_id > len(self.args.speakers)-1:
                    self.logger.warn("Speaker ID {} was larger than the list of provided speakers!".format(speaker_id))
                else:
                    speaker = self.args.speakers[speaker_id]
            self.speaker_segments.append({"start": int(turn.start), "end": int(turn.end), "speaker": speaker})
            print("[{:02.0f}:{:02.0f}:{:02.0f}] -> [{:02.0f}:{:02.0f}:{:02.0f}]: {}".format(start_hours, start_minutes, start_remaining_seconds, end_hours, end_minutes, end_remaining_seconds, speaker))
    
    def create_wav(self, input_file):
        mp3_file = AudioSegment.from_mp3(input_file)
        # fine the basename of the file (remove the extension)
        input_filename = os.path.basename(input_file)
        input_basename, _ = os.path.splitext(input_filename)
        input_dir = os.path.dirname(input_file)
        output_file = os.path.join(input_dir, input_basename + ".wav")
        mp3_file.export(output_file, format="wav")
        self.logger.info("Created {}".format(output_file))
        return output_file

    def destroy_wav(self, filepath):
        self.logger.info("Removing {}".format(filepath))
        os.remove(filepath)

    def transcribe(self, input_file):
        self.logger.info("Running transcription...")
        
        options = whisper.DecodingOptions(without_timestamps=False)
        result = self.whisper_model.transcribe(input_file)

        self.logger.info("Finished transcription")
        
        # Build the output object
        for segment in result["segments"]:
            start_seconds = segment["start"]
            start_hours = start_seconds // 3600
            start_minutes = (start_seconds % 3600) // 60
            start_remaining_seconds = start_seconds % 60
            end_seconds = segment["end"]
            end_hours = end_seconds // 3600
            end_minutes = (end_seconds % 3600) // 60
            end_remaining_seconds = end_seconds % 60
            text = segment["text"]
            # Find the current speaker from the diarization table, with a bit of margin since the segment times might be slightly different
            speaker = self.lookup_speaker(start_seconds + SPEAKER_LOOKUP_MARGIN_S)
            # If we're using clean verbatim, apply substitutions to clean up the style
            if not self.args.fullverbatim:
                text = self.clean_substitutions(text)
            text = self.full_substitutions(text)
            # Add the compiled data to the output object
            self.output.add_line(start_hours, start_minutes, start_remaining_seconds, end_hours, end_minutes, end_remaining_seconds, speaker, text)

        # Ensure the run-on sentences are combined correctly
        self.output.combine_sentences()

    def lookup_speaker(self, time_s):
        for segment in self.speaker_segments:
            # Check to see if we're in a later segment
            if(time_s > segment["end"]):
                continue
            if(time_s <= segment["end"] and time_s >= segment["start"]):
                return segment["speaker"]
        # Handle case when there's a gap, check to see if the surrounding speaker is the same, as just assume it was that same person
        # TODO: This will still fail if the speaker changes during the gap
        if(self.lookup_speaker(time_s - 5) == self.lookup_speaker(time_s + 5)):
            if self.args.verbose:
                self.logger.warn("Gap in speaker data at time {}, using neighboring speaker".format(time_s))
            return self.lookup_speaker(time_s - 5)
        return None

    # This method handles exact replacements of strings that are special cases
    def full_substitutions(self, text: str) -> str:
        for old_word, new_word in self.special_substitution_map.items():
            text = text.replace(old_word, new_word)
        return text

    # This method handles replacement strings that maintain capitialization and punctuation
    def clean_substitutions(self, text: str) -> str:
        new_text = text
        for old_word, new_word in self.clean_substitution_map.items():
            new_text = self.replace_word(new_text, old_word, new_word)
        if new_text != text and self.args.verbose:
            self.logger.info("Replacement: {} -> {}".format(old_word, new_word))
        return new_text

    def replace_word(self, sentence: str, old_word: str, new_word: str) -> str:
        # Split the sentence into a list of words
        words = re.split(r'(\W+)', sentence)
        # Go through the list of words and replace the old word with the new word
        # while maintaining the original capitalization and punctuation
        for i, word in enumerate(words):
            if word.lower() == old_word.lower():
                if word.isupper():
                    words[i] = new_word.upper()
                elif word[0].isupper():
                    words[i] = new_word.capitalize()
                else:
                    words[i] = new_word
        # Join the list of words back into a single string and return it
        return "".join(words)

    def run(self):
        wav_filepath = self.create_wav(args.input)
        self.diaritize(wav_filepath)
        self.destroy_wav(wav_filepath)
        self.transcribe(args.input)
        self.finished = True

    def dump_output(self, timestamps = False):
        if self.finished:
            self.output.print(timestamps)
        else:
            self.logger.error("You must call run() first")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="The audio input file")
    parser.add_argument("-m", "--model", default="medium.en", help="OpenAI Whisper model to use (tiny[.en], base[.en], small[.en], medium[.en], large)")
    parser.add_argument("-f", "--fullverbatim", action=argparse.BooleanOptionalAction, help="Use Full Verbatim instead of default Clean Verbatim")
    parser.add_argument("-t", "--timestamps", action=argparse.BooleanOptionalAction, help="Include timestamps in output")
    parser.add_argument("-s", "--speakers", nargs="*", help="Speaker names, if available")
    parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction, help="Show verbose debug information")
    args = parser.parse_args()

    word_flow = WordFlow(args)
    word_flow.run()
    word_flow.dump_output(args.timestamps)
