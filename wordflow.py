from pyannote.audio import Pipeline
from pydub import AudioSegment
import argparse
import logging
import os
import whisper

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

    def diaritize(self, input_file, num_speak = 1):
        self.logger.info("Running diarization...")
        diarization = self.diarization_pipeline(input_file)
        self.logger.info("Finished diarization")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            self.speaker_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
            start_seconds = turn.start
            start_hours = start_seconds // 3600
            start_minutes = (start_seconds % 3600) // 60
            start_remaining_seconds = start_seconds % 60
            end_seconds = turn.end
            end_hours = end_seconds // 3600
            end_minutes = (end_seconds % 3600) // 60
            end_remaining_seconds = end_seconds % 60
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
        # result = self.whisper_model.transcribe(input_file)
        audio = whisper.load_audio(input_file)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
        # Placeholder for options if we need them
        options = whisper.DecodingOptions(language="en", without_timestamps=False)
        # Run the decoder
        result = whisper.decode(self.whisper_model.device, mel, options)
        self.logger.info("Finished transcription\n\n")
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
            speaker = self.lookup_speaker((start_seconds + end_seconds) / 2.0)
            # If we're using clean verbatim, apply substitutions to clean up the style
            if not self.args.fullverbatim:
                text = self.clean_substitutions(text)
            # Print the output to stdout
            if self.args.timestamps:
                print("[{:02.0f}:{:02.0f}:{:02.0f}] -> [{:02.0f}:{:02.0f}:{:02.0f}] {}: {}".format(start_hours, start_minutes, start_remaining_seconds, end_hours, end_minutes, end_remaining_seconds, speaker, text))
            else:
                print("{}: {}".format(speaker, text))

    def lookup_speaker(self, time_s):
        for segment in self.speaker_segments:
            # Check to see if we're in a later segment
            if(time_s > segment["end"]):
                continue
            if(time_s <= segment["end"] and time_s >= segment["start"]):
                return segment["speaker"]
        return None

    def clean_substitutions(self, text):
        # Iterate over every word
        for word in text:
            # First see if the word needs a substitute
            if word in self.clean_substitution_map:
                # Next remember the capitalization of the first letter
                cap = word.isupper()
                # Next substitude
                word = self.clean_substitution_map[word.lower()]
                # Apply capitialization, if needed
                if cap:
                    word = word.capitalize()
        return text

    def run(self):
        wav_filepath = self.create_wav(args.input)
        self.diaritize(wav_filepath)
        self.destroy_wav(wav_filepath)
        self.transcribe(args.input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="The audio input file")
    parser.add_argument("-m", "--model", default="medium.en", help="OpenAI Whisper model to use (tiny[.en], base[.en], small[.en], medium[.en], large)")
    parser.add_argument("-f", "--fullverbatim", action=argparse.BooleanOptionalAction, help="Use Full Verbatim instead of default Clean Verbatim")
    parser.add_argument("-t", "--timestamps", action=argparse.BooleanOptionalAction, help="Include timestamps in output")
    args = parser.parse_args()

    word_flow = WordFlow(args)
    word_flow.run()
