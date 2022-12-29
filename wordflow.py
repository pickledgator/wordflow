from pyannote.audio import Pipeline
from pydub import AudioSegment
import argparse
import logging
import os
import whisper

PYANNOTE_TOKEN="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"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s.%(msecs)03d UTC: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

class WordFlow:
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger("WordFlow")
        self.logger.info("Initializing the pyannote diarization model pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=PYANNOTE_TOKEN)
        self.logger.info("Initializing the whisper model pipeline...")
        self.whisper_model = whisper.load_model(args.model)

    def diaritize(self, input_file, num_speak = 1):
        self.logging.info("Running diarization...")
        diarization = self.diarization_pipeline(input_file, num_speakers = num_speak)
        self.logging.info("Finished diarization")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    
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
        result = self.whisper_model.transcribe(input_file)
        self.logger.info("Finished transcription")
        for segment in result["segments"]:
            seconds = segment["start"]
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            remaining_seconds = seconds % 60
            text = segment["text"]
            print("[{:02.0f}:{:02.0f}:{:02.0f}]: {}".format(hours, minutes, remaining_seconds, text))

    def run(self):
        # wav_filepath = self.create_wav(args.input)
        # self.diaritize(wav_filepath)
        # self.destroy_wav(wav_filepath)
        self.transcribe(args.input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="The audio input file")
    parser.add_argument("-m", "--model", default="medium.en", help="OpenAI Whisper model to use (tiny[.en], base[.en], small[.en], medium[.en], large)")
    args = parser.parse_args()

    word_flow = WordFlow(args)
    word_flow.run()
