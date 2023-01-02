import glob
import os
import numpy as np
import re
from pydub import AudioSegment
import struct
import string
import wave

# Replaces a word in a sentence and maintains its capitalization
def replace_word(sentence: str, old_word: str, new_word: str) -> str:
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

# This method handles exact replacements of strings that are special cases
def replace_exact(text: str, map: dict) -> str:
    for old_word, new_word in map.items():
        text = text.replace(old_word, new_word)
    return text

# This method handles replacement strings that maintain capitialization and punctuation
def replace_maintain_capitalization(text: str, map: dict) -> str:
    new_text = text
    for old_word, new_word in map.items():
        new_text = replace_word(new_text, old_word, new_word)
    return new_text

def ends_in_punctuation(s: str, ignore_comma = False) -> bool:
    # remove leading and trailing whitespace
    s = s.strip()
    if s[-1] in string.punctuation:
        if ignore_comma and s[-1] == ",":
            return False
        return True
    return False

def replace_numbers(s):
  def replace(match):
    # Check if the matched string is a dollar value
    dollar_value_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
    if re.match(dollar_value_pattern, match.group(0)):
      # If the matched string is a dollar value, return the original string
      return match.group(0)
    # Convert the matched string to an integer
    n = int(match.group(0))
    # Return the text representation of the integer, if it is not a single digit
    # prepended by "$" and followed by ","
    if not (len(match.group(0)) == 1 and match.group(0) in '123456789'
            and match.start() > 0 and s[match.start() - 1] == '$'
            and match.end() < len(s) and s[match.end()] == ','):
      if n == 0:
        return 'zero'
      elif n < 0:
        return 'minus ' + str(-1 * n)
      elif n < 10:
        # For numbers between 0 and 9, use a special mapping
        return [
          'one', 'two', 'three', 'four', 'five', 'six', 'seven',
          'eight', 'nine'
        ][n-1]
      else:
        # For numbers 10 and above, return the original string
        return match.group(0)
    # Otherwise, return the original string
    else:
      return match.group(0)
  # Use a regex to find all integers in the string
  pattern = r'\d+'
  # Replace the integers with their text representation
  return re.sub(pattern, replace, s)

def split_wav_segments(input_file: str, segments: list) -> int:
    with wave.open(input_file, "rb") as wave_file:
        # Read the wave file properties
        num_channels = wave_file.getnchannels()
        sample_width = wave_file.getsampwidth()
        frame_rate = wave_file.getframerate()
        print("Frame rate: {}".format(frame_rate))
        num_frames = wave_file.getnframes()
        # Read the wave file frames
        wave_data = wave_file.readframes(num_frames)

    # Convert wave data to a NumPy array
    data = np.array(struct.unpack_from("%dh" % num_frames * num_channels, wave_data))

    # Split the segments and write the files
    for i, segment in enumerate(segments):
        start = segment["start"] * frame_rate
        end = segment["end"] * frame_rate
        print(f"Start: {start}, End: {end}")
        segment_data = data[int(start):int(end)]
        filepath = "segment_{:02.0f}.wav".format(i+1)
        with wave.open(filepath, "wb") as wave_file:
            wave_file.setnchannels(num_channels)
            wave_file.setsampwidth(sample_width)
            wave_file.setframerate(frame_rate)
            wave_file.writeframes(struct.pack("%dh" % len(segment_data), *segment_data))
            segments["file"] = filepath
    
    return segments

def mp3_to_wav(input_file):
    mp3_file = AudioSegment.from_mp3(input_file)
    # fine the basename of the file (remove the extension)
    input_filename = os.path.basename(input_file)
    input_basename, _ = os.path.splitext(input_filename)
    input_dir = os.path.dirname(input_file)
    output_file = os.path.join(input_dir, input_basename + ".wav")
    mp3_file.export(output_file, format="wav")
    return output_file

def destroy_wav_files():
    files = glob.glob("*.wav")
    for file in files:
        os.remove(file)
