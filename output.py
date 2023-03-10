from helpers import ends_in_punctuation

class OutputSegment:
    def __init__(self, start, end, speaker, text):
        self.start = start
        self.end = end
        self.speaker = speaker
        self.text = text

class Output:
    segments = []

    def __init__(self, logger):
        self.logger = logger

    def add_segment(self, start, end, speaker, text):
        self.segments.append(OutputSegment(start, end, speaker, text))

    # def combine_runons(self):
    #     lines_to_delete = []
    #     for i, output_line in enumerate(reversed(self.output)):
    #         index = len(self.output) - 1 - i
    #         # Stop when we reach the last line
    #         if index == 0:
    #             break
    #         # check to see if the first letter is not capitalized, and if the last character of the previous sentence is not punctuation
    #         current_line_cap = self.output[index].text[0].isupper()
    #         prev_line_punctuated = ends_in_punctuation(self.output[index-1].text, True) # ignore_comma
    #         if(not current_line_cap and not prev_line_punctuated):
    #             self.combine_prev_line(index, index-1)
    #             # record the id so we can delete the line later
    #             lines_to_delete.append(index)
        
    #     # Remove the lines that were combined
    #     self.output = [item for i, item in enumerate(self.output) if i not in lines_to_delete]

    def combine_same_speaker_sentences(self, max_words: int):
        paragraph_words = 0
        paragraph_start_index = 0
        lines_to_delete = []
        for index, segment in enumerate(self.segments):
            # Skip the first line
            if index == 0:
                continue

            # Check to see if the speakaer matches the previous line
            if segment.speaker == self.segments[index-1].speaker:
                # Add the words from this line to the counter
                paragraph_words += len(segment.text.split())
                # If the number of words is still less than the max_words, combine it
                if paragraph_words < max_words:
                    self.combine_prev_line(index, paragraph_start_index)
                    lines_to_delete.append(index)
                else:
                    # Once we've exceeded the max words, update the paragraph_start_index for the next paragraph
                    paragraph_start_index = index
                    # Ensure the next line word counter has already included the first line in the paragraph
                    paragraph_words = len(segment.text.split())
            else:
                # If the speaker changes, reset things
                paragraph_start_index = index
                paragraph_words = len(segment.text.split())
        
        # Remove the lines that were combined
        self.segments = [item for i, item in enumerate(self.segments) if i not in lines_to_delete]


    def combine_prev_line(self, index, prev_index):
        self.logger.info("Combining line {} with previous line".format(index))
        # Append the text of the current line onto the end of the previous line
        self.segments[prev_index].text += self.segments[index].text
        # Update the end time of the previous line to the end time of the current line
        self.segments[prev_index].end = self.segments[index].end

    def print(self, timestamps = False):
        print("======================================")
        for segment in self.segments:

            # Convert the timestamp seconds output by the model into hours, minutes and seconds
            start_seconds = segment.start
            start_hours = start_seconds // 3600
            start_minutes = (start_seconds % 3600) // 60
            start_remaining_seconds = start_seconds % 60
            end_seconds = segment.end
            end_hours = end_seconds // 3600
            end_minutes = (end_seconds % 3600) // 60
            end_remaining_seconds = end_seconds % 60

            str = ""
            if timestamps:
                str = "[{:02.0f}:{:02.0f}:{:02.0f}] -> [{:02.0f}:{:02.0f}:{:02.0f}] {}: {}".format(start_hours, start_minutes, start_remaining_seconds, end_hours, end_minutes, end_remaining_seconds, segment.speaker, segment.text)
            else:
                str = "{}: {}".format(segment.speaker, segment.text)

            print(str)
