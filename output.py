from helpers import ends_in_punctuation

class OutputLine:
    def __init__(self, start_hours, start_minutes, start_seconds, end_hours, end_minutes, end_seconds, speaker, text):
        self.start_hours = start_hours
        self.start_minutes = start_minutes
        self.start_seconds = start_seconds
        self.end_hours = end_hours
        self.end_minutes = end_minutes
        self.end_seconds = end_seconds
        self.speaker = speaker
        self.text = text

class Output:
    output = []

    def __init__(self, logger):
        self.logger = logger

    def add_line(self, start_hours, start_minutes, start_seconds, end_hours, end_minutes, end_seconds, speaker, text):
        self.output.append(OutputLine(start_hours, start_minutes, start_seconds, end_hours, end_minutes, end_seconds, speaker, text))

    def combine_sentences(self):
        lines_to_delete = []
        for i, output_line in enumerate(reversed(self.output)):
            index = len(self.output) - 1 - i
            # Stop when we reach the last line
            if index == 0:
                break
            # check to see if the first letter is not capitalized, and if the last character of the previous sentence is not punctuation
            current_line_cap = self.output[index].text[0].isupper()
            prev_line_punctuated = ends_in_punctuation(self.output[index-1].text, True) # ignore_comma
            if(not current_line_cap and not prev_line_punctuated):
                self.logger.info("Combining line {} with previous line".format(index))
                # Append the text of the current line onto the end of the previous line
                self.output[index-1].text += self.output[index].text
                # Update the end time of the previous line to the end time of the current line
                self.output[index-1].end_hours = self.output[index].end_hours
                self.output[index-1].end_minutes = self.output[index].end_minutes
                self.output[index-1].end_seconds = self.output[index].end_seconds
                # record the id so we can delete the line later
                lines_to_delete.append(index)
        # Remove the lines that were combined
        for id in lines_to_delete:
            self.output.pop(id)

    def print(self, timestamps = False):
        print("======================================")
        for output_line in self.output:
            str = ""
            if timestamps:
                str = "[{:02.0f}:{:02.0f}:{:02.0f}] -> [{:02.0f}:{:02.0f}:{:02.0f}] {}: {}".format(output_line.start_hours, output_line.start_minutes, output_line.start_seconds, output_line.end_hours, output_line.end_minutes, output_line.end_seconds, output_line.speaker, output_line.text)
            else:
                str = "{}: {}".format(output_line.speaker, output_line.text)
            print(str)
