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

    def add_line(self, start_hours, start_minutes, start_seconds, end_hours, end_minutes, end_seconds, speaker, text):
        self.output.append(OutputLine(start_hours, start_minutes, start_seconds, end_hours, end_minutes, end_seconds, speaker, text))

    def print(self, timestamps = False):
        print()
        for output_line in self.output:
            str = ""
            if timestamps:
                str = "[{:02.0f}:{:02.0f}:{:02.0f}] -> [{:02.0f}:{:02.0f}:{:02.0f}] {}: {}".format(output_line.start_hours, output_line.start_minutes, output_line.start_seconds, output_line.end_hours, output_line.end_minutes, output_line.end_seconds, output_line.speaker, output_line.text)
            else:
                str = "{}: {}".format(output_line.speaker, output_line.text)
            print(str)
