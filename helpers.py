import re
import string

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
