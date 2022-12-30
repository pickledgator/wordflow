import string

def ends_in_punctuation(s: str) -> bool:
    s = s.strip()  # remove leading and trailing whitespace
    if s[-1] in string.punctuation:
        return True
    return False
