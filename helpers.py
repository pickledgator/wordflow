import string

def ends_in_punctuation(s: str, ignore_comma = False) -> bool:
    # remove leading and trailing whitespace
    s = s.strip()
    if s[-1] in string.punctuation:
        if ignore_comma and s[-1] == ",":
            return False
        return True
    return False
