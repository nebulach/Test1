import tensorflow as tf
import collections
import unicodedata
import six
from apply_bpe import BPE
from bnlp import BnlpPy

preoccupied_vocab = ["[PAD]", "[unused1]", "[unused2]", "[unused3]", "[unused4]", "[unused5]", "[unused6]", "[unused7]", "[unused8]", "[unused9]", "[unused10]", "[unused11]", "[unused12]", "[unused13]", "[unused14]", "[unused15]", "[unused16]", "[unused17]", "[unused18]", "[unused19]", "[unused20]", "[unused21]", "[unused22]", "[unused23]", "[unused24]", "[unused25]", "[unused26]", "[unused27]", "[unused28]", "[unused29]", "[unused30]", "[unused31]", "[unused32]", "[unused33]", "[unused34]", "[unused35]", "[unused36]", "[unused37]", "[unused38]", "[unused39]", "[unused40]", "[unused41]", "[unused42]", "[unused43]", "[unused44]", "[unused45]", "[unused46]", "[unused47]", "[unused48]", "[unused49]", "[unused50]", "[unused51]", "[unused52]", "[unused53]", "[unused54]", "[unused55]", "[unused56]", "[unused57]", "[unused58]", "[unused59]", "[unused60]", "[unused61]", "[unused62]", "[unused63]", "[unused64]", "[unused65]", "[unused66]", "[unused67]", "[unused68]", "[unused69]", "[unused70]", "[unused71]", "[unused72]", "[unused73]", "[unused74]", "[unused75]", "[unused76]", "[unused77]", "[unused78]", "[unused79]", "[unused80]", "[unused81]", "[unused82]", "[unused83]", "[unused84]", "[unused85]", "[unused86]", "[unused87]", "[unused88]", "[unused89]", "[unused90]", "[unused91]", "[unused92]", "[unused93]", "[unused94]", "[unused95]", "[unused96]", "[unused97]", "[unused98]", "[unused99]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]

def remove_accents(string):
    string = string.replace("i̇", "i")
    string = string.replace(" "," ").replace("　"," ").replace(" "," ").replace("%(", "% (").replace("%)", "% )").replace("'(", "' (").replace("')", "' )").replace('"(', '" (').replace('")','" )').replace("',","' ,").replace("'\"", "' \"").replace("\"'", "\" '").replace('"\'', '" \'').replace('\'"', '\' "')
    return string

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
      if isinstance(text, str):
        return text
      elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
      else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    # elif six.PY2:
    #   if isinstance(text, str):
    #     return text.decode("utf-8", "ignore")
    #   elif isinstance(text, unicode):
    #     return text
    #   else:
    #     raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
      raise ValueError("Not running on Python2 or Python 3?")

def load_bpe_vocab(vocab_file, do_lower_case=False):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
      while True:
        token = convert_to_unicode(reader.readline())
        if not token:
          break
        token = token.strip()
        if do_lower_case and token not in preoccupied_vocab:
          token = token.lower()
        vocab[token] = index
        index += 1
    return vocab

class FullBPETokenizer(object):
    """Run default BPE tokenizer"""

    def __init__(self, vocab_file, code_file, blink_token_delimiter=False, do_lower_case=True, use_bnlp=False):
        self.use_bnlp = use_bnlp
        if self.use_bnlp is True:
          self.bnlp = BnlpPy()
        self.vocab = load_bpe_vocab(vocab_file, do_lower_case)
        with open(code_file, "r") as code:
          self.bpe = BPE(code, -1, "@@", set(self.vocab.keys()), None, blink_token_delimiter)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        if self.use_bnlp is True:
            sl = self.bnlp.analyze(text)
            temp_text = []
            for i in range(sl.size()):
                for j in range(sl[i].getWordList().size()):
                    for k in range(sl[i].getWordList()[j].getTokenList().size()):
                        temp_text.append(sl[i].getWordList()[j].getTokenList()[k].getRawText())
            text = " ".join(temp_text)

        if self.do_lower_case is True:
            text = text.lower()
        text = remove_accents(text)

        text = self.bpe.process_line(text)
        tokens = text.strip().split(" ")

        split_tokens = []
        for token in tokens:
            if len(token) != 0:
                split_tokens.append(token)

        return split_tokens
