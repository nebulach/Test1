"""Use operations learned with learn_bpe.py to encode a new text.
The text will not be smaller, but use only a fixed vocabulary, with rare words
encoded as variable-length sequences of subword units.
A special encode method, i.e. encode_pos_tokens, is added for handling pos-tokenized text as input
Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals, division

import sys
import os
import inspect
import codecs
import io
import argparse
import re
import warnings

# hack for python2/3 compatibility
from io import open
argparse.open = open

preoccupied_vocab = ["[PAD]", "[unused1]", "[unused2]", "[unused3]", "[unused4]", "[unused5]", "[unused6]", "[unused7]", "[unused8]", "[unused9]", "[unused10]", "[unused11]", "[unused12]", "[unused13]", "[unused14]", "[unused15]", "[unused16]", "[unused17]", "[unused18]", "[unused19]", "[unused20]", "[unused21]", "[unused22]", "[unused23]", "[unused24]", "[unused25]", "[unused26]", "[unused27]", "[unused28]", "[unused29]", "[unused30]", "[unused31]", "[unused32]", "[unused33]", "[unused34]", "[unused35]", "[unused36]", "[unused37]", "[unused38]", "[unused39]", "[unused40]", "[unused41]", "[unused42]", "[unused43]", "[unused44]", "[unused45]", "[unused46]", "[unused47]", "[unused48]", "[unused49]", "[unused50]", "[unused51]", "[unused52]", "[unused53]", "[unused54]", "[unused55]", "[unused56]", "[unused57]", "[unused58]", "[unused59]", "[unused60]", "[unused61]", "[unused62]", "[unused63]", "[unused64]", "[unused65]", "[unused66]", "[unused67]", "[unused68]", "[unused69]", "[unused70]", "[unused71]", "[unused72]", "[unused73]", "[unused74]", "[unused75]", "[unused76]", "[unused77]", "[unused78]", "[unused79]", "[unused80]", "[unused81]", "[unused82]", "[unused83]", "[unused84]", "[unused85]", "[unused86]", "[unused87]", "[unused88]", "[unused89]", "[unused90]", "[unused91]", "[unused92]", "[unused93]", "[unused94]", "[unused95]", "[unused96]", "[unused97]", "[unused98]", "[unused99]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]

class BPE(object):

    def __init__(self, codes, merges=-1, separator='@@', vocab=None, glossaries=None, blink_token_delimiter=False, to_lower=True):

        codes.seek(0)
        offset=1

        # check version information
        firstline = codes.readline()
        if firstline.startswith('#version:'):
            self.version = tuple([int(x) for x in re.sub(r'(\.0+)*$','', firstline.split()[-1]).split(".")])
            offset += 1
        else:
            self.version = (0, 1)
            codes.seek(0)

        self.bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(codes) if (n < merges or merges == -1)]

        for i, item in enumerate(self.bpe_codes):
            if len(item) != 2:
                sys.stderr.write('Error: invalid line {0} in BPE codes file: {1}\n'.format(i+offset, ' '.join(item)))
                sys.stderr.write('The line should exist of exactly two subword units, separated by whitespace\n')
                sys.exit(1)

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])

        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair,i in self.bpe_codes.items()])

        self.separator = separator

        self.vocab = vocab

        self.glossaries = glossaries if glossaries else []

        self.cache = {}

        self.blink_token_delimiter = blink_token_delimiter

        self.to_lower = to_lower

    def process_line(self, line):
        """segment line, dealing with leading and trailing whitespace"""

        out = ""

        leading_whitespace = len(line)-len(line.lstrip('\r\n '))
        if leading_whitespace:
            out += line[:leading_whitespace]

        out += self.segment(line)

        trailing_whitespace = len(line)-len(line.rstrip('\r\n '))
        if trailing_whitespace and trailing_whitespace != len(line):
            out += line[-trailing_whitespace:]

        return out

    def segment(self, sentence):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        if self.to_lower == True:
            segments = self.segment_tokens(sentence.lower().strip('\r\n ').split(' '))
        else:
            segments = self.segment_tokens(sentence.strip('\r\n ').split(' '))
        return ' '.join(segments)

    def segment_tokens(self, tokens):
        """segment a sequence of tokens with BPE encoding"""
        output = []
        if self.blink_token_delimiter == True:
            for word in tokens:
                # eliminate double spaces
                if not word:
                    continue
                if word == "" or word == "_<b>":
                    continue



                new_word = [out for segment in self._isolate_glossaries(word)
                            for out in encode_pos_tokens(segment,
                                              self.bpe_codes,
                                              self.bpe_codes_reverse,
                                              self.vocab,
                                              self.separator,
                                              self.version,
                                              self.cache,
                                              self.glossaries)]

                for item in new_word[:-1]:
                    output.append(item + self.separator)
                output.append(new_word[-1])

        else:
            for word in tokens:
                # eliminate double spaces
                if not word:
                    continue

                new_word = [out for segment in self._isolate_glossaries(word)
                            for out in encode(segment,
                                              self.bpe_codes,
                                              self.bpe_codes_reverse,
                                              self.vocab,
                                              self.separator,
                                              self.version,
                                              self.cache,
                                              self.glossaries)]

                for item in new_word[:-1]:
                    output.append(item + self.separator)
                output.append(new_word[-1])
        return output

    def _isolate_glossaries(self, word):
        word_segments = [word]
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                                 for out_segments in isolate_glossary(segment, gloss)]
        return word_segments

def create_parser(subparsers=None):

    if subparsers:
        parser = subparsers.add_parser('apply-bpe',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")
    else:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--codes', '-c', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--merges', '-m', type=int, default=-1,
        metavar='INT',
        help="Use this many BPE operations (<= number of learned symbols)"+
             "default: Apply all the learned merge operations")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument(
        '--vocabulary', type=argparse.FileType('r'), default=None,
        metavar="PATH",
        help="Vocabulary file (built with get_vocab.py). If provided, this script reverts any merge operations that produce an OOV.")
    parser.add_argument(
        '--vocabulary-threshold', type=int, default=None,
        metavar="INT",
        help="Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV")
    parser.add_argument(
        '--blink_token_delimiter', '-b', type=bool, default=False,
        help="Delimiter option - True: Use the blink token as a delimiter, False: Use the blink as a delimiter (default: %(default)s))")
    parser.add_argument(
        '--to_lower', '-l', type=bool, default=True,
        help="Low case option - True: Lower all the characters, False: Keep capitality of the characters (default: %(default)s))")
    parser.add_argument(
        '--glossaries', type=str, nargs='+', default=None,
        metavar="STR",
        help="Glossaries. Words matching any of the words/regex provided in glossaries will not be affected "+
             "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords. "+
             "Can be provided as a list of words/regex after the --glossaries argument. Enclose each regex in quotes.")

    return parser

def get_pairs(word):
    """Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def encode_pos_tokens(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, cache, glossaries=None):
    """
    Encode word based on list of BPE merge operations, which are applied consecutively
    The word that has <b> tag is considered as a whole word (i.e. attach </w> to its end)
    Target text = Pre-tokenized text from the pos tagger
    """
    # if '_<b>' in orig:
    #     if orig.split('_<')[0] in cache:
    #         return cache[orig.split('_<')[0]]
    # else:
    #     if orig in cache:
    #         return cache[orig]

    # for glossary in glossaries:
    #     if '_<b>' in orig:
    #         if re.match('^'+glossary+'$', orig.split('_<')[0]):
    #             cache[orig] = (orig.split('_<')[0],)
    #             return (orig.split('_<')[0],)
    #     else:
    #         if re.match('^'+glossary+'$', orig):
    #             cache[orig] = (orig,)
    #             return (orig,)

    if version == (0, 1):
        if orig.endswith('_<b>'):
            if orig != "" and orig != "_<b>":
                word = tuple(orig.split('_<')[0]) + ('</w>',)
        else:
            word = tuple(orig)
    elif version == (0, 2): # more consistent handling of word-final segments
        if orig.endswith('_<b>'):
            if orig != "" and orig != "_<b>":
                word = tuple(orig.split('_<')[0][:-1]) + (orig.split('_<')[0][-1] + '</w>',)
        else:
            word = tuple(orig[:-1]) + (orig[-1],)
    else:
        raise NotImplementedError

    pairs = get_pairs(word)

    if not pairs:
        return orig.split('_<')[0]

    while True:
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
        if bigram not in bpe_codes:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>',''),)

    if vocab:
        word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

    cache[orig] = word
    return word

def encode(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, cache, glossaries=None):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if orig in cache:
        return cache[orig]

    for glossary in glossaries:
        if re.match('^'+glossary+'$', orig):
            cache[orig] = (orig,)
            return (orig,)

    if version == (0, 1):
        word = tuple(orig) + ('</w>',)
    elif version == (0, 2): # more consistent handling of word-final segments
        word = tuple(orig[:-1]) + (orig[-1] + '</w>',)
    else:
        raise NotImplementedError

    pairs = get_pairs(word)

    if not pairs:
        return orig

    while True:
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
        if bigram not in bpe_codes:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>',''),)

    if vocab:
        word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

    cache[orig] = word
    return word

def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher."""

    try:
        if final:
            left, right = bpe_codes[segment + '</w>']
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except:
        #sys.stderr.write('cannot split {0} further.\n'.format(segment))
        yield segment
        return

    if left + separator in vocab:
        yield left
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

    if (final and right in vocab) or (not final and right + separator in vocab):
        yield right
    else:
        for item in recursive_split(right, bpe_codes, vocab, separator, final):
            yield item

def check_vocab_and_split(orig, bpe_codes, vocab, separator):
    """Check for each segment in word if it is in-vocabulary,
    and segment OOV segments into smaller units by reversing the BPE merge operations"""

    out = []

    for segment in orig[:-1]:
        if segment + separator in vocab:
            out.append(segment)
        else:
            #sys.stderr.write('OOV: {0}\n'.format(segment))
            for item in recursive_split(segment, bpe_codes, vocab, separator, False):
                out.append(item)

    segment = orig[-1]
    if segment in vocab:
        out.append(segment)
    else:
        #sys.stderr.write('OOV: {0}\n'.format(segment))
        for item in recursive_split(segment, bpe_codes, vocab, separator, True):
            out.append(item)

    return out


def read_vocabulary(vocab_file, threshold):
    """read vocabulary file produced by get_vocab.py, and filter according to frequency threshold.
    """

    vocabulary = set()

    for line in vocab_file:
        if line.strip() in preoccupied_vocab:
            continue
        word, freq = line.strip('\r\n ').split(' ')
        freq = int(freq)
        if threshold == None or freq >= threshold:
            vocabulary.add(word)

    return vocabulary

def isolate_glossary(word, glossary):
    """
    Isolate a glossary present inside a word.
    Returns a list of subwords. In which all 'glossary' glossaries are isolated 
    For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
        ['1934', 'USA', 'B', 'USA']
    """
    # regex equivalent of (if word == glossary or glossary not in word)
    if re.match('^'+glossary+'$', word) or not re.search(glossary, word):
        return [word]
    else:
        splits = re.split(glossary, word)
        segments = [segment.strip('\r\n ') for (n_split, split) in enumerate(splits[:-1]) for segment in [split, re.findall(glossary, word)[n_split]] if segment != '']
        return segments + [splits[-1].strip('\r\n ')] if splits[-1] != '' else segments

if __name__ == '__main__':

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    newdir = os.path.join(currentdir, 'subword_nmt')
    if os.path.isdir(newdir):
        warnings.simplefilter('default')
        warnings.warn(
            "this script's location has moved to {0}. This symbolic link will be removed in a future version. Please point to the new location, or install the package and use the command 'subword-nmt'".format(newdir),
            DeprecationWarning
        )

    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True, line_buffering=True)

    parser = create_parser()
    args = parser.parse_args()

    # read/write files as UTF-8
    args.codes = codecs.open(args.codes.name, encoding='utf-8')
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    if args.vocabulary:
        args.vocabulary = codecs.open(args.vocabulary.name, encoding='utf-8')

    if args.vocabulary:
        vocabulary = read_vocabulary(args.vocabulary, args.vocabulary_threshold)
    else:
        vocabulary = None

    if sys.version_info < (3, 0):
        args.separator = args.separator.decode('UTF-8')
        if args.glossaries:
            args.glossaries = [g.decode('UTF-8') for g in args.glossaries]


    bpe = BPE(args.codes, args.merges, args.separator, vocabulary, args.glossaries, args.blink_token_delimiter, args.to_lower)

    for line in args.input:
        args.output.write(bpe.process_line(line))
