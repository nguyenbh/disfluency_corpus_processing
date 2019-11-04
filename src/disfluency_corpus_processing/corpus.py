'''Corpus Data Processor.'''
from nltk.tokenize import TreebankWordTokenizer
from nltk import sent_tokenize

import logging
import os
import re
import string

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Corpus():
    FILE_TYPES = ['dff', 'dps', 'mgd', 'scotus', 'fcic', 'callhome']

    def __init__(self,
                 input_file,
                 file_type='dps',
                 punctuation=False):
        '''Initialization for a Corpus object.

        Parameters
        ----------
        input_file : str
            a path to corpus file

        file_type: str
            a corpus filetype, default is 'dps'
        '''
        if file_type not in self.FILE_TYPES:
            logger.error('File type \'%s\' is not supported!' % file_type, exc_info=True)
            self.file_type = None
        else:
            self.file_type = file_type

        if not os.path.exists(input_file):
            logger.error('Input file \'%s\' does not exist!' % input_file, exc_info=True)
            self.input_file = None
        else:
            self.input_file = input_file

        self.punctuation = punctuation

        self.tokenizer = TreebankWordTokenizer()

    def parse(self):
        """Read a particular file."""
        parser_map = {
            'dps': self._parse_dps,
            'dff': self._parse_dff,
            'mgd': self._parse_mgd,
            'scotus': self._parse_scotus,
            'fcic': self._parse_fcic,
            'callhome': self._parse_callhome
        }
        parser = parser_map[self.file_type]
        return parser()

    def __process_curly(self, stack, last):
        flat = []
        # Ignore the coordinating conjunction marker {C ...}
        # and explicit editing term {E ...}
        if stack[last + 1] not in ['C', 'E']:
            for w in stack[last + 2:]:
                if not self.punctuation:  # no punctuations
                    if w[0] not in string.punctuation:
                        if '@dis' not in w:
                            flat.append(w + '/@dis')
                        else:
                            flat.append(w)
                else:
                    if '@dis' not in w:
                        flat.append(w + '/@dis')
                    else:
                        flat.append(w)
        else:  # ignore coordinating conjunction
            if not self.punctuation:  # no punctuations
                flat = [w for w in stack[last + 2:] if w[0] not in string.punctuation]
            else:
                flat = stack[last + 2:]

        stack = stack[0:last]
        stack.extend(flat)

        return stack

    def __process_square(self, stack, last):
        flat = ' '.join(stack[last + 1:])

        # process [RM + {} RR] or [RM +]
        plus_idx = len(stack) - 1 - stack[::-1].index('+')
        flat_dis = []
        for w in stack[last + 1:plus_idx]:
            if not self.punctuation:  # no punctuations
                if w[0] not in string.punctuation:
                    if '@dis' not in w:
                        flat_dis.append(w + '/@dis')
                    else:
                        flat_dis.append(w)
            else:
                if '@dis' not in w:
                    flat_dis.append(w + '/@dis')
                else:
                    flat_dis.append(w)

        # deal with string after +
        if not self.punctuation:
            repair = [w for w in stack[plus_idx + 1:] if w[0] not in string.punctuation]
        else:
            repair = stack[plus_idx + 1:]

        stack = stack[0:last]
        stack.extend(flat_dis)
        stack.extend(repair)

        return stack

    def _process_dps_segment(self, segment):
        """Convert a DPS annotated string to flat disfluency annotation.

        Follow [Johnson & Charniak, 2004], [Honibal & Johnson, 2014], we process raw data as follow
            1. Ignore the coordinating conjunction marker {C ...}
            2. Remove punctuations
            3. Remove partial words
            4. Remove E_S & N_S

        References:
            A TAG-based noisy-channel model of speech repairs, Johnson & Charniak, ACL 2004
            Joint incremental disfluency detection and dependency parsing, Honibal & Johnson, TACL 2014
        """
        # standardize {, }, [, ]
        segment = re.sub(r"(\{|\[)(.*?)", r"\1 \2", segment)
        segment = re.sub(r"(.*?)(\}|\])", r"\1 \2", segment)

        stack = []
        for w in segment.split():
            if '-/' in w:  # ignore partial incomplete words
                continue
            if w[-1] == '-':  # ignore partial incomplete words
                continue
            if 'E_S' in w or 'N_S' in w:  # ignore E_S marker
                continue

            if w == '}':
                last = len(stack) - 1 - stack[::-1].index('{')
                stack = self.__process_curly(stack, last)
            elif w == ']':
                last = len(stack) - 1 - stack[::-1].index('[')
                stack = self.__process_square(stack, last)
            else:  # token outside of curly and square brackets
                if not self.punctuation:  # no punctuation
                    if w in ['{', '[', '+']:
                        stack.append(w)
                    elif w.split('/')[0] not in string.punctuation:
                        stack.append(w)
                else:
                    stack.append(w)

        return stack

    def _parse_dps(self):
        """Parse dps file type."""
        processed, block = [], []
        with open(self.input_file, 'r', encoding='utf-8') as fh:
            for l in fh:
                line = l.rstrip()

                if line.startswith('Speaker') and line.endswith('./.'):
                    if 'Speaker' in block[0]:
                        # print('DEBUG: block=', '|||'.join(block[1:]))
                        for segment in block[1:]:
                            if segment:
                                output = self._process_dps_segment(segment)
                                processed.append(output)

                    block = []

                block.append(line)

        return processed

    def _parse_scotus(self):
        """Parse SCOTUS file type."""
        processed, block = [], []
        with open(self.input_file, 'r', encoding='utf-8') as fh:
            for l in fh:
                line = l.rstrip()
                # apply Treebank tokenizer
                toked = ' '.join((self.tokenizer.tokenize(line)))
                output = self._process_dps_segment(toked)
                processed.append(output)

        return processed

    def _parse_fcic(self):
        """Parse FCIC file type."""
        processed, block = [], []
        with open(self.input_file, 'r', encoding='utf-8') as fh:
            for l in fh:
                line = l.rstrip()
                if len(line) == 0:
                    if len(block) > 0:
                        # logging.info(block)
                        sentences = sent_tokenize(' '.join(block))
                        for segment in sentences:
                            # apply Treebank tokenizer
                            toked = ' '.join((self.tokenizer.tokenize(segment)))
                            if toked:
                                output = self._process_dps_segment(toked)
                                processed.append(output)
                    block = []
                else:
                    if line[-1] == ":":  # speaker ID
                        continue
                    block.append(line)

        return processed

    def _parse_callhome(self):
        """Parse CallHome file type."""
        def block_process(block, processed):
            if len(block) > 0:
                sentences = sent_tokenize(' '.join(block))
                if len(sentences) > 1:
                    segments = sentences
                else:
                    segments = block

                for segment in segments:
                    # apply Treebank tokenizer
                    toked = ' '.join((self.tokenizer.tokenize(segment)))
                    if toked:
                        # logging.info(toked)
                        output = self._process_dps_segment(toked)
                        processed.append(output)
            return processed

        processed = []
        markers = [
            '((', '))', '&',
            # replace % word marker
            '%uh', '%um', '%eh', '%mm', '%hm', '%ah', '%huh',
            # replace speech markers {}
            '{breath}', '{laugh}', '{lipsmack}', '{inhale}', '{exhale}', '{sniff}', '{sigh}'
        ]
        with open(self.input_file, 'r', encoding='utf-8') as fh:
            block_A, block_B = [], []
            for l in fh:
                if l[0] == '#':
                    continue

                # replace marker
                for marker in markers:
                    cleaned_line = l.replace(marker, ' ')

                line = cleaned_line.rstrip()
                if len(line) > 0:
                    timestamp, text = line.split(':')
                    if 'A' in timestamp:
                        block_A.append(text)
                    elif 'B' in timestamp:
                        block_B.append(text)

            processed = block_process(block_A, processed)
            processed = block_process(block_B, processed)

        return processed

    def _parse_dff(self):
        """Parse dff file type."""
        pass

    def _parse_mgd(self):
        """Parse dps file type."""
        pass
