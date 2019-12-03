#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging

from vncorenlp import VnCoreNLP


def simple_usage():
    # Uncomment this line for debugging
    # logging.basicConfig(level=logging.DEBUG)

    vncorenlp_file = r'Dataset/VnCoreNLP-master/VnCoreNLP-1.1.1.jar'

    sentences = 'VTV đồng ý chia sẻ bản quyền World Cup 2018 cho HTV để khai thác. ' \
                'Nhưng cả hai nhà đài đều phải chờ sự đồng ý của FIFA mới thực hiện được điều này.'

    # Use "with ... as" to close the server automatically
    with VnCoreNLP(vncorenlp_file) as vncorenlp:
        print('Tokenizing:', vncorenlp.tokenize(sentences))
        print('POS Tagging:', vncorenlp.pos_tag(sentences))
        print('Named-Entity Recognizing:', vncorenlp.ner(sentences))
        print('Dependency Parsing:', vncorenlp.dep_parse(sentences))
        print('Annotating:', vncorenlp.annotate(sentences))
        print('Language:', vncorenlp.detect_language(sentences))

    # In this way, you have to close the server manually by calling close function
    vncorenlp = VnCoreNLP(vncorenlp_file)

    print('Tokenizing:', vncorenlp.tokenize(sentences))
    print('POS Tagging:', vncorenlp.pos_tag(sentences))
    print('Named-Entity Recognizing:', vncorenlp.ner(sentences))
    print('Dependency Parsing:', vncorenlp.dep_parse(sentences))
    print('Annotating:', vncorenlp.annotate(sentences))
    print('Language:', vncorenlp.detect_language(sentences))

    # Do not forget to close the server
    vncorenlp.close()


if __name__ == '__main__':
    simple_usage()
