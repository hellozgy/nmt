#!/bin/env python

import re
import os,sys
from sys import stderr

def mybleu(id):
    bin = os.path.abspath(os.path.dirname(__file__))
    work_dir = os.path.abspath(os.path.join(bin, '..'))
    os.system("perl %s/wrap_xml.pl zh %s/valid.en.sgm Result < %s/result_%s.txt > %s/hyp_%s.sgm" % (bin, work_dir, work_dir, id, work_dir, id))
    os.system("perl %s/chi_char_segment.pl -type xml < %s/hyp_%s.sgm > %s/hyp_%s.seg.sgm" % (bin, work_dir, id, work_dir, id))
    os.system("perl %s/chi_char_segment.pl -type xml < %s/valid.zh.sgm > %s/ref.seg.sgm" % (bin, work_dir, work_dir))
    os.system("perl %s/mteval-v11b.pl -s %s/valid.en.sgm -r %s/ref.seg.sgm -t %s/hyp_%s.seg.sgm -c > %s/bleu_%s " % (bin, work_dir, work_dir, work_dir, id, work_dir, id))
    bleu_score=0
    try:
        raw_bleu = " ".join(open(work_dir + "/bleu_"+id, 'r').readlines()).replace("\n", " ")
        gps = re.search(r'NIST score = (?P<NIST>[\d\.]+)  BLEU score = (?P<BLEU>[\d\.]+) ', raw_bleu)
        if gps:
            bleu_score = gps.group('BLEU')
        else:
            print("ERROR: unable to get bleu and nist score", file=stderr)
            sys.exit(1)
    except:
        print("ERROR: exception during calculating bleu score", file=stderr)
    return bleu_score

def generate_sgm(id):
    bin = os.path.abspath(os.path.dirname(__file__))
    test_dir = os.path.abspath(os.path.join(bin, '../../ai_challenger_translation_test_a_20170923/'))
    os.system("perl %s/wrap_xml.pl zh %s/test_a.sgm Result < %s/result_%s.txt > %s/hyp_%s.sgm" % (bin, test_dir, test_dir, id, test_dir, id))

if __name__=='__main__':
    id = sys.argv[1]
    print(id)
    print('bleu score:{}'.format(mybleu(id)))


