#!/usr/bin/python3.9

from nmrespy.load import pickle_load

info = pickle_load('cyclosporin.pkl')

txt_description = \
"Sample Bruker spectrum of cyclosporin, in benzene-d6."

pdf_description = \
"Testing the capability of description using \\LaTeX.\\\\$\\exp(\\mathrm{i} \\pi) = -1$"
info.write_result(description=txt_description, fname='cyclosporin', force_overwrite=True)
exit()
info.write_result(description=pdf_description, fname='cyclosporin', fmt='pdf')
