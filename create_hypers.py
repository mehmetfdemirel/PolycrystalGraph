from __future__ import print_function

import os
import numpy as np
import argparse
import json
import itertools

# THIS CODE WILL CREATE HYPERPARAMETER FILES IN A FOLDER CALLED hyper/
# THEN, YOU CAN PASS THESE FROM YOUR BASH SCRIPT TO YOUR PYTHON SCRIPT

if __name__ == '__main__':
    folder = 'hyper/'
    lr_list = [1e-4,1e-5,1e-6] # learning rate list
    epoch_list = [500,800,1000] # epoch list
    optim_list = ["Adam"]
    latent_dim1_list = [5,10,20]
    latent_dim2_list = [40,50,60]

    # just create lists for all your different hyperparameters
    # and put them all in the all_list list below.

    all_list = [
      lr_list,
      epoch_list,
      optim_list,
      latent_dim1_list,
      latent_dim2_list
    ]

    combs = list(itertools.product(*all_list))

    if not os.path.exists(folder):
        os.mkdir(folder)

    for i, [lr, epoch, optim, latentdim1, latentdim2] in enumerate(combs):
        out_file = '{}{}.json'.format(folder, i)
        wf = open(out_file, 'w+')

        wf.write("{\n")

        wf.write("\t\"lr\": {},\n".format(lr))
        wf.write("\t\"epoch\": {},\n".format(epoch))
        wf.write("\t\"optim\": \"{}\",\n".format(optim))
        wf.write("\t\"latent_dim1\": {},\n".format(latentdim1))
        wf.write("\t\"latent_dim2\": {}".format(latentdim2))

        wf.write("\n}\n")

        wf.close()

    print('{} hyperparameter files have been created in {}'.format((i + 1), folder))
