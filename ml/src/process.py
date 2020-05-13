#!/usr/bin/env python

import pandas as pd


if __name__ == '__main__':
        data = pd.read_csv("./ml/data/ISEAR.csv",
                           names=["index", "emotion", "sentence"])
        print(data.shape)
        print("The columns names are {}".format(data.columns))
