# In this program we create the random sample of the trade-data corresponding to the random sample we made for
# the book-data. It also checks the last stock file it process and starts from there.

import volat as vl
import pandas as pd

# We read the relevant files

filenames = vl.files('/home/abatsis/Λήψεις/volatility parquet/trade_train.parquet')
filename1 = '/home/abatsis/Λήψεις/contracted/test_data.csv'

# Find the stock_id of the last file

existing = vl.files("/home/abatsis/Λήψεις/contracted/trade_test_sample")
existing = list(map(vl.stock_id, existing))
existing = existing + [-1]
m = max(existing)
print(m)


# Create random samples

frames = []
for i in filenames:
    sid = vl.stock_id(i)
    if sid > m:
        frame = vl.counterpart_file(filename1, i)
        size = frame.shape[0]
        column = [sid] * size
        frame['stock_id'] = column
        frame.to_csv("/home/abatsis/Λήψεις/contracted/trade_test_sample/trade_test_data_{}.csv".format(sid))


# Concatenate

filenames = vl.files('/home/abatsis/Λήψεις/contracted/trade_test_sample')
frames = []
for i in filenames:
    frame = vl.csv_frame(i)
    frames.append(frame)
frame = pd.concat(frames)

frame.to_csv("/home/abatsis/Λήψεις/contracted/trade_test_sample/trade_test.csv")
