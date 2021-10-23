# In this program we create a random sample of the book-data. The random sample contains 5/127 of the total size of book
# data and we split the sample into two parts, one for training and one for testing.

import volat as vl


filenames = vl.files('/home/abatsis/Λήψεις/volatility parquet/book_train.parquet')
percentage = 5/127
train_size = 4/5

frames = vl.global_random_sample(filenames, percentage, train_size)

frames[0].to_csv("/home/abatsis/Λήψεις/contracted/train_data.csv")
frames[1].to_csv("/home/abatsis/Λήψεις/contracted/test_data.csv")
