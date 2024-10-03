
# Assignment Sentiment Analysis

A Naive Bayes Model in order to perform sentiment analysis on a Rotten Tomatoes movie review dataset.

## Authors

- Alex Ganvir


## Libraries Required

Detailed in Requirements.txt

- argparse
- pandas
- numpy
- nltk
- scipy
## Usage

You will see in main.py here that you have to select if you want to run the model on the development or test dataset, leave the one you do not want to run commmented out. The Code Snippet is at line 402-404, line 403 is for the Dev set, and line 404 is for the Test set. In the below example, the Test set is commmented out, so the Dev set will be ran:

```python
# comment out for which one you want to run (must run both to output files)
calculate_estimate_sentiment(training_data, dev_data, all_words, prior_probabilties, number_classes, word_for_sentiment_len_dict)  # run on development data 
# calculate_estimate_sentiment(training_data, test_data, all_words, prior_probabilties, number_classes, word_for_sentiment_len_dict)  # run on test data
```
When running the model on the Test set you also need to comment out lines 412-413, shown here:
```python
else:
    f1_macro_score, f1_scores = compare_estimate_sentiment_to_real(dev_data, number_classes, confusion_matrix)
```

Leave this uncommented if running both. 

Do not run the confusion matrix with the test set. Run both the dev and the test set when using output_files.

Otherwise run as stated in the brief:

```bash
python NB_sentiment_analyser.py <TRAINING_FILE> <DEV_FILE> <TEST_FILE> -classes
<NUMBER_CLASSES> -features <all_words,features> -output_files -confusion_matrix
```
Where:
- <TRAINING_FILE> <DEV_FILE> <TEST_FILE> are the paths to the training, dev and test files, respectively;
- -classes <NUMBER_CLASSES> should be either 3 or 5, i.e. the number of classes being predicted;
- -features is a parameter to define whether you are using your selected features or no features (i.e. all words);
- -output_files is an optional value defining whether or not the prediction files should be saved (see below – default is "files are not saved"); and
- -confusion_matrix is an optional value defining whether confusion matrices should be shown (default is "confusion matrices are not shown").

