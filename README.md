# Spark project : Kickstarter campaigns

In order to run the project in your computer, please change the "path_to_spark" 
line in the "build_and_submit.sh" file and replace it with your SPARK_HOME.

Some precisions:
- The data used by the Trainer class is the data obtained after treating the data
in the Preprocessor class, and not the data provided with the project.
- Some extra-cleaning was made in order to boost performance. In particular, 
commas were deleted in the description field. This was causing a bad importing
with the csv function and several valuable entries were being dropped in the
cleaning part

- Best performances are reached with minDF = 35 and alpha = 1e-5 for the logistic regression.
- This parameters lead to f1 = 0.70 - 0.72, depending on the seed of the train/test split.
    - For instance, with a seed = 100 in the train/test split, we obtain f1 = 0.71