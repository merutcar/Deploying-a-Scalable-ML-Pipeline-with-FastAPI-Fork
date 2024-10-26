# Model Card

### Model Details

> The model is a Random Forest Classifier that was trained to predict whether an individual's income is more or less than 50 thousand dollars. It analyses various features from the [U.S. Census](https://archive.ics.uci.edu/dataset/20/census+income) dataset.

### Intended Use

> The model was created as part of a final project for a class. Thus, it should only be used for general purposes. It is not recommended to be used for critical decision-making or situations of similar importance.

### Training Data

> The model was trained on a subset of the U.S. Census dataset, which includes demographic features such as workclass, education, marital status, occupation, relationship, race, sex, and native country. The data was split into a training dataset (80%) and a test dataset (20%) to train and evaluate the model's performance.

### Evaluation Data

> The evaluation data is the test dataset, which constitutes 20% of the original U.S. Census dataset. This dataset includes the same demographic features as the training data and was used to assess the model's performance.

### Metrics

> After assessing the model's performance, these are the metrics that > were obtained:<br>
> - ***Precision***: 0.7419
> - ***Recall***: 0.6384
> - ***F1 score***: 0.6863

### Ethical Considerations

> The dataset contains [sensitive information](https://csrc.nist.gov/glossary/term/sensitive_information) that can lead to bias unless it is handled properly. If the model is used, ensure it is corrected or noted for potential bias.

### Caveats and Recommendations

> It is recommended to use this model for educational purposes or exploratory data analysis only.