from data_preprocessing.sentiment import calculate_polarity, calculate_subjectivity


def test_calculate_polarity(input_text):
    polarity = calculate_polarity(input_text)
    assert polarity == -0.325


def test_calculate_subjectivity(input_text):
    subjectivity = calculate_subjectivity(input_text)
    assert subjectivity == 0.52
