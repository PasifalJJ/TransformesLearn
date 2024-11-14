from transformers import pipeline

if __name__ == '__main__':
    classifier = pipeline('sentiment-analysis')
    print(classifier("We are very happy to introduce pipeline to the transformers repository."))
