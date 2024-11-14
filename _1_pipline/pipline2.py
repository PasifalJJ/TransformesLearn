from transformers import pipeline

if __name__ == '__main__':
    question_answerer = pipeline('question-answering')
    answer = question_answerer({
        'question': 'What is the name of the repository ?',
        'context': 'Pipeline has been included in the huggingface/transformers repository'
    })
    print(answer)

