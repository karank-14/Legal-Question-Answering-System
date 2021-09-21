from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import sys
import re

# make the output folder if it doesn't already exist
Path("/home/king-karan/Project/tfidf-bert/tf_idf_output").mkdir(parents=True, exist_ok=True)

all_txt_files = []
for files in Path("txt").rglob("*.txt"):
    all_txt_files.append(files.parent / files.name)

all_docs = []
for txt_file in all_txt_files:
    with open(txt_file) as f:
        txt_file_as_string = f.read()
    all_docs.append(txt_file_as_string)

vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, norm=None, sublinear_tf=False)
transformed_documents = vectorizer.fit_transform(all_docs)

transformed_documents_as_array = transformed_documents.toarray()

# construct a list of output file paths using the previous list of text files the relative path for tf_idf_output
output_filenames = [str(txt_file).replace(".txt", ".csv").replace("txt/", "tf_idf_output/") for txt_file in
                    all_txt_files]

# loop each item in transformed_documents_as_array, using enumerate to keep track of the current position
print('What is your question?')

query_string = sys.stdin.readline()
transformed_question = vectorizer.transform([query_string]).toarray()

answertextarray = []

for counter, doc in enumerate(transformed_documents_as_array):
    # construct a dataframe
    tf_idf_tuples = list(zip(vectorizer.get_feature_names(), doc))
    one_doc_as_df = pd.DataFrame.from_records(tf_idf_tuples, columns=['term', 'score']).sort_values(by='score',
                                                                                                    ascending=False).reset_index(
        drop=True)
    theta = cosine_similarity(np.array(transformed_question.reshape(1, -1)), np.array(doc.reshape(1, -1)))

    if theta > 0.1:
        print (theta)
        print (output_filenames[counter])
        answertextarray.append(output_filenames[counter])
    # output to a csv using the enumerated value for the filename
    one_doc_as_df.to_csv(output_filenames[counter])

output_filenames_spans = [str(txt_file).replace(".csv", ".txt").replace("tf_idf_output/", "txt/") for txt_file in
                    answertextarray]

for counter, doc in enumerate(transformed_question):
    tf_idf_question = list(zip(vectorizer.get_feature_names(), doc))
    one_question_as_df = pd.DataFrame.from_records(tf_idf_question, columns=['term', 'score']).sort_values(by='score',
                                                                                                           ascending=False).reset_index(
        drop=True)
one_question_as_df.to_csv('Output.csv')

qapipe = pipeline('question-answering')

for txt in output_filenames_spans:
    answer_text = open(txt).read()
    ans = qapipe({
        'question': query_string,
        'context': answer_text
    })
    tests = answer_text.split("\n")
    charCount = 0
    paragraphNo = -1
    for para in tests:
        charCount = charCount + len(para)
        if ans["start"] < charCount:
            paragraphNo = para
            break;
    print(ans)
    print(paragraphNo)
