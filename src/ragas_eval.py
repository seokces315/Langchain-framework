from bert_score import score
from ragas.metrics import answer_correctness, answer_relevancy, faithfulness
from ragas import evaluate

from datasets import Dataset

import json
import time


# Method that returns collection of test prompts
def get_test_set(test_path):

    # Load json file
    with open(test_path, "r") as test_file:
        test_set = json.load(test_file)

    # Split into 2 kinds of prompt
    questions = test_set["questions"]
    ground_truths = test_set["ground_truths"]

    return questions, ground_truths


# Method that evaluates the langchain
def do_evaluate(rag_chain, retriever, questions, ground_truths):

    # Define BERT-score metrics
    total_Precision = 0
    total_Recall = 0
    total_F1 = 0
    cnt = 0

    # Prepare outputs to evaluate
    contexts = []
    answers = []

    for query in questions:
        context = [doc.page_content for doc in retriever.get_relevant_documents(query)]
        answer = rag_chain.invoke(query)

        print("Done!")

        contexts.append(context)
        answers.append(answer)

        if cnt == 30:
            time.sleep(30)
        time.sleep(3)
        cnt += 1

    # Evaluate the task by BERT-score
    Precision, Recall, F1 = score(
        answers, ground_truths, lang="eng", device="cuda", verbose=True
    )

    # Averaging
    for i, (precision, recall, f1) in enumerate(zip(Precision, Recall, F1)):
        print(
            f"Sentence {i+1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )
        total_Precision += precision
        total_Recall += recall
        total_F1 += f1
    total_Precision /= len(answers)
    total_Recall /= len(answers)
    total_F1 /= len(answers)

    """ 
    # Build dataset consists of outputs
    output = {
        "question": questions[:1],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths[:1]
      }
      output_dataset = Dataset.from_dict(output)
  
    # Apply 3 main metrics for question-answering task
    result = evaluate
    (
        output_dataset,
        metrics=[
            answer_correctness,
            answer_relevancy,
            faithfulness
        ]
    )
    """

    return total_Precision, total_Recall, total_F1


# Test scripts for debugging
if __name__ == "__main__":

    # Local field
    test_path = "../datasets/benchmark.json"

    questions, ground_truths = get_test_set(test_path)
