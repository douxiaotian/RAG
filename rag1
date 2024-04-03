from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever

def rag_example(question, index_name="wiki_dpr", use_dummy_dataset=False):
    """
    Simple RAG example to generate an answer to a question using a retriever-generator approach.

    Args:
    - question (str): The question to answer.
    - index_name (str): The name of the index to use for retrieving documents. Default is "wiki_dpr".
    - use_dummy_dataset (bool): Whether to use a dummy dataset for demonstration. Useful for testing.

    Returns:
    - answer (str): The generated answer to the question.
    """

    # Initialize the tokenizer and model
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

    # Initialize the retriever
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name=index_name, use_dummy_dataset=use_dummy_dataset)

    # Update the model's retriever
    model.set_retriever(retriever)

    # Encode the input
    inputs = tokenizer(question, return_tensors="pt")

    # Generate an answer
    output = model.generate(input_ids=inputs["input_ids"])

    # Decode and print the answer
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Example usage
question = "What is the capital of France?"
print(rag_example(question, use_dummy_dataset=True))  # Using a dummy dataset for demonstration
