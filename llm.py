import cohere


def ask(query, api_key, llm_model='command'):
    co = cohere.Client(api_key=api_key)
    response = co.chat(message=query, model=llm_model)
    return response.text


def augment(query, api_key, index, embedding_model, prompt, llm_model='command', context_size=10):
    query_results = index.query(
        vector=embedding_model.encode(query).tolist(),
        top_k=context_size,
        include_values=True,
        include_metadata=True
    )
    text_matches = [match['metadata']['text'] for match in query_results['matches']]
    context = "\n\n".join(text_matches)
    prompt = prompt.replace('CONTEXT', context).replace('QUERY', query).strip()
    return ask(prompt, api_key, llm_model=llm_model), context
