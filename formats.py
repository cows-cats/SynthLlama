FORMAT_TEMPLATES = {
    # Question & Answer (Q&A) Format
    "qa_format": {
        "question": "What is the capital of Turkey?",
        "answer": "Ankara"
    },

    # Chat Format
    "chat_format": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {"role": "user", "content": "Can I check the weather?"}
    ],

    # Completion Format
    "completion_format": {
        "prompt": "Once upon a time,",
        "completion": " long ago, in a faraway land..."
    },

    # Text Classification Format
    "text_classification_format": {
        "text": "This movie was amazing!",
        "label": "positive"
    },

    # Translation Format
    "translation_format": {
        "source": "Hello, how are you?",
        "target": "Merhaba, nasılsın?"
    },

    # Instruction-Response Format
    "instruction_response_format": {
        "instruction": "Summarize the following text",
        "input": "Long text...",
        "output": "Summary text..."
    },

    # Summarization Format
    "summarization_format": {
        "article": "Long article text...",
        "summary": "Summary text..."
    },

    # Dialogue Format
    "dialogue_format": [
        {"speaker": "A", "text": "How are you?"},
        {"speaker": "B", "text": "I'm fine, and you?"},
        {"speaker": "A", "text": "I'm good too"}
    ]
}

