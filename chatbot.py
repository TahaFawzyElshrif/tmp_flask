from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "facebook/blenderbot-400M-distill"
# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="./local_cache")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./local_cache")
conversation_history = []
#---> The transformers library function you are using expects to receive the conversation history as a string, with each element separated by the newline character '\n'. Thus, you create such a string.
history_string = "\n".join(conversation_history)




input_text ="hello, how are you doing?"
inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print(response)
conversation_history.append(input_text)
conversation_history.append(response)
print(conversation_history)


while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)