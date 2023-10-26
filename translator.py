import torch
from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self, src_language="en", tgt_language="es"):
        model_name = f"Helsinki-NLP/opus-mt-{src_language}-{tgt_language}"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

        # Moving the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def translate_text(self, text):
        """
        Translates a given text.

        :param text: Text to be translated.
        :return: Translated text.
        """
        # Convert non-string input to string
        if not isinstance(text, str):
            text = str(text)

        # If the text is empty or contains only whitespace, return it as-is
        if not text.strip():
            return text

        # Split the input text into chunks of approximately 100 words
        text_chunks = text.split(' ')
        chunk_size = 100
        chunks = [' '.join(text_chunks[i:i + chunk_size]) for i in range(0, len(text_chunks), chunk_size)]

        translated_chunks = []
        for chunk in chunks:
            # Tokenize and translate the chunk
            tokenized_chunk = self.tokenizer(chunk, return_tensors="pt")
            input_ids = tokenized_chunk["input_ids"].to(device)  # Move input tensor to the device

            translated_tokens = self.model.generate(input_ids, max_length=720)
            translated_chunk = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

            translated_chunks.append(translated_chunk)

        # Combine the translated chunks
        translated_text = ' '.join(translated_chunks)
        return translated_text
