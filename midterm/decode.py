import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Code to experiment with beam-search and top-k sampling decoding strategies.
# You can play around with different prefixes, sequence lengths, and
# beam widths to understand the behavior of the strategies.


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)


# Context/prefix of the generation. You can explore different prefixes.
input_ids = tokenizer.encode('I know', return_tensors='tf')


# beam search.
beam_output = model.generate(
    input_ids,
    max_length=50, # Length of sequence.
    num_beams=10, # Beam size
)
print("Beam Search Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))


# Top-k sampling
tf.random.set_seed(100)
sample_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=50, # Length of sequence.
    top_k=100 # Beam size
)
print("Top-k Sampling Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
