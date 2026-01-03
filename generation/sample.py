from generation.generate import generate_text_simple
from generation.utils import text_to_token_ids, token_ids_to_text

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    token_ids = generate_text_simple(
        model, encoded, max_new_tokens=50, context_size=context_size
    )

    print(token_ids_to_text(token_ids, tokenizer))
    model.train()
