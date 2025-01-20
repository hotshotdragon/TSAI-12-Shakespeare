import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from huggingface_hub import hf_hub_download
import gradio as gr

def load_model_from_huggingface():
    """
    Load the Shakespeare model from Hugging Face along with its tokenizer.
    """
    model_id = "hotshotdragon/Shakespeare"  # The Hugging Face model ID

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Move the model to the appropriate device (CUDA if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, tokenizer

class TextGenerator:
    """
    Class to handle text generation using a pre-trained model.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.model, self.tokenizer = load_model_from_huggingface()
        print("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> str:
        """
        Generate text based on the input prompt.
        """
        try:
            # Tokenize the input prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            # Generate text
            output_sequences = self.model.generate(
                input_ids=input_ids,
                max_length=max_length + len(input_ids[0]),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

            # Decode the generated text
            generated_sequences = []
            for generated_sequence in output_sequences:
                generated_sequence = generated_sequence.tolist()
                text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                # Remove the input prompt from the generated text
                text = text[len(self.tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)):]
                generated_sequences.append(text)

            return " ".join(generated_sequences)

        except Exception as e:
            return f"An error occurred: {str(e)}"


def create_interface():
    """
    Create a Gradio interface for text generation.
    """
    generator = TextGenerator()

    def generate_text(prompt: str, max_length: int, temperature: float, top_k: int, top_p: float) -> str:
        return generator.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    with gr.Blocks() as demo:
        gr.Markdown("# Shakespeare Text Generation")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Input Prompt",
                    placeholder="Enter your prompt here...",
                    lines=5,
                )
                with gr.Row():
                    with gr.Column():
                        max_length = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=100,
                            step=10,
                            label="Maximum Length",
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                        )
                    with gr.Column():
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Top K",
                        )
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top P",
                        )
                generate_button = gr.Button("Generate")

            with gr.Column():
                output = gr.Textbox(label="Generated Text", lines=10)

        generate_button.click(
            fn=generate_text,
            inputs=[prompt, max_length, temperature, top_k, top_p],
            outputs=output,
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)