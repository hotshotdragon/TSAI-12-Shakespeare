# ShakespeareSpeaks 🎭

[![Open In HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20HuggingFace-blue.svg)](https://huggingface.co/spaces/hotshotdragon/ShakespeareSpeaks)
[![Model License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Training Loss](https://img.shields.io/badge/Training%20Loss-<0.09-brightgreen.svg)](https://huggingface.co/spaces/hotshotdragon/ShakespeareSpeaks)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

A fine-tuned GPT-2 model that generates Shakespearean-style text with remarkably low training loss (<0.09). This model can generate creative, period-appropriate text in the distinctive style of William Shakespeare.

## 🌟 Features

- Fine-tuned on Shakespeare's complete works
- Training loss < 0.09
- Generates text with period-appropriate vocabulary and style
- Maintains Shakespearean meter and rhythm
- Supports both prose and verse generation

## 🚀 Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("hotshotdragon/ShakespeareSpeaks")
model = AutoModelForCausalLM.from_pretrained("hotshotdragon/ShakespeareSpeaks")

# Generate text
prompt = "To be, or not to be,"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📊 Model Performance

- Training Loss: <0.09
- Successfully captures Shakespearean linguistic patterns and style
- Fine-tuned on RTX 4060 8GB

## 💻 Usage Examples

### Poetry Generation
```python
prompt = "Shall I compare thee to"
# Generate romantic sonnet-style poetry

### Dramatic Dialogue
prompt = "Romeo: My love,"
# Generate character dialogue

### Soliloquy Creation
prompt = "Now is the winter of"
# Generate introspective monologues
```

## 📝 Input Format

- The model accepts any text prompt
- For best results, start with Shakespearean-style phrases
- Can handle both modern and period-appropriate English
- Supports multiple creative formats (sonnets, dialogues, soliloquies)

## 🔍 Limitations

- May occasionally generate anachronistic language
- Best results achieved with longer context windows
- Performance varies based on prompt quality
- May require multiple attempts for optimal output

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📫 Contact

For questions and feedback, please open an issue on the HuggingFace space.

## 🙏 Acknowledgments

- The Hugging Face team for their excellent transformers library
- Shakespeare's timeless works that made this project possible
- The open-source AI community

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@misc{shakespeare-gpt2-2024,
  author = {hotshotdragon},
  title = {ShakespeareSpeaks: A Fine-tuned GPT-2 Model for Shakespearean Text Generation},
  year = {2024},
  publisher = {HuggingFace},
  journal = {HuggingFace Model Hub}
}
```
