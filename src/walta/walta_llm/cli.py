# src/walta/walta_llm/cli.py
import click
import os
import json
from typing import Optional, List

from walta.walta_llm.llm_manager import ModelFactory
from walta.walta_llm.providers.base import ChatMessage # Keep ChatMessage for example usage

@click.group()
def cli():
    """Walta LLM CLI tools."""
    pass

@cli.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to the directory containing input JSON files."
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Path to the directory to save output JSON files."
)
@click.option(
    "--provider",
    type=click.Choice(["deepseek"]), # Only DeepSeek now
    default="deepseek", # Default to deepseek
    help="The LLM provider to use."
)
@click.option(
    "--local-model-path",
    type=str,
    default=None,
    help="Path to the local GGUF model file if using the 'deepseek' provider."
)
@click.option(
    "--n-gpu-layers",
    type=int,
    default=0,
    help="Number of layers to offload to GPU for local LLM (0 for CPU, -1 for all layers)."
)
@click.option(
    "--max-tokens",
    type=int,
    default=1000,
    help="Maximum tokens to generate."
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature for text generation."
)
@click.option(
    "--system-prompt",
    type=str,
    default="You are a helpful AI assistant.",
    help="System prompt to guide the AI's behavior."
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output."
)
def process(
    input_dir: str,
    output_dir: str,
    provider: str,
    local_model_path: Optional[str],
    n_gpu_layers: int,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    verbose: bool
):
    """
    Processes text files in input_dir using the specified LLM provider
    and saves results to output_dir.
    """
    if verbose:
        click.echo(f"Input directory: {input_dir}")
        click.echo(f"Output directory: {output_dir}")
        click.echo(f"LLM Provider: {provider}")
        click.echo(f"Local Model Path: {local_model_path}")
        click.echo(f"GPU Layers: {n_gpu_layers}")
        click.echo(f"Max Tokens: {max_tokens}")
        click.echo(f"Temperature: {temperature}")
        click.echo(f"System Prompt: {system_prompt}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Pass all relevant kwargs to the ModelFactory.create method
        llm_manager = ModelFactory.create(
            provider,
            model_path=local_model_path, # For local/deepseek provider
            n_gpu_layers=n_gpu_layers,   # For local/deepseek provider
            max_tokens=max_tokens,
            temperature=temperature
        )
    except Exception as e:
        click.echo(f"Error initializing LLM provider: {e}", err=True)
        return

    input_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    if not input_files:
        click.echo(f"No JSON files found in input directory: {input_dir}")
        return

    for filename in input_files:
        input_filepath = os.path.join(input_dir, filename)
        output_filepath = os.path.join(output_dir, f"processed_{filename}")

        if verbose:
            click.echo(f"\nProcessing {filename}...")

        try:
            with open(input_filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "text_content" not in data:
                click.echo(f"Warning: 'text_content' not found in {filename}. Skipping.", err=True)
                continue

            user_prompt = data["text_content"]

            messages: List[ChatMessage] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Generate response using chat completion for better DeepSeek integration
            response_content = llm_manager.generate_chat_completion(messages)

            # Save processed data
            output_data = {
                "original_filename": filename,
                "original_text_content": user_prompt,
                "ai_response": response_content,
                "llm_provider": provider,
                "model_parameters": llm_manager.get_model_parameters()
            }

            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4)

            if verbose:
                click.echo(f"Successfully processed {filename}. Output saved to {output_filepath}")
                click.echo(f"AI Response (first 200 chars):\n{response_content[:200]}...")

        except Exception as e:
            click.echo(f"Error processing {filename}: {e}", err=True)

    click.echo("\nProcessing complete.")

if __name__ == "__main__":
    cli()
