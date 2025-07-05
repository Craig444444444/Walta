# src/walta/walta_llm/cli.py
"""
Walta Framework CLI - Command-line interface for LLM operations.
"""

import os
import sys
import asyncio
import logging
import click
from typing import Optional
from datetime import datetime
import shutil # Added for directory cleanup/creation

from .llm_manager import WaltaLLM

logger = logging.getLogger(__name__)

# Configure logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def setup_provider_logging(provider: str):
    """Configure logging for specific provider."""
    logger = logging.getLogger(f"walta.llm.{provider}")
    logger.setLevel(logging.INFO)
    return logger

@click.group()
def walta_cli():
    """Walta Framework CLI with LLM support"""
    pass

@walta_cli.command()
@click.option(
    '--provider',
    default='gemini',
    help='LLM provider (gemini/openai)',
    type=click.Choice(['gemini', 'openai'], case_sensitive=False)
)
@click.option(
    '--image',
    type=click.Path(exists=True),
    help='Optional image file path'
)
@click.option(
    '--temperature',
    default=0.7,
    help='Temperature for generation (0.0-1.0)',
    type=click.FloatRange(0, 1)
)
@click.option(
    '--max-tokens',
    default=1000,
    help='Maximum tokens to generate',
    type=click.IntRange(1, 4096)
)
@click.argument('query')
async def analyze(
    provider: str,
    image: Optional[str],
    temperature: float,
    max_tokens: int,
    query: str
):
    """Analyze text/image using specified provider"""
    try:
        setup_provider_logging(provider)
        llm = WaltaLLM(primary_provider_name=provider)

        image_data = None
        if image:
            try:
                with open(image, 'rb') as f:
                    image_data = f.read()
            except Exception as e:
                click.echo(f"Error: Failed to read image file: {e}", err=True)
                return

        click.echo(f"Analyzing with {provider} provider...")

        result = await llm.analyze(
            query,
            image=image_data
        )

        click.echo("\nAnalysis Result:")
        click.echo("=" * 40)
        click.echo(f"Provider: {result['provider']}")
        click.echo(f"Latency: {result['latency']:.2f}s")
        click.echo(f"Timestamp: {result['timestamp']}")
        click.echo(f"Request ID: {result['request_id']}")
        click.echo("=" * 40)
        click.echo("\nResult:")
        click.echo(result['result'])

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        click.echo(f"An error occurred during analysis: {e}", err=True)

@walta_cli.command()
@click.option(
    '--provider',
    default='gemini',
    help='LLM provider (gemini/openai)',
    type=click.Choice(['gemini', 'openai'], case_sensitive=False)
)
@click.argument('text_input')
async def embed(provider: str, text_input: str):
    """Get embedding vector for text using specified provider"""
    try:
        setup_provider_logging(provider)
        llm = WaltaLLM(primary_provider_name=provider)

        click.echo(f"Getting embedding with {provider} provider...")
        embedding_result = await llm.get_embedding(text_input)

        click.echo("\nEmbedding Result:")
        click.echo("=" * 40)
        click.echo(f"Provider: {embedding_result['provider']}")
        click.echo(f"Latency: {embedding_result['latency']:.2f}s")
        click.echo(f"Vector Dimension: {len(embedding_result['vector'])}")
        click.echo(f"Timestamp: {embedding_result['timestamp']}")
        click.echo(f"Request ID: {embedding_result['request_id']}")
        click.echo("=" * 40)
        click.echo("\nFirst 5 components:")
        click.echo(embedding_result['vector'][:5])

    except Exception as e:
        logger.error(f"Embedding failed: {e}", exc_info=True)
        click.echo(f"An error occurred during embedding: {e}", err=True)

# --- NEW COMMAND FOR PROJECT PROCESSING ---
@walta_cli.command()
@click.option(
    '--input-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help='Input directory containing project files to enhance.'
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help='Output directory where enhanced project files will be saved.'
)
@click.option(
    '--provider',
    default='gemini',
    help='LLM provider (gemini/openai) to use for enhancement.',
    type=click.Choice(['gemini', 'openai'], case_sensitive=False)
)
async def process(input_dir: str, output_dir: str, provider: str):
    """
    Process and enhance an entire project directory using an LLM.

    This command will iterate through files in the input directory,
    apply LLM-based enhancements, and save results to the output directory.
    """
    try:
        setup_provider_logging(provider)
        llm = WaltaLLM(primary_provider_name=provider)

        click.echo(f"Starting project enhancement with {provider} provider...")
        click.echo(f"Input Directory: {input_dir}")
        click.echo(f"Output Directory: {output_dir}")

        # Ensure output directory exists and is clean
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            click.echo(f"Cleaned up existing output directory: {output_dir}")
        os.makedirs(output_dir)
        click.echo(f"Created output directory: {output_dir}")

        # --- Placeholder for project enhancement logic ---
        # This is where you'll implement the actual file processing,
        # LLM calls (using llm.analyze or llm.generate_text/chat_completion),
        # and writing the enhanced content to the output_dir.

        # Example: Just copying files for now (replace with actual LLM logic)
        for root, _, files in os.walk(input_dir):
            relative_path = os.path.relpath(root, input_dir)
            current_output_dir = os.path.join(output_dir, relative_path)
            os.makedirs(current_output_dir, exist_ok=True) # Ensure subdirectories exist in output

            for file_name in files:
                input_file_path = os.path.join(root, file_name)
                output_file_path = os.path.join(current_output_dir, file_name)
                
                # --- Here's where your LLM logic would go for each file ---
                # Read content:
                # with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                #     file_content = f.read()

                # Call LLM (example):
                # enhanced_content = await llm.analyze(file_content) # Or generate_text/chat_completion

                # Write enhanced content:
                # with open(output_file_path, 'w', encoding='utf-8') as f:
                #     f.write(enhanced_content)
                
                # For now, just copy the file as a placeholder
                shutil.copy2(input_file_path, output_file_path)
                click.echo(f"Copied (or enhanced) file: {file_name}")
        # --- End of placeholder ---

        click.echo("\nProject enhancement complete!")

    except Exception as e:
        logger.error(f"Project processing failed: {e}", exc_info=True)
        click.echo(f"An error occurred during project processing: {e}", err=True)


@walta_cli.command()
def version():
    """Display version information"""
    from . import __version__
    click.echo(f"Walta Framework v{__version__}")
    click.echo(f"Created by: Craig444444444")
    click.echo(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

def main():
    """Entry point for the Walta CLI."""
    asyncio.run(walta_cli())

if __name__ == "__main__":
    main()
