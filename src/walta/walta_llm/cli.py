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

@walta_cli.command()
def version():
    """Display version information"""
    from . import __version__
    click.echo(f"Walta Framework v{__version__}")
    click.echo(f"Created by: Craig444444444")
    click.echo(f"Last Updated: 2025-07-05 03:20:56 UTC")

def main():
    """Entry point for the Walta CLI."""
    asyncio.run(walta_cli())

if __name__ == "__main__":
    main()
