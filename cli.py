#!/usr/bin/env python3
"""
Command-line interface for the Legal Document Analyzer.
"""
import click
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.rag import rag_pipeline
from src.data.cuad_loader import cuad_loader
from src.utils.monitoring import monitoring_system
from src.utils.logging import app_logger


@click.group()
def cli():
    """Legal Document Analyzer CLI."""
    pass


@cli.command()
@click.argument('file_paths', nargs=-1, type=click.Path(exists=True))
def add_documents(file_paths):
    """Add documents to the knowledge base."""
    if not file_paths:
        click.echo("No files provided.")
        return
    
    click.echo(f"Adding {len(file_paths)} documents...")
    
    try:
        result = rag_pipeline.add_documents(list(file_paths))
        
        click.echo(f"✅ Successfully processed: {len(result['success'])} files")
        click.echo(f"❌ Failed to process: {len(result['failed'])} files")
        click.echo(f"📄 Total chunks created: {result['total_chunks']}")
        
        if result['failed']:
            click.echo("\nFailed files:")
            for failed in result['failed']:
                click.echo(f"  - {failed['file']}: {failed['error']}")
                
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@cli.command()
@click.argument('question')
@click.option('--top-k', default=5, help='Number of sources to consider')
def query(question, top_k):
    """Query the knowledge base."""
    click.echo(f"🔍 Querying: {question}")
    
    try:
        result = rag_pipeline.query(question, top_k=top_k)
        
        click.echo(f"\n💬 Answer:")
        click.echo(result['answer'])
        
        click.echo(f"\n📊 Confidence: {result['confidence']:.2f}")
        
        if result['sources']:
            click.echo(f"\n📑 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources']):
                click.echo(f"{i+1}. Score: {source['score']:.3f}")
                click.echo(f"   {source['content'][:100]}...")
                
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@cli.command()
def stats():
    """Show knowledge base statistics."""
    try:
        stats = rag_pipeline.get_database_stats()
        
        click.echo("📊 Knowledge Base Statistics:")
        click.echo(f"  Documents: {stats.get('total_documents', 0)}")
        click.echo(f"  Vector Dimension: {stats.get('embedding_dimension', 0)}")
        click.echo(f"  Model: {stats.get('model_name', 'Unknown')}")
        click.echo(f"  Queries Made: {stats.get('query_count', 0)}")
        
        if stats.get('query_count', 0) > 0:
            click.echo(f"  Avg Confidence: {stats.get('average_confidence', 0):.2f}")
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@cli.command()
def clear():
    """Clear the knowledge base."""
    if click.confirm("Are you sure you want to clear all documents?"):
        try:
            rag_pipeline.clear_knowledge_base()
            click.echo("✅ Knowledge base cleared")
        except Exception as e:
            click.echo(f"❌ Error: {str(e)}")


@cli.command()
@click.option('--samples', default=5, help='Number of CUAD samples to use')
def benchmark(samples):
    """Run benchmark using CUAD dataset."""
    click.echo(f"🧪 Running benchmark with {samples} CUAD samples...")
    
    try:
        result = cuad_loader.run_cuad_benchmark(samples)
        
        if "error" in result:
            click.echo(f"❌ Benchmark failed: {result['error']}")
            return
        
        click.echo("📈 Benchmark Results:")
        click.echo(f"  Total Questions: {result.get('total_questions', 0)}")
        click.echo(f"  Successful Queries: {result.get('successful_queries', 0)}")
        click.echo(f"  Success Rate: {result.get('success_rate', 0):.2%}")
        click.echo(f"  Average Confidence: {result.get('average_confidence', 0):.2f}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@cli.command()
def analytics():
    """Show analytics report."""
    try:
        report = monitoring_system.generate_analytics_report()
        
        if "error" in report:
            click.echo(f"❌ Analytics failed: {report['error']}")
            return
        
        click.echo("📊 Analytics Report:")
        click.echo(f"  Total Queries: {report.get('total_queries_all_time', 0)}")
        click.echo(f"  Average Confidence: {report.get('average_confidence_all_time', 0):.2f}")
        
        session = report.get('session_analytics', {})
        if session:
            click.echo(f"  Session Queries: {session.get('total_queries', 0)}")
            click.echo(f"  Session Avg Confidence: {session.get('average_confidence', 0):.2f}")
        
        recommendations = report.get('recommendations', [])
        if recommendations:
            click.echo("\n💡 Recommendations:")
            for rec in recommendations:
                click.echo(f"  • {rec}")
                
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@cli.command()
def download_cuad():
    """Download the CUAD dataset."""
    click.echo("📥 Downloading CUAD dataset...")
    
    try:
        success = cuad_loader.download_cuad_dataset()
        if success:
            click.echo("✅ CUAD dataset downloaded successfully")
        else:
            click.echo("❌ Failed to download CUAD dataset")
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


if __name__ == '__main__':
    cli()
