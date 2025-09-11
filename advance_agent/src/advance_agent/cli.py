#!/usr/bin/env python3
"""Command-line interface for the Advanced Agent System."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import click
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax

from .core.config import get_settings
from .core.logging import setup_logging, get_logger
from .core.security import create_security_context
from .main import agent_system, app
from .models.schemas import AgentConfiguration, SecurityClearance
from .examples import main as run_examples


console = Console()
logger = get_logger("cli")


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
def cli(debug: bool, config: Optional[str]):
    """Advanced Agent System CLI - Enterprise-grade AI agent management."""
    
    if debug:
        click.echo("Debug mode enabled")
    
    if config:
        click.echo(f"Using config file: {config}")
    
    # Setup logging
    setup_logging()


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def server(host: str, port: int, reload: bool):
    """Start the Advanced Agent System server."""
    
    console.print(Panel(
        "[bold blue]Advanced Agent System[/bold blue]\n"
        "Enterprise-grade AI agent management system\n"
        f"Starting server on {host}:{port}",
        title="🤖 Agent System",
        expand=False
    ))
    
    try:
        uvicorn.run(
            "advance_agent.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Server error: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
async def status():
    """Show system status and metrics."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Checking system status...", total=None)
        
        try:
            # Initialize system if needed
            if not agent_system.is_running:
                await agent_system.startup()
            
            status_data = agent_system.get_system_status()
            
            progress.update(task, description="Generating status report...")
            
            # Create status table
            table = Table(title="System Status", show_header=True, header_style="bold magenta")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="white")
            
            # System status
            table.add_row(
                "System",
                status_data["status"],
                f"Uptime: {status_data.get('uptime_seconds', 0):.0f}s"
            )
            
            # Agents
            agent_count = len(status_data.get("agents", {}))
            table.add_row(
                "Agents",
                f"{agent_count} active",
                f"Total created: {status_data['metrics']['system']['total_agents_created']}"
            )
            
            # Requests
            total_requests = status_data['metrics']['system']['total_requests']
            success_rate = (
                status_data['metrics']['system']['successful_requests'] / max(total_requests, 1) * 100
            )
            table.add_row(
                "Requests",
                f"{total_requests} total",
                f"Success rate: {success_rate:.1f}%"
            )
            
            # Guardrails
            guardrail_metrics = status_data['metrics']['guardrails']
            table.add_row(
                "Guardrails",
                f"{guardrail_metrics['total_evaluations']} evaluations",
                f"Violations: {guardrail_metrics['total_violations']}"
            )
            
            # Tracing
            tracing_metrics = status_data['metrics']['tracing']
            table.add_row(
                "Tracing",
                f"{tracing_metrics['traces_created']} traces",
                f"Spans: {tracing_metrics['spans_created']}"
            )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error getting status: {str(e)}[/red]")
        finally:
            progress.stop()


@cli.command()
@click.option('--name', required=True, help='Agent name')
@click.option('--description', required=True, help='Agent description')
@click.option('--instructions', required=True, help='Agent instructions')
@click.option('--model', default='gpt-4', help='Model to use')
@click.option('--temperature', default=0.7, type=float, help='Temperature setting')
@click.option('--max-tokens', default=4096, type=int, help='Maximum tokens')
@click.option('--tools/--no-tools', default=True, help='Enable tools')
@click.option('--clearance', 
              type=click.Choice(['public', 'internal', 'confidential', 'restricted', 'top_secret']),
              default='public', help='Security clearance level')
async def create_agent(name: str, description: str, instructions: str, model: str, 
                      temperature: float, max_tokens: int, tools: bool, clearance: str):
    """Create a new agent."""
    
    console.print(f"[bold blue]Creating agent: {name}[/bold blue]")
    
    try:
        # Initialize system if needed
        if not agent_system.is_running:
            await agent_system.startup()
        
        # Create security context
        security_context = create_security_context(
            user_id="cli_user",
            clearance_level="top_secret",
            permissions=["manage_agents"]
        )
        
        # Create agent configuration
        config = AgentConfiguration(
            name=name,
            description=description,
            instructions=instructions,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools_enabled=tools,
            security_clearance=SecurityClearance(clearance)
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Creating agent...", total=None)
            
            agent = await agent_system.create_agent(config, security_context)
            
            progress.update(task, description="Agent created successfully!")
        
        console.print(Panel(
            f"[green]Agent created successfully![/green]\n\n"
            f"[bold]ID:[/bold] {agent.id}\n"
            f"[bold]Name:[/bold] {agent.name}\n"
            f"[bold]Status:[/bold] {agent.state.status}\n"
            f"[bold]Tools Enabled:[/bold] {config.tools_enabled}\n"
            f"[bold]Security Clearance:[/bold] {config.security_clearance.value}",
            title="🤖 Agent Created",
            expand=False
        ))
        
    except Exception as e:
        console.print(f"[red]Error creating agent: {str(e)}[/red]")


@cli.command()
async def list_agents():
    """List all agents."""
    
    try:
        # Initialize system if needed
        if not agent_system.is_running:
            await agent_system.startup()
        
        from .main import agents_registry
        
        if not agents_registry:
            console.print("[yellow]No agents found[/yellow]")
            return
        
        # Create agents table
        table = Table(title="Active Agents", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Tools", style="blue")
        table.add_column("Requests", style="white")
        table.add_column("Last Activity", style="dim")
        
        for agent in agents_registry.values():
            status = agent.get_status()
            table.add_row(
                agent.id[:8] + "...",
                agent.name,
                status["status"],
                "Yes" if agent.capabilities.can_use_tools else "No",
                str(status["metrics"]["total_requests"]),
                status["last_activity"]
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing agents: {str(e)}[/red]")


@cli.command()
@click.argument('agent_id')
@click.argument('message')
async def chat(agent_id: str, message: str):
    """Send a message to an agent."""
    
    console.print(f"[bold blue]Sending message to agent: {agent_id[:8]}...[/bold blue]")
    
    try:
        # Initialize system if needed
        if not agent_system.is_running:
            await agent_system.startup()
        
        # Create security context
        security_context = create_security_context(
            user_id="cli_user",
            clearance_level="internal",
            permissions=["basic_chat"]
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing message...", total=None)
            
            result = await agent_system.process_message(
                agent_id=agent_id,
                message=message,
                security_context=security_context
            )
            
            progress.update(task, description="Message processed!")
        
        # Display response
        console.print(Panel(
            f"[bold]Response:[/bold]\n{result.get('response', 'No response')}\n\n"
            f"[dim]Tokens used: {result.get('tokens_used', 0)} | "
            f"Response time: {result.get('response_time', 0):.2f}s[/dim]",
            title="🤖 Agent Response",
            expand=False
        ))
        
    except Exception as e:
        console.print(f"[red]Error processing message: {str(e)}[/red]")


@cli.command()
async def examples():
    """Run comprehensive examples demonstrating all system features."""
    
    console.print(Panel(
        "[bold blue]Running Comprehensive Examples[/bold blue]\n"
        "This will demonstrate all Advanced Agent System features:\n"
        "• Agent creation and management\n"
        "• Tool integration\n"
        "• Guardrail system\n"
        "• Hook system\n"
        "• Distributed tracing\n"
        "• Workflow orchestration\n"
        "• Security and permissions\n"
        "• Error handling",
        title="🚀 Examples",
        expand=False
    ))
    
    try:
        await run_examples()
        console.print("[green]✓ All examples completed successfully![/green]")
    except Exception as e:
        console.print(f"[red]Examples failed: {str(e)}[/red]")


@cli.command()
@click.option('--output', type=click.Path(), help='Output file for metrics')
async def metrics(output: Optional[str]):
    """Show system metrics."""
    
    try:
        # Initialize system if needed
        if not agent_system.is_running:
            await agent_system.startup()
        
        status_data = agent_system.get_system_status()
        metrics_data = status_data["metrics"]
        
        if output:
            # Save to file
            with open(output, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            console.print(f"[green]Metrics saved to {output}[/green]")
        else:
            # Display in console
            console.print(Syntax(
                json.dumps(metrics_data, indent=2, default=str),
                "json",
                theme="monokai",
                line_numbers=True
            ))
    
    except Exception as e:
        console.print(f"[red]Error getting metrics: {str(e)}[/red]")


@cli.command()
@click.option('--limit', default=10, type=int, help='Number of traces to show')
@click.option('--user-id', help='Filter by user ID')
async def traces(limit: int, user_id: Optional[str]):
    """Show trace data."""
    
    try:
        from .tracing.base import global_tracing_processor
        
        if user_id:
            traces_data = global_tracing_processor.get_traces_by_user(user_id, limit)
        else:
            traces_data = global_tracing_processor.get_recent_traces(limit)
        
        if not traces_data:
            console.print("[yellow]No traces found[/yellow]")
            return
        
        # Create traces table
        table = Table(title="Recent Traces", show_header=True, header_style="bold magenta")
        table.add_column("Trace ID", style="cyan")
        table.add_column("Operation", style="green")
        table.add_column("User", style="yellow")
        table.add_column("Spans", style="blue")
        table.add_column("Duration", style="white")
        table.add_column("Status", style="dim")
        
        for trace in traces_data:
            duration = ""
            if trace.end_time:
                duration_seconds = (trace.end_time - trace.start_time).total_seconds()
                duration = f"{duration_seconds:.2f}s"
            
            root_span = trace.get_root_span()
            operation_name = root_span.operation_name if root_span else "Unknown"
            
            table.add_row(
                trace.trace_id[:8] + "...",
                operation_name,
                trace.user_id or "Unknown",
                str(len(trace.spans)),
                duration,
                "Completed" if trace.end_time else "Active"
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error getting traces: {str(e)}[/red]")


@cli.command()
async def tools():
    """List available tools."""
    
    try:
        from .tools.base import default_tool_registry
        
        tools_data = default_tool_registry.get_all_tools()
        
        if not tools_data:
            console.print("[yellow]No tools available[/yellow]")
            return
        
        # Create tools table
        table = Table(title="Available Tools", show_header=True, header_style="bold magenta")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Security Clearance", style="yellow")
        table.add_column("Parameters", style="blue")
        table.add_column("Usage", style="white")
        
        for tool_name, tool in tools_data.items():
            table.add_row(
                tool_name,
                tool.definition.description[:50] + "..." if len(tool.definition.description) > 50 else tool.definition.description,
                tool.definition.security_clearance,
                str(len(tool.definition.parameters)),
                f"{tool.call_count} calls"
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing tools: {str(e)}[/red]")


@cli.command()
@click.option('--config-file', type=click.Path(), help='Configuration file to validate')
def validate_config(config_file: Optional[str]):
    """Validate system configuration."""
    
    try:
        settings = get_settings()
        
        # Create validation table
        table = Table(title="Configuration Validation", show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        # Validate key settings
        validations = [
            ("OpenAI API Key", "***" if settings.openai_api_key else "Not set", 
             "✓" if settings.openai_api_key else "✗"),
            ("Database URL", settings.database_url, "✓"),
            ("Redis URL", settings.redis_url, "✓"),
            ("Secret Key", "***" if settings.secret_key else "Not set",
             "✓" if settings.secret_key else "✗"),
            ("Debug Mode", str(settings.debug), "✓"),
            ("Port", str(settings.port), "✓"),
        ]
        
        for setting, value, status in validations:
            table.add_row(setting, value, status)
        
        console.print(table)
        
        # Check for critical issues
        issues = []
        if not settings.openai_api_key:
            issues.append("OpenAI API key is not set")
        if not settings.secret_key:
            issues.append("Secret key is not set")
        
        if issues:
            console.print(Panel(
                "\n".join(f"• {issue}" for issue in issues),
                title="⚠️ Configuration Issues",
                style="red"
            ))
        else:
            console.print("[green]✓ Configuration is valid[/green]")
        
    except Exception as e:
        console.print(f"[red]Error validating configuration: {str(e)}[/red]")


@cli.command()
@click.confirmation_option(prompt="Are you sure you want to reset the system?")
async def reset():
    """Reset the system (cleanup all agents and data)."""
    
    console.print("[yellow]Resetting system...[/yellow]")
    
    try:
        if agent_system.is_running:
            await agent_system.shutdown()
        
        # Clear registries
        from .main import agents_registry
        agents_registry.clear()
        
        # Reset metrics
        agent_system.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "active_agents": 0,
            "total_agents_created": 0
        }
        
        console.print("[green]✓ System reset complete[/green]")
        
    except Exception as e:
        console.print(f"[red]Error resetting system: {str(e)}[/red]")


def main():
    """Main CLI entry point."""
    
    # Handle async commands
    import inspect
    
    original_cli = cli
    
    def async_command_wrapper(f):
        if inspect.iscoroutinefunction(f):
            def wrapper(*args, **kwargs):
                return asyncio.run(f(*args, **kwargs))
            return wrapper
        return f
    
    # Wrap async commands
    for command_name in cli.commands:
        command = cli.commands[command_name]
        if hasattr(command, 'callback') and inspect.iscoroutinefunction(command.callback):
            command.callback = async_command_wrapper(command.callback)
    
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"[red]CLI error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
