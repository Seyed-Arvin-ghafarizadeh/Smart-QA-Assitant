"""OpenTelemetry tracing initialization for LLM calls."""
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

from app.utils.logger import logger


def initialize_tracing(
    service_name: str = "smart-document-qa",
    service_version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    tracing_enabled: bool = True,
) -> Optional[TracerProvider]:
    """
    Initialize OpenTelemetry tracing for LLM calls.

    Args:
        service_name: Name of the service for traces
        service_version: Version of the service
        otlp_endpoint: Optional OTLP endpoint URL (e.g., http://localhost:4318/v1/traces)
                      If None, uses console exporter
        tracing_enabled: Enable/disable tracing

    Returns:
        TracerProvider instance if tracing is enabled, None otherwise
    """
    if not tracing_enabled:
        logger.info("Tracing is disabled")
        return None

    try:
        # Create resource with service information
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": service_version,
            }
        )

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Configure exporter
        if otlp_endpoint:
            # Use OTLP HTTP exporter for production
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            logger.info(f"Tracing initialized with OTLP exporter: {otlp_endpoint}")
        else:
            # Use console exporter for development
            exporter = ConsoleSpanExporter()
            logger.info("Tracing initialized with console exporter")

        # Add span processor
        span_processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(span_processor)

        # Instrument OpenAI SDK (used for DeepSeek API)
        OpenAIInstrumentor().instrument()

        logger.info("OpenTelemetry tracing initialized successfully")
        return tracer_provider

    except Exception as e:
        logger.error(f"Failed to initialize tracing: {str(e)}", exc_info=True)
        return None


def shutdown_tracing(tracer_provider: Optional[TracerProvider]) -> None:
    """
    Shutdown tracing gracefully.

    Args:
        tracer_provider: TracerProvider instance to shutdown
    """
    if tracer_provider:
        try:
            tracer_provider.shutdown()
            logger.info("Tracing shutdown completed")
        except Exception as e:
            logger.warning(f"Error during tracing shutdown: {str(e)}")

