"""OpenTelemetry tracing initialization for LLM calls."""
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from app.utils.logger import logger

# Optional OpenAI instrumentation (must NOT crash app)
try:
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
except ImportError:
    OpenAIInstrumentor = None


def initialize_tracing(
    service_name: str = "smart-document-qa",
    service_version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    tracing_enabled: bool = True,
) -> Optional[TracerProvider]:
    """
    Initialize OpenTelemetry tracing for LLM calls.
    """
    if not tracing_enabled:
        logger.info("Tracing is disabled")
        return None

    try:
        # Resource metadata
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": service_version,
            }
        )

        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Exporter selection
        if otlp_endpoint:
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            logger.info(f"Tracing initialized with OTLP exporter: {otlp_endpoint}")
        else:
            exporter = ConsoleSpanExporter()
            logger.info("Tracing initialized with console exporter")

        tracer_provider.add_span_processor(
            BatchSpanProcessor(exporter)
        )

        # Optional OpenAI instrumentation
        if OpenAIInstrumentor:
            OpenAIInstrumentor().instrument()
            logger.info("OpenAI instrumentation enabled")
        else:
            logger.warning("OpenAI instrumentation not installed — skipping")

        logger.info("OpenTelemetry tracing initialized successfully")
        return tracer_provider

    except Exception as e:
        logger.error("Failed to initialize tracing", exc_info=True)
        return None


def shutdown_tracing(tracer_provider: Optional[TracerProvider]) -> None:
    """Shutdown tracing gracefully."""
    if tracer_provider:
        try:
            tracer_provider.shutdown()
            logger.info("Tracing shutdown completed")
        except Exception as e:
            logger.warning("Error during tracing shutdown", exc_info=True)
