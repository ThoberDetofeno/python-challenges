# Technical Design Document: SupportWise AI Co-pilot

## Executive Summary

This document outlines the technical architecture for SupportWise's AI Co-pilot, an intelligent system that empowers non-technical users to extract actionable insights from Zendesk support data through natural language interactions. The solution balances real-time responsiveness with cost-effectiveness while ensuring data security and system scalability.

## Context and Scope

### Project Context
SupportWise Inc. requires an AI-powered solution to democratize access to customer support insights locked within Zendesk. The system must bridge the gap between complex data engineering requirements and business user needs, enabling natural language querying of millions of support tickets and comments.

### Scope Definition
**In Scope:**
- Natural language interface for querying Zendesk data
- Real-time and batch processing capabilities
- Data visualization generation
- Persistent query/report management
- Sentiment analysis and complex business insights
- Multi-year historical data analysis

**Out of Scope (MVP):**
- Multi-tenant architecture
- Real-time ticket streaming
- Integration with non-Zendesk data sources
- Mobile application

## Requirements

### Functional Requirements

| ID | Requirement | Type | Priority |
|----|------------|------|----------|
| FR1 | Process natural language queries about ticket metrics | Core | P0 |
| FR2 | Generate data visualizations (charts, graphs) | Core | P0 |
| FR3 | Analyze sentiment and extract themes from ticket comments | Core | P0 |
| FR4 | Save and re-run named reports/queries | Core | P0 |
| FR5 | Handle both real-time and batch processing requests | Core | P0 |
| FR6 | Provide query result caching for performance | Enhancement | P1 |
| FR7 | Support export of results to common formats | Enhancement | P2 |

### Non-Functional Requirements

| ID | Requirement | Type | Priority |
|----|------------|------|----------|
| NFR1 | Response time <3 seconds for simple queries | Performance | P0 |
| NFR2 | Support analysis of 5M+ tickets | Scalability | P0 |
| NFR3 | 99.9% uptime for query interface | Reliability | P0 |
| NFR4 | Encrypt all data at rest and in transit | Security | P0 |
| NFR5 | Audit log all data access and queries | Compliance | P0 |
| NFR6 | Cost optimization for variable workloads | Operational | P1 |
| NFR7 | Provider-agnostic AI model integration | Maintainability | P1 |

## System Architecture Overview

The SupportWise AI Co-pilot employs a modular, event-driven architecture with clear separation of concerns:

### Core Components

**1. API Gateway & Chat Interface**
- WebSocket-based real-time communication layer
- RESTful API for structured requests
- Session management and rate limiting
- Request routing to appropriate processing pipelines

**2. Query Orchestrator**
- Natural language understanding and intent classification
- Query complexity assessment and routing logic
- Parallel execution management for complex queries
- Result aggregation and formatting

**3. Data Pipeline Layer**
- **ETL Pipeline**: Scheduled Zendesk data extraction and transformation
- **Data Warehouse**: Columnar storage optimized for analytical queries
- **Vector Store**: Embeddings for semantic search and RAG operations
- **Cache Layer**: Multi-tier caching for query results and intermediate data

**4. AI Processing Engine**
- **LLM Gateway**: Abstraction layer for multiple LLM providers
- **Prompt Engineering Module**: Dynamic prompt construction and optimization
- **Embedding Service**: Text vectorization for semantic search
- **Fine-tuning Pipeline**: Domain-specific model adaptation

**5. Visualization Service**
- Chart generation engine (supporting multiple chart types)
- Template-based report builder
- Export functionality (PNG, PDF, CSV)

**6. Metadata & Configuration Store**
- User preferences and saved queries
- System configuration and feature flags
- Query templates and visualization preferences

## Data Flow Description

### Primary Data Flow Path

1. **Data Ingestion (Batch)**
   - Scheduled job triggers Zendesk API extraction every 15 minutes
   - Raw data lands in staging tables (S3/GCS)
   - Transformation pipeline cleanses and enriches data
   - Processed data loads into analytical warehouse
   - Embeddings generated for comment text and stored in vector database

2. **Query Processing Flow**
   - User submits natural language query via chat interface
   - Query Orchestrator classifies intent and complexity
   - Simple queries route to SQL generation pipeline
   - Complex queries trigger multi-step RAG pipeline
   - Results cached with TTL based on data freshness requirements

3. **Visualization Generation**
   - Structured data passed to visualization service
   - Chart type determined by data characteristics and user preference
   - Rendered visualization returned as base64 encoded image
   - Metadata stored for future reference

## Technology Choices

### Infrastructure Layer
- **Cloud Provider**: AWS (primary) with multi-cloud abstraction layer
- **Container Orchestration**: Kubernetes (EKS) for service deployment
- **Message Queue**: Apache Kafka for event streaming
- **Justification**: AWS provides mature managed services; Kubernetes enables portability; Kafka handles high-throughput event processing

### Data Layer
- **Data Warehouse**: Snowflake
- **Vector Database**: Pinecone
- **Cache**: Redis with ElastiCache
- **Object Storage**: S3 for raw data and backups
- **Justification**: Snowflake excels at analytical queries; Pinecone optimized for vector similarity; Redis provides sub-millisecond latency

### AI/ML Stack
- **Primary LLM**: OpenAI GPT-4 with fallback to Claude 3
- **Embeddings**: OpenAI text-embedding-3-small
- **ML Framework**: LangChain for orchestration
- **Justification**: GPT-4 leads in reasoning capability; LangChain provides abstraction for provider switching

### Application Layer
- **Backend**: Python (FastAPI) for API services
- **Frontend**: React with TypeScript
- **Visualization**: D3.js with Plotly fallback
- **Justification**: Python dominates AI/ML ecosystem; React provides rich interactive UX; D3.js offers maximum visualization flexibility

## Data Schema Structures

### Core Data Models

```python
# Warehouse Schema (Snowflake)
class TicketFact:
    ticket_id: str  # Primary key
    created_at: timestamp
    updated_at: timestamp
    status: str
    priority: str
    tags: array<str>
    assignee_id: str
    requester_id: str
    custom_fields: json
    resolution_time_hours: float
    first_response_time_minutes: float
    
class CommentDimension:
    comment_id: str  # Primary key
    ticket_id: str  # Foreign key
    author_type: str  # 'agent' or 'customer'
    created_at: timestamp
    body_text: text
    sentiment_score: float  # -1 to 1
    extracted_topics: array<str>
    embedding_id: str  # Reference to vector store

# Vector Store Schema
class CommentEmbedding:
    embedding_id: str
    vector: array<float>  # 1536 dimensions
    ticket_id: str
    comment_id: str
    metadata: {
        created_at: timestamp
        priority: str
        tags: array<str>
    }

# Query Cache Schema
class CachedQuery:
    query_hash: str  # Primary key
    natural_language_query: str
    sql_query: str
    result_data: json
    created_at: timestamp
    ttl_seconds: int
    access_count: int
```

### API Contracts

```python
# Query Request
class QueryRequest:
    query: str
    user_id: str
    session_id: str
    context: Optional[dict]  # Previous query context
    visualization_preference: Optional[str]
    
# Query Response
class QueryResponse:
    query_id: str
    status: str  # 'complete', 'processing', 'error'
    result_type: str  # 'data', 'visualization', 'insight'
    data: Optional[dict]
    visualization: Optional[str]  # Base64 encoded image
    natural_language_response: str
    execution_time_ms: int
    confidence_score: float
```

## Detailed Technical Explanations

### 1. User Experience Flow

When Brenda types "How many urgent tickets did we get last week?":

**Step-by-step flow:**
1. WebSocket connection transmits query to API Gateway
2. Query Orchestrator receives request and initiates processing
3. Intent classifier identifies this as a "simple metric query"
4. SQL generator constructs: `SELECT COUNT(*) FROM tickets WHERE priority='urgent' AND created_at >= DATE_SUB(CURRENT_DATE, 7)`
5. Query executor checks cache (Redis) for recent identical queries
6. If cache miss, executes against Snowflake
7. Result formatted with natural language wrapper: "You received 127 urgent tickets last week"
8. Response streamed back via WebSocket in <2 seconds

**Key technical challenges:**
- **Query understanding ambiguity**: "Last week" could mean last 7 days or previous calendar week. Solution: Implement context-aware date parsing with user preference learning
- **Real-time feel**: Implement optimistic UI updates and progressive result streaming
- **Error handling**: Graceful degradation with helpful error messages when queries fail

### 2. Historical Data Challenges

**Challenge**: Analyzing millions of tickets while maintaining sub-second response times.

**Solution Architecture:**
- **Pre-aggregation Strategy**: Materialized views for common metrics (daily/weekly/monthly aggregates)
- **Partitioning Scheme**: Partition by created_at with monthly granularity
- **Indexing Strategy**: Composite indexes on (priority, status, created_at) for common query patterns
- **Caching Hierarchy**:
  - L1: Redis for exact query matches (TTL: 5 minutes)
  - L2: Snowflake result cache (TTL: 1 hour)
  - L3: Pre-computed aggregates refreshed hourly

**Asynchronous Processing**: For complex historical analyses, implement job queue with progress tracking:
```python
async def handle_complex_query(query):
    job_id = queue_analysis_job(query)
    notify_user("Analysis started, estimated time: 2 minutes")
    # Background worker processes
    result = await process_historical_data(query)
    notify_user("Analysis complete", result)
```

### 3. Business Data Risks

**Identified Risks and Mitigations:**

**Data Exposure Risks:**
- **Risk**: LLM hallucinations exposing incorrect metrics
- **Mitigation**: Implement fact-checking layer that validates LLM responses against source data

**Query Injection Risks:**
- **Risk**: Malicious queries causing data exfiltration
- **Mitigation**: Parameterized queries only; no direct SQL execution from LLM output

**Privacy Concerns:**
- **Risk**: PII exposure in customer comments
- **Mitigation**: PII detection and masking pipeline before LLM processing

**Access Control:**
- **Risk**: Unauthorized data access
- **Mitigation**: Row-level security with user context validation

**Audit Requirements:**
```python
class AuditLog:
    user_id: str
    query: str
    data_accessed: list[str]  # Tables/fields accessed
    timestamp: datetime
    ip_address: str
    result_row_count: int
```

### 4. Complex Business Questions

**Approach for "Why are customers unhappy with our new feature?"**

**Multi-stage Pipeline:**
1. **Feature Identification**: Use NER to identify "new feature" references in tickets
2. **Sentiment Analysis**: Apply fine-tuned sentiment model to relevant comments
3. **Theme Extraction**: Use LLM to extract common complaint patterns
4. **Correlation Analysis**: Identify statistical correlations with ticket metadata

**Implementation Strategy:**
```python
def analyze_feature_sentiment(feature_name):
    # Stage 1: Retrieve relevant tickets
    relevant_tickets = vector_search(
        query=f"issues problems {feature_name}",
        filters={"created_at": "last_30_days"}
    )
    
    # Stage 2: Sentiment analysis
    sentiments = batch_analyze_sentiment(relevant_tickets)
    
    # Stage 3: Theme extraction via LLM
    themes = extract_themes_llm(
        negative_comments=filter_negative(sentiments),
        prompt_template="identify_pain_points"
    )
    
    # Stage 4: Generate insights
    return generate_insight_report(themes, sentiments)
```

### 5. Workflow Efficiency

**Solution: Named Query System with Smart Scheduling**

**Architecture:**
```python
class SavedQuery:
    query_id: str
    name: str  # "Weekly Ticket Report"
    query_template: str
    parameters: dict  # Dynamic date ranges
    schedule: Optional[CronExpression]
    output_format: str
    notification_preferences: dict
```

**Features:**
- **Natural language aliasing**: "Run my Monday report" → retrieve and execute saved query
- **Parameterization**: Automatic date range updates for recurring reports
- **Proactive execution**: Pre-compute scheduled reports during off-peak hours
- **Version control**: Track query modifications with rollback capability

### 6. Data Visualization Needs

**Technical Implementation:**

**Visualization Pipeline:**
1. Data shape analysis determines optimal chart type
2. D3.js renders chart in headless browser (Puppeteer)
3. Screenshot captured and optimized
4. Base64 encoded image returned with metadata

**Challenges and Solutions:**
- **Challenge**: Chart type selection
- **Solution**: Rule engine + LLM suggestion with user override

- **Challenge**: Performance of image generation
- **Solution**: Pre-rendered template library for common chart types

- **Challenge**: Interactivity requirements
- **Solution**: Progressive enhancement - static image first, interactive version on-demand

```python
class VisualizationEngine:
    def generate_chart(self, data, chart_type=None):
        if not chart_type:
            chart_type = self.infer_chart_type(data)
        
        if chart_type in TEMPLATE_CHARTS:
            return self.render_template(chart_type, data)
        else:
            return self.render_custom_d3(chart_type, data)
```

### 7. Operational Cost Management

**Cost Optimization Strategy:**

**Query Classification and Routing:**
```python
class QueryClassifier:
    def classify(self, query):
        complexity = self.estimate_complexity(query)
        if complexity == "simple":
            return "direct_sql"  # Lowest cost
        elif complexity == "moderate":
            return "cached_aggregates"  # Medium cost
        else:
            return "llm_pipeline"  # Highest cost
```

**Cost Control Mechanisms:**
- **Caching aggressive**: 80% cache hit rate target
- **Query complexity limits**: Timeout and row limits for expensive operations
- **Batch processing**: Aggregate similar queries for batch LLM calls
- **Tiered processing**: Use smaller models for simple tasks, GPT-4 only when necessary
- **Usage quotas**: Implement daily LLM token budgets with alerts

**Monitoring Dashboard:**
- Real-time cost tracking per query type
- User-level cost attribution
- Automated cost anomaly detection

### 8. Technology Evolution

**Future-Proofing Architecture:**

**Abstraction Layers:**
```python
class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt, **kwargs):
        pass

class OpenAIProvider(LLMProvider):
    def complete(self, prompt, **kwargs):
        # OpenAI-specific implementation
        
class AnthropicProvider(LLMProvider):
    def complete(self, prompt, **kwargs):
        # Anthropic-specific implementation
```

**Strategies:**
- **Provider abstraction**: All AI services behind interfaces
- **Feature flags**: Gradual rollout of new capabilities
- **Model versioning**: Support multiple model versions simultaneously
- **Fallback chains**: Automatic failover to alternative providers
- **Continuous evaluation**: A/B testing framework for model performance

**Migration Path Planning:**
- Modular architecture enables component-level updates
- Comprehensive integration tests for provider switching
- Data export capabilities to prevent vendor lock-in

## Risk Evaluation

### Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|-------------------|
| LLM hallucination causing incorrect insights | High | High | Implement validation layer, confidence scoring |
| Zendesk API rate limiting | Medium | High | Implement adaptive rate limiting, caching layer |
| Query performance degradation at scale | Medium | High | Pre-aggregation, query optimization, monitoring |
| Provider outage (OpenAI/Pinecone) | Low | High | Multi-provider fallback, self-hosted alternatives |
| Data freshness vs. cost trade-off | High | Medium | Tiered refresh strategies, user expectations setting |

### Business Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|-------------------|
| User adoption challenges | Medium | High | Intuitive UX, comprehensive onboarding |
| Compliance/regulatory issues | Low | High | Audit logging, data governance framework |
| Scope creep beyond MVP | High | Medium | Clear roadmap, stakeholder alignment |
| Cost overruns from AI usage | Medium | Medium | Usage monitoring, budget alerts, optimization |

## Implementation Roadmap

### Phase 1 (Weeks 1-4): Foundation
- Set up data pipeline from Zendesk
- Implement basic SQL query generation
- Deploy simple chat interface

### Phase 2 (Weeks 5-8): Intelligence Layer
- Integrate LLM for natural language understanding
- Implement sentiment analysis pipeline
- Add visualization generation

### Phase 3 (Weeks 9-12): Advanced Features
- Build saved query system
- Implement complex multi-step analyses
- Add comprehensive monitoring and optimization

## Success Metrics

- Query response time P95 < 3 seconds
- System uptime > 99.9%
- User satisfaction score > 4.5/5
- 50% reduction in time to generate reports
- Cost per query < $0.10 average

## Conclusion

This architecture provides a robust, scalable foundation for the SupportWise AI Co-pilot while maintaining flexibility for future evolution. The design prioritizes user experience, cost efficiency, and data security while acknowledging the inherent trade-offs in building an AI-powered analytics system. The modular approach ensures we can iterate quickly on the MVP while building toward a comprehensive enterprise solution.
