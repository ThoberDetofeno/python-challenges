# Technical Design Document: SupportWise AI Co-pilot

## Executive Summary

This document outlines the technical architecture for SupportWise's AI Co-pilot, an intelligent system that empowers non-technical users to extract actionable insights from Zendesk support data through natural language interactions. The solution balances real-time responsiveness with cost-effectiveness while ensuring data security and system scalability.

*The solution presented below is intended for a quick MVP, without going into deep implementation details or relying on any specific cloud environment.*

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

### Functional Requirements (FR)

| ID | Requirement | Priority | Type |
|---|---|---|---|
| FR-001 | Support natural language queries for ticket metrics and aggregations | P0 | Core Feature |
| FR-002 | Generate data visualizations (bar charts, line graphs, pie charts) from query results | P0 | Core Feature |
| FR-003 | Save and re-run report templates with dynamic date parameters | P0 | Core Feature |
| FR-004 | Analyze sentiment and extract insights from ticket comments | P1 | Advanced Feature |
| FR-005 | Support follow-up questions with conversation context | P1 | UX Enhancement |
| FR-006 | Schedule automated report generation and delivery | P2 | Automation |
| FR-007 | Export visualizations in multiple formats (PNG, SVG, HTML) | P2 | Data Export |
| FR-008 | Provide query suggestions based on historical usage | P2 | UX Enhancement |

### Non-Functional Requirements (NFR)

| ID | Requirement | Target | Type |
|---|---|---|---|
| NFR-001 | Response time for simple queries | < 5 seconds | Performance |
| NFR-002 | Response time for complex analytical queries | < 45 seconds (with progress updates) | Performance |
| NFR-003 | System availability | 99.9% uptime | Reliability |
| NFR-004 | Concurrent user support | Max 200 simultaneous users | Scalability |
| NFR-005 | Data freshness for real-time metrics | < 30 minutes lag | Data Quality |
| NFR-006 | Historical data retention | Max 3 years | Data Management |
| NFR-007 | Query result accuracy | 99.5% accuracy vs. manual validation | Data Quality |
| NFR-008 | Maximum data processing volume | Max 5M tickets, and 100M comments | Scalability |

### Security Requirements (SR)

| ID | Requirement | Implementation | Type |
|---|---|---|---|
| SR-001 | Enforce read-only access to production data | SQL validation layer | Data Protection |
| SR-002 | Implement role-based access control (RBAC) | JWT tokens with role claims | Access Control |
| SR-003 | Encrypt sensitive data at rest | AES-256 encryption | Data Protection |
| SR-004 | Encrypt data in transit | TLS 1.3 minimum | Data Protection |
| SR-005 | Audit log all queries and data access | Immutable audit trail in separate database | Compliance |
| SR-006 | Mask PII in query results | Automated PII detection and redaction | Privacy |
| SR-007 | Prevent prompt injection attacks | Input sanitization and validation | AI Safety |
| SR-008 | Implement API rate limiting | Token bucket algorithm per user/role | DDoS Protection |

### Integration Requirements (IR)

| ID | Requirement | Method | Type |
|---|---|---|---|
| IR-001 | Synchronize with Zendesk API | REST API with incremental sync | External System |
| IR-002 | Export to business intelligence tools | Standard SQL interface or API | Data Export |
| IR-003 | Integrate with notifications tools | API for alerts | Communication |
| IR-004 | Support SSO authentication | SAML 2.0 / OAuth 2.0 | Authentication |
| IR-005 | Connect to multiple LLM providers | Abstracted provider interface | AI Services |

### Operational Requirements (OR)

| ID | Requirement | Specification | Type |
|---|---|---|---|
| OR-001 | Automated backup and recovery | Daily backups with 30-day retention | Disaster Recovery |
| OR-002 | Monitoring and alerting | Real-time metrics with threshold alerts | Observability |
| OR-003 | Cost tracking per query/user | Detailed usage analytics | Cost Management |
| OR-004 | Multi-region deployment support | Active-passive DR setup | Geographic Distribution |

### Compliance Requirements (CR)

| ID | Requirement | Standard | Type |
|---|---|---|---|
| CR-001 | GDPR compliance for EU customer data | Right to deletion, data portability | Legal |
| CR-002 | Data residency requirements | Configurable data location | Regional |

## 1. System Architecture Overview

### Core Components

The architecture follows a modular, event-driven design with clear separation of concerns:

**Frontend Layer**
- React-based chat interface with real-time WebSocket connections
- Visualization rendering engine using D3.js/Recharts
- Progressive web app for cross-platform compatibility
- **Requirements Addressed**: FR-001, FR-002, FR-005, NFR-001

**API Gateway & Orchestration**
- FastAPI application server handling HTTP/WebSocket connections
- Request router determining query complexity and routing strategy
- Session management and user context tracking
- **Requirements Addressed**: NFR-004, SR-008, FR-005

**AI Agent Platform**
- LangGraph how agent orchestration engine
- Ability to create agents for specific tasks
- Orchestrate multiple agents
- Short-term memory management
- Connection with the MCP Server
- **Requirements Addressed**: FR-001, FR-005, SR-007, IR-005, OR-002
  
**MCP (Model Context Protocol) Server**
- Python framework FastMCP to implement the MCP
- Tools allow servers execute functions that can be invoked by clients and used by Agents to perform actions.
- Resources that provide contextual information to AI applications
- Prompts templates that help structure interactions with LLM
- **Requirements Addressed**: FR-003, IR-005  

**Query Processing**
- Framework LangChain to facilitate the integration of LLMs into applications
- Natural Language Understanding (NLU) module using LLM (GPT-4, Claude, Gemini)
- SQL/query generation layer with validation and safety checks
- Adaptive query planner choosing between cached, real-time, or batch processing paths
- **Requirements Addressed**: FR-001, SR-001, SR-007, NFR-001, NFR-002

**Data Layer**
- PostgreSQL as primary analytical database
- Pgvector is a Postgres extension for semantic search over ticket comments (vector database)
- Redis for caching frequently accessed data and query results
- Pandas for medium-scale in-memory computations
- PostgreSQL with TimescaleDB extension (time-series optimization) for processing of historical analyses
- S3-compatible object storage for historical data archival and large result sets
- **Requirements Addressed**: FR-004, NFR-002, NFR-006, NFR-008, SR-003, OR-001

**Integration Layer**
- Zendesk API connector with incremental sync capabilities
- Change Data Capture (CDC) pipeline
- Data validation and transformation services
- **Requirements Addressed**: IR-001, NFR-005

**Background Processing**
- Celery task queue for asynchronous job processing
- Scheduled jobs for data synchronization and cache warming
- Result notification service via WebSocket/email
- **Requirements Addressed**: FR-006, IR-003, OR-003

## 2. Data Flow Description

### Real-time Query Flow (< 5 seconds response)
**Journey 1**: Basic Reporting
1. **User Input**: Brenda types "How many urgent tickets were created last week?" in the chat interface
2. **WebSocket Transmission**: Query sent to API gateway via persistent WebSocket connection
3. **Query Classification**: Determining query complexity and routing strategy
4. **Cache Check**: System checks Redis for recent identical queries (TTL: 1 hour for real-time metrics)
5. **AI Agent**: AI Agent identifies this as a simple query
6. **SQL Generation**: If not cached, LLM generates parameterized SQL query
7. **Query Execution**: PostgreSQL executes optimized query against indexed columns
8. **Pos-Processing**: Result is format as JSON with metadata (query time, data freshness)
9. **Answer**: Result pushed back through WebSocket

**Journey 2**: Business Insights
1. **User Input**: Brenda types "We saw a 15% spike in 'high' priority tickets last month. What were our customers complaining about?" in the chat interface
2. **WebSocket Transmission**: Query sent to API gateway via persistent WebSocket connection
3. **Query Classification**: Determining query complexity and routing strategy
4. **Cache Check**: System checks Redis for recent identical queries (TTL: 1 hour for real-time metrics)
5. **AI Agent**: Identifies this as Resource (MCP Server)
6. **MCP Server**: Execute the Resource function and return the data needed for this question
7. **AI Agent**: AI Agent generates the Response
8. **Pos-Processing**: Result is format as JSON with metadata (query time, data freshness)
9. **Answer**: Result pushed back through WebSocket

**Requirements Validated**: NFR-001 (< 5 seconds), SR-001 (read-only), NFR-005 (data freshness)

### Complex Analysis Flow (Background Processing)
**Journey 3**: Persistent, Reusable Analyses 
1. **User Input**: Brenda types "Show me a bar chart of tickets created per day last week." in the chat interface
2. **WebSocket Transmission**: Query sent to API gateway via persistent WebSocket connection
3. **Query Classification**: Determining query complexity and routing strategy
4. **AI Agent**: Complex Query Detection, system identifies queries requiring historical analysis or ML processing
5. **MCP Server**: Execute the Tools function and return the artefact with the chart
6. **AI Agent**: AI Agent generates the Response
7. **Pos-Processing**: Result is format as JSON with metadata (query time, data freshness)
8. **Answer**: Result pushed back through WebSocket with the option to salve the artefact and create a schedule job
9. **User Input**: Brenda select "Schedule job" option to create the same report every week.
10. **Job Creation**: Job created with query parameters and estimated completion time
11. **Immediate Acknowledgment**: User receives confirmation with job ID and progress tracking
12. **Notification**: User notified via UI notification and optional email when complete

**Requirements Validated**: NFR-002 (< 45 seconds with updates), NFR-008 (5M tickets)

### Data Synchronization Flow
1. **Incremental Sync**: Every 15 minutes, poll Zendesk API for updated tickets/comments
2. **CDC Pipeline**: Transform and validate incoming data
3. **Database Updates**: Upsert changes to PostgreSQL maintaining referential integrity
4. **Cache Invalidation**: Invalidate affected cache entries in Redis
5. **Embedding Updates**: Queue updated comments for re-embedding in vector database

**Requirements Validated**: IR-001, NFR-005 (< 30 minutes lag)

## 3. Technology Choices

### Core Infrastructure
- **PostgreSQL + TimescaleDB**: Battle-tested RDBMS with time-series optimization. Supports complex queries, ACID compliance, and scales to billions of rows with proper partitioning.
  - *Addresses*: FR-003, NFR-006, NFR-008, SR-001, SR-003, OR-001, OR-003
- **Redis**: Sub-millisecond latency for cache hits, supports complex data structures, proven at scale.
  - *Addresses*: NFR-001, NFR-004, SR-001

### AI/ML Stack
- **Anthropic Claude 4**: An of the best-in-class language understanding and SQL generation and to create artefacts as a reports with charts.
  - *Addresses*: FR-001, FR-002, NFR-002, IR-005, OR-004
- **Gemini 2.5 Flash-Lite**: LLM model with very fast results and excellent cost-benefit to answer simple questions.
  - *Addresses*: FR-001, FR-008, NFR-001, IR-005, OR-004
- **LangChain**: Abstraction layer for LLM interactions, simplifies prompt management and chain composition.
  - *Addresses*: IR-005, SR-005, SR-006, SR-007
- **LangGraph**:  AI agent framework designed to build, deploy and manage generative AI agent workflows.
  - *Addresses*: IR-005, SR-005, SR-006, SR-007
- **PGVector**: Managed vector database for semantic search, handles scale without operational overhead.
  - Chunking strategy: The strategy should be decided after evaluating the comments size. I believe it will not be necessary to use techniques as a semantic chunking.
  - Semantic search with cosine similarity query
  - Vectorize & Indexing: HNSW (Hierarchical Navigable Small Worlds)
  - Embedding: OpenAI **text-embedding-3-small** with 1.536 dimensions
  - *Addresses*: NFR-007, OR-005, NFR-008, OR-004
- **BERT / spaCy**: Python frameworks and libraries for sentiment analysis
  - *Addresses*: FR-004
    
### Application Layer
- **FastAPI**: Modern Python framework with async support, automatic API documentation, WebSocket support.
  - *Addresses*: NFR-001, , NFR-003, NFR-004, SR-002, SR-004, IR-002, IR-004, OR-003
- **FastMCP**: Standard framework for building MCP applications.
  - *Addresses*: FR-003, NFR-004, SR-002
- **Celery + Redis**: Proven task queue combination, supports priority queues and result backends.
  - *Addresses*: FR-003, FR-006, NFR-002, IR-003
- **React + TypeScript**: Type-safe frontend development, rich ecosystem, excellent developer experience.
  - *Addresses*: FR-001, FR-002, FR-005, FR-008, FR-003, SR-004, IR-004, OR-004
- **Matplotlib**: Python framework to export visualizations in multiple formats.
  - *Addresses*: FR-007
 
### Data Pipeline
- **Airbyte**: Open-source data integration with Zendesk connector, handles schema evolution gracefully.
  - *Addresses*: IR-001, NFR-005
- **dbt**: SQL-based transformation framework for maintaining data quality and lineage.
  - *Addresses*: NFR-007, SR-005

### Observability
- **Datadog**: Unified monitoring across infrastructure, applications, and logs.
  - *Addresses*: OR-002, NFR-003, SR-008

## 4. Data Schema Structures

### Core Database Schema

```sql
-- Optimized for analytical queries with proper indexing
-- Addresses: NFR-008, NFR-006, NFR-007
CREATE TABLE tickets (
    id BIGSERIAL PRIMARY KEY,
    zendesk_id VARCHAR(50) UNIQUE NOT NULL,
    status VARCHAR(20) NOT NULL,
    priority VARCHAR(20),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    resolved_at TIMESTAMPTZ,
    assignee_id INTEGER,
    requester_id INTEGER,
    tags TEXT[],
    custom_fields JSONB,
    -- Partitioned by month for query optimization
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Performance indexes for NFR-001 compliance
CREATE INDEX idx_tickets_priority_created ON tickets(priority, created_at DESC);
CREATE INDEX idx_tickets_status ON tickets(status) WHERE status != 'closed';
CREATE INDEX idx_tickets_tags ON tickets USING GIN(tags);

CREATE TABLE comments (
    id BIGSERIAL PRIMARY KEY,
    ticket_id BIGINT REFERENCES tickets(id),
    zendesk_id VARCHAR(50) UNIQUE NOT NULL,
    author_id INTEGER,
    content TEXT NOT NULL,
    public BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL,
    -- Derived fields for FR-004 (sentiment analysis)
    sentiment_score FLOAT,
    -- Embedding with 1536 dimensions
    embedding vector(1536), 
    word_count INTEGER GENERATED ALWAYS AS (array_length(string_to_array(content, ' '), 1)) STORED
);

CREATE INDEX idx_comments_ticket_created ON comments(ticket_id, created_at DESC);

-- HNSW index for efficient vector search (recommended for production)
CREATE INDEX idx_comments_embedding_hnsw ON comments
USING hnsw (embedding vector_cosine_ops);

-- Audit table for SR-005 (compliance logging)
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,
    query_text TEXT,
    result_count INTEGER,
    execution_time_ms INTEGER,
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Saved reports table for FR-003
CREATE TABLE saved_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    owner_id VARCHAR(100) NOT NULL,
    query_template TEXT NOT NULL,
    parameters JSONB,
    schedule_cron VARCHAR(100),
    last_run TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### API Contracts

```python
# Request/Response models using Pydantic
# Addresses: FR-001, FR-002, FR-003, SR-002
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class QueryType(Enum):
    SIMPLE_AGGREGATION = "simple_aggregation"
    COMPLEX_ANALYSIS = "complex_analysis"
    VISUALIZATION = "visualization"
    SAVED_REPORT = "saved_report"

class QueryRequest(BaseModel):
    query: str = Field(..., max_length=5000)  # SR-007: Limit input size
    context: Optional[Dict[str, Any]] = {}
    session_id: str
    timestamp: datetime
    user_role: str  # SR-002: RBAC support

class QueryResponse(BaseModel):
    query_id: str
    type: QueryType
    status: str  # "complete", "processing", "error"
    result: Optional[Any]
    visualization: Optional[Dict[str, Any]]  # FR-002
    metadata: Dict[str, Any]  # execution_time, data_freshness, cost
    follow_up_suggestions: List[str]  # FR-005
    data_freshness: datetime  # NFR-005

class SavedReport(BaseModel):
    id: str
    name: str = Field(..., max_length=255)
    query_template: str
    parameters: Dict[str, Any]
    schedule: Optional[str]  # FR-006
    created_by: str
    created_at: datetime
    last_run: Optional[datetime]
    access_level: str  # SR-002: RBAC

class VisualizationRequest(BaseModel):
    data: List[Dict[str, Any]]
    artefact_type: str  # "chart", "html", etc.
    options: Dict[str, Any]
    export_format: Optional[str]  # FR-007: "png", "svg", "html"
```

### Cache Schema

```python
# Redis key patterns addressing NFR-001, NFR-004
CACHE_PATTERNS = {
    "query_result": "query:result:{query_hash}:{user_role}",  # SR-002: Role-based caching
    "user_session": "session:{session_id}",
    "saved_report": "report:{report_id}",
    "metric_snapshot": "metric:{metric_type}:{date}",
    "processing_job": "job:{job_id}",
    "rate_limit": "rate:{user_id}:{window}"  # SR-008: Rate limiting
}

# Example cached query result structure
{
    "query_hash": "abc123...",
    "query": "SELECT COUNT(*) FROM tickets WHERE priority = 'urgent'",
    "result": {"count": 42},
    "executed_at": "2024-01-15T10:30:00Z",
    "ttl": 3600,
    "hit_count": 5,
    "cost_estimate": 0.002,  # OR-003: Cost tracking
    "data_freshness": "2024-01-15T10:25:00Z"  # NFR-005
}
```

## 5. Detailed Technical Explanations

### 1. User Experience Flow

**Challenge**: Creating a seamless experience from natural language input to actionable insight requires orchestrating multiple complex systems while maintaining sub-5-second response times.

**Technical Approach**:

When Brenda types "How many urgent tickets did we get last week?", the system initiates a multi-stage pipeline optimized for speed and reliability:
- **WebSocket Connection**: We maintain persistent WebSocket connections to eliminate TCP handshake overhead (saves ~100-200ms per request). The connection includes automatic reconnection logic with exponential backoff to handle network interruptions gracefully.
- **Query Classification Layer**: Using a lightweight classifier (Gemini 2.5 Flash-Lite), we categorize queries in <500ms into simple aggregations, complex analyses, or visualization requests. This early classification enables optimal routing - simple queries bypass heavy processing entirely.
- **Intelligent Caching**: We implement a multi-tier cache strategy using Redis. Query results are cached with intelligent TTLs based on data volatility (real-time metrics: 1 hour, historical analyses: 24 hours). We use query fingerprinting (normalized SQL hash) to identify semantically identical queries despite syntactic differences.
- **Simple Query Response**: The AI agent identifies simple queries, generates SQL through the LLM, and executes optimized queries against indexed columns in PostgreSQL.

**Key Technical Challenges Addressed**:

- **Latency Budget Management**: We allocate specific time budgets to each component (LLM: 2s, SQL execution: 2s, network: 500ms, rendering: 500ms) with circuit breakers if any component exceeds its budget.
- **Error Recovery**: Implement graceful degradation - if the LLM fails, we fall back to keyword-based query matching against common patterns.

**Requirements Addressed**: 
- FR-001 (natural language queries)
- FR-005 (follow-up questions with context)
- NFR-001 (< 5 seconds response time)
- NFR-002 (< 45 seconds for complex queries with progress updates)
- NFR-003 (99.9% availability through fallback mechanisms)
- SR-004 (TLS encryption for WebSocket)
- OR-002 (monitoring with circuit breakers)

### 2. Historical Data Challenges

**Challenge**: Analyzing millions of tickets spanning years while maintaining interactive response times requires sophisticated data architecture and query optimization.

**Technical Approach**:

- **Hybrid Storage Architecture**: We implement a three-tier storage strategy:
  - **Hot Data** (last 90 days): Stored in PostgreSQL with all indexes, optimized for sub-second queries
  - **Warm Data** (90 days - 1 year): PostgreSQL with TimescaleDB compression, 2-5 second query times
  - **Cold Data** (>1 year): PostgreSQL with aggressive TimescaleDB compression and archived partitions, accessed via background jobs

- **Intelligent Partitioning**: Tables are partitioned by month with automatic partition management. This enables partition pruning, reducing query scope by 90%+ for time-bounded queries.

- **Materialized Views & Aggregations**: We pre-compute common aggregations (daily ticket counts by priority/status) in materialized views, refreshed every 30 minutes.

- **Adaptive Query Planning**: The system analyzes query patterns and automatically creates covering indexes for frequently accessed column combinations. We use pg_stat_statements to identify slow queries and optimize them proactively.

**Performance Guarantees**:
- Simple aggregations: <1 second (using materialized views)
- Time-bounded queries: <5 seconds (partition pruning)
- Full historical scans: Background processing with progress updates

**Requirements Addressed**:
- NFR-001 (< 5 seconds for simple queries)
- NFR-002 (< 45 seconds for complex analyses)
- NFR-005 (< 30 minutes data freshness via continuous aggregates)
- NFR-006 (3 years retention with compression)
- NFR-008 (5M tickets, 100M comments capacity)
- OR-001 (automated backups with TimescaleDB policies)
- OR-003 (cost tracking via compression ratios)

### 3. Business Data Risks

**Challenge**: The system has access to sensitive customer communications and operational metrics. Careless implementation could lead to data breaches, compliance violations, or business intelligence leaks.

**Technical Safeguards**:

- **Read-Only Access Enforcement**: The application connects to PostgreSQL using a read-only role with explicit REVOKE on all write operations. Even if SQL injection occurs, no data modification is possible.

- **Query Validation Pipeline**: 
  ```python
  def validate_query(sql: str) -> bool:
      # Whitelist allowed operations
      if not sql.strip().upper().startswith(('SELECT', 'WITH')):
          return False
      
      # Blacklist dangerous patterns
      dangerous_patterns = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'GRANT']
      if any(pattern in sql.upper() for pattern in dangerous_patterns):
          return False
      
      # Validate against SQL parser
      try:
          parsed = sqlparse.parse(sql)[0]
          # Additional AST validation
      except:
          return False
      
      return True
  ```

- **PII Protection Layer**: We implement automatic PII detection using spaCy's named entity recognition. Detected PII is either masked or excluded based on user permissions:
  ```python
  # Example: "Customer John Smith (john@example.com) complained..."
  # Becomes: "Customer [REDACTED] ([REDACTED]) complained..."
  ```

- **Audit Trail**: Every query is logged with user ID, timestamp, query text, result count, and execution time in an immutable audit table. This provides forensic capability and compliance evidence.

- **Data Access Governance**:
  - Role-based access control with JWT tokens containing permission claims
  - Row-level security for multi-tenant scenarios
  - Encryption at rest (AES-256) and in transit (TLS 1.3)

**Risk Mitigation Matrix**:
- Unauthorized access → OAuth/SAML SSO + MFA
- Data exfiltration → Result size limits + rate limiting
- Prompt injection → Input sanitization + query validation
- Compliance violations → Automated GDPR compliance checks

**Requirements Addressed**:
- SR-001 (read-only access enforcement)
- SR-002 (RBAC implementation)
- SR-003 (AES-256 encryption at rest)
- SR-004 (TLS 1.3 encryption in transit)
- SR-005 (immutable audit trail)
- SR-006 (PII masking)
- SR-007 (prompt injection prevention)
- SR-008 (rate limiting)
- CR-001 (GDPR compliance)
- CR-002 (data residency)

### 4. Complex Business Questions

**Challenge**: Questions like "Why are customers unhappy with our new feature?" require understanding sentiment, identifying patterns, and correlating multiple data points - capabilities not present in raw ticket data.

**Technical Approach**:

- **Semantic Layer Construction**: We build a semantic understanding layer on top of raw data:
  ```python
  # Pipeline
  ticket_data → sentiment_analysis → entity_extraction → topic_modeling → insight_generation
  ```

- **Multi-Model Sentiment Analysis**:
  - Real-time: BERT-based sentiment classifier
    - Results stored as `sentiment_score` (-1 to 1)

- **Vector Embeddings for Semantic Search**: 
  - Comments are embedded using OpenAI's text-embedding-3-small (1536 dimensions)
  - Stored in pgvector with HNSW indexing for efficient similarity search
  
- **Dynamic Feature Extraction**: Using LangChain, we create dynamic analysis chains:
  ```python
  chain = (
      semantic_search
      | sentiment_filter
      | theme_extraction
      | root_cause_analysis
      | summary_generation
  )
  ```

- **Contextual Understanding**: The AI agent maintains conversation context, understanding that "the new feature" refers to "auto-sync" based on previous interactions or temporal correlation with feature release dates.

**Example Processing Flow**:
1. Query: "Why are customers unhappy with auto-sync?"
2. Semantic search finds tickets mentioning "auto-sync" or related terms
3. Sentiment analysis filters for negative sentiment (score < -0.3)
4. Theme extraction identifies common complaints (data loss, sync delays, confusion)
5. Root cause analysis via LLM: "Customers are primarily frustrated with data loss issues (73% of complaints) and slow synchronization (45%)"
6. System generates actionable insight with supporting data

**Requirements Addressed**:
- FR-001 (natural language queries)
- FR-004 (sentiment analysis)
- NFR-002 (< 45 seconds for complex analysis)
- NFR-007 (99.5% accuracy through multi-model validation)
- NFR-008 (handle 100M comments via vector indexing)
- IR-005 (multiple LLM providers)

### 5. Workflow Efficiency

**Challenge**: Repetitive report generation wastes valuable time and introduces human error. The system must enable report templates while maintaining flexibility.

**Solution Architecture**:

- **Report Template System**: Report template as a flexible, reusable framework for defining, managing, and executing database reports with dynamic parameters.

- **Intelligent Parameterization**: The system automatically identifies variable components in queries:
  - Dates: "last week" → `{start_date}` to `{end_date}`
  - Filters: "urgent tickets" → `{priority_filter}`
  - Groupings: "by day" → `{time_bucket}`

- **Natural Language Scheduling**: Users can say "Run this every Monday at 9 AM" which translates to expression valid

- **Version Control for Reports**: Templates are versioned with Git-like semantics, enabling rollback and change tracking

- **Smart Suggestions**: Based on query history, the system proactively suggests:
  - "You run similar reports every Monday. Would you like to save this as 'Weekly Priority Report'?"
  - "This report is similar to 'Daily Metrics'. Would you like to use that template?"

**Implementation Details**:
- Templates stored in PostgreSQL with JSONB for flexible schema
- Celery Beat for scheduled execution
- Results delivered via email/Slack with configurable formats
- Automatic report optimization based on execution patterns

**Requirements Addressed**:
- FR-003 (save and re-run reports)
- FR-006 (scheduled report generation)
- FR-008 (query suggestions)
- NFR-001 (< 5 seconds via caching)
- SR-002 (RBAC for report access)
- OR-003 (cost tracking per report execution)

### 6. Data Visualization Needs

**Challenge**: Generating dynamic, interactive visualizations from natural language requests requires understanding visualization intent and handling various data shapes.

**Technical Solution**:

- **Visualization Intent Detection**: Using Claude 4's capability to generate structured outputs:
  ```python
  # LLM interprets: "show me a bar chart of tickets by priority"
  visualization_spec = {
      "type": "bar",
      "x_axis": "priority",
      "y_axis": "count",
      "title": "Tickets by Priority",
      "color_scheme": "categorical"
  }
  ```

- **Multi-Format Rendering Pipeline**:
  - Server-side: Matplotlib for static images (PNG/SVG)
  - Client-side: D3.js/Recharts for interactive charts
  - Hybrid: Server generates data structure, client renders

- **Adaptive Visualization Selection**: The system automatically selects appropriate visualization types based on data characteristics:
  - Time series → Line chart
  - Categorical comparison → Bar chart
  - Part-of-whole → Pie chart
  - Correlation → Scatter plot

- **Chart Artifact Generation**: Using Claude 4's artifact capability to generate complete HTML/JS visualization code that can be embedded or exported

**Technical Challenges Addressed**:
- **Large Dataset Handling**: Automatic data sampling/aggregation for datasets >10,000 points
- **Responsive Design**: Charts automatically adapt to container size
- **Accessibility**: All charts include ARIA labels and keyboard navigation
- **Export Options**: PNG (reports), SVG (presentations), HTML (interactive dashboards)

### 7. Operational Cost Management

**Challenge**: Query costs vary by 1000x between simple lookups and complex AI analyses. Without careful management, costs could spiral out of control.

**Cost Optimization Strategy**:

- **Tiered Processing Model**:
  ```python
  COST_TIERS = {
      "cache_hit": 0.0001,      # Redis lookup only
      "simple_sql": 0.001,       # Direct SQL query
      "complex_sql": 0.01,       # Multi-table joins
      "llm_simple": 0.02,        # Gemini Flash
      "llm_complex": 0.10,       # Claude 4
      "embedding_search": 0.05,   # Vector similarity
      "batch_analysis": 0.50     # Full historical scan
  }
  ```

- **Intelligent Router**: Queries are routed to the most cost-effective processor:
  - Exact match previous queries → Cache (free)
  - Simple aggregations → SQL only (no LLM)
  - Known patterns → Template execution
  - Complex questions → Full LLM pipeline

- **Cost Budget Management**:
  - Per-user daily budgets with soft/hard limits
  - Department-level cost allocation
  - Real-time cost tracking with warnings at 80% budget

- **Optimization Techniques**:
  - Query result caching (reduces cost by 60%)
  - Batch processing for non-urgent queries
  - Incremental computation for time-series data
  - LLM prompt optimization (shorter prompts = lower cost)

**Cost Monitoring Dashboard**:
- Real-time cost per query
- User/department cost attribution
- Cost trend analysis
- ROI metrics (time saved vs. cost incurred)

**Requirements Addressed**:
- OR-003 (cost tracking per query/user)
- NFR-001 (< 5 seconds through caching)
- NFR-002 (< 45 seconds through batch processing)
- NFR-004 (200 concurrent users with budget limits)
- IR-005 (multi-LLM with cost optimization)

### 8. Technology Evolution

**Challenge**: The AI landscape changes rapidly - models deprecate, new capabilities emerge, and providers experience outages. The architecture must remain stable despite this volatility.

**Future-Proofing Strategy**:

- **Provider Abstraction Layer**: Software design pattern that creates a unified interface between an application and multiple underlying service providers or implementations. It acts as an intermediary that abstracts away the differences between various providers.

- **Multi-Provider Fallback Chain**:
  - Primary: Claude 4 (best quality)
  - Secondary: GPT-4 (fallback)
  - Tertiary: Gemini (cost-optimized)
  - Emergency: Cached responses + template matching

- **Version-Agnostic Interfaces**: All AI interactions go through standardized interfaces that hide provider-specific details

- **Capability Detection**: System automatically detects and adapts to available model capabilities:

- **Gradual Migration Path**: New models are tested in shadow mode (parallel execution without user impact) before promotion

- **Model Performance Tracking**: Continuous monitoring of model performance enables data-driven provider selection:
  - Response time percentiles
  - Accuracy metrics
  - Cost per query
  - Error rates

**Adaptation Mechanisms**:
- Feature flags for instant provider switching
- A/B testing framework for new models
- Automated fallback on provider outage
- Regular model retraining

**Requirements Addressed**:
- IR-005 (multiple LLM providers)
- NFR-003 (99.9% availability through fallbacks)
- OR-002 (monitoring and alerting)
- OR-004 (multi-region support)
- NFR-001 & NFR-002 (performance through optimal model selection)
 
This architecture ensures the system remains functional and performant regardless of changes in the AI ecosystem, while enabling rapid adoption of new capabilities as they become available.

## Risk Assessment & Mitigation

| Risk | Impact | Probability | Mitigation Strategy | Requirements Affected |
|---|---|---|---|---|
| LLM provider outage | High | Medium | Multi-provider fallback, cached responses | FR-001, NFR-003 |
| Data synchronization lag | Medium | Low | Real-time CDC pipeline, monitoring alerts | NFR-005, IR-001 |
| Query performance degradation | High | Medium | Query optimization, caching, resource scaling | NFR-001, NFR-002 |
| Security breach | Critical | Low | Defense in depth, regular audits | SR-001 through SR-008 |
| Cost overrun | Medium | Medium | Usage monitoring, tiered processing | OR-003 |
| Compliance violation | High | Low | Automated compliance checks, audit trails | CR-001 through CR-002 |

## Success Metrics

| Metric | Target | Measurement Method | Related Requirements |
|---|---|---|---|
| Query response time (P95) | < 5 seconds | Application metrics | NFR-001 |
| System availability | 99.9% | Uptime monitoring | NFR-003 |
| User adoption rate | 80% of support team | Usage analytics | FR-001, FR-005 |
| Query accuracy | 99.5% | Manual validation sampling | NFR-007 |
| Cost per query | < $0.10 average | Cost tracking system | OR-003 |
| Data freshness | < 30 minutes | Sync monitoring | NFR-005 |
| Security incidents | 0 critical/quarter | Security monitoring | SR-001 through SR-008 |

## Conclusion

This architecture provides a robust foundation for the SupportWise AI Co-pilot, balancing immediate utility with long-term scalability. The modular design enables rapid iteration while maintaining production stability, and the multi-layered approach to data processing ensures both performance and cost-effectiveness.

The comprehensive requirements matrix ensures all stakeholder needs are addressed, from functional capabilities to security and compliance. Through careful abstraction and forward-thinking design patterns, we've created a platform that can adapt to the evolving AI landscape while delivering immediate value to our users.

The phased implementation approach allows for early value delivery while systematically addressing more complex requirements, ensuring a sustainable path from MVP to enterprise-grade solution.
