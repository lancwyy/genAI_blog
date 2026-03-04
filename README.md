# AI Technology News Agent

An AI-powered web application that searches, summarizes, and generates professional blog articles from the latest AI-related technical news and research papers.

## Features

- **Automatic Search** — One-click fetch of the latest AI news using pre-configured queries (*AI*, *large language model*, *AI agent*)
- **Keyword Search** — Enter any keywords or description to find targeted content
- **Multi-Source** — Aggregates results from arXiv (with rate limiting), Hacker News, and AI blog RSS feeds
- **Local Summarization** — Fast, cost-effective summaries using **TextRank** (no LLM calls during search)
- **Selectable Blog Generation** — Research and synthesize a professional blog post from selected articles using your choice of **Gemini**, **OpenAI (GPT-4o)**, or **Claude (3.5 Sonnet)**
- **Persistent Storage** — All results are stored in SQLite for later retrieval
- **DB Admin UI** — Built-in `sqlite-web` interface for database management

## Tech Stack

| Layer     | Technology                              |
|-----------|-----------------------------------------|
| Backend   | Python 3.11, FastAPI                    |
| Frontend  | HTML + HTMX (minimal JS)               |
| AI (Gen)  | Gemini, GPT-4o, Claude 3.5              |
| AI (Sum)  | TextRank (Local via `sumy`)             |
| Sources   | arXiv API, HN Algolia, RSS             |
| Storage   | SQLite                                  |
| Infra     | Docker, Docker Compose                  |

## Quick Start

### 1. Clone the repository

```bash
git clone <repo-url>
cd <project-dir>
```

### 2. Set your API keys

Create a `.env` file in the project root with at least one provider key:

```
# Required for Gemini
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: Required for OpenAI/Claude features
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

> **Note**: Summaries always use local TextRank to save cost. LLM keys are only used when you click the "Generate Article" button.

### 3. Start the application

```bash
docker compose up --build
```

The app will be available at **http://localhost:8000**.
The Database Admin UI will be available at **http://localhost:8080**.

## Usage

### Article Discovery

1. Click **Fetch Latest AI News** or use the **Search** box.
2. The agent aggregates and summarizes the latest content.

### Blog Generation

1. Browse the results in the list view.
2. **Tick the checkboxes** next to the articles you want to research.
3. Click the **🚀 研究並生成文章** button in the bottom action bar.
4. The generated Markdown article will be saved to `blog_article/YYYY-MM-DD/`.

## API Endpoints

| Method | Path              | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/`               | Main webpage                         |
| POST   | `/fetch-news`     | Trigger automatic search             |
| POST   | `/search`         | Keyword search (form field: `query`) |
| POST   | `/generate-blog`  | Generate blog from selected IDs      |
| GET    | `/articles`       | Return stored articles as JSON       |

## Project Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── app/
│   ├── main.py             # FastAPI routes
│   ├── agent.py            # AIResearchAgent (Gemini integration)
│   ├── search_sources.py   # arXiv / HN / RSS integrations
│   ├── database.py         # SQLite helpers
│   └── models.py           # Pydantic models
├── templates/
│   ├── index.html          # Main page
│   └── partials/
│       └── articles.html   # List view partial
├── static/
│   └── style.css           # Premium dark-mode styling
└── blog_article/           # Generated articles (auto-created)
```

## License

MIT
