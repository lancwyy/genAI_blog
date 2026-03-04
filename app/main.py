"""
main.py — FastAPI application entry point.

Defines routes and wires together the agent, database, and templates.
"""

from __future__ import annotations

import logging
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.agent import AIResearchAgent
from app.database import get_articles, init_db

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan — initialise DB on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown hook."""
    init_db()
    logger.info("Database initialised.")
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Technology News Agent",
    description="Search and summarize the latest AI news and research.",
    version="0.1.0",
    lifespan=lifespan,
)

# Static files & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """Render the main page with any previously stored articles."""
    articles = get_articles(limit=50)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "articles": articles},
    )


@app.post("/fetch-news", response_class=HTMLResponse)
async def fetch_news(request: Request):
    """Trigger automatic search using default AI queries."""
    agent = AIResearchAgent()
    articles = await agent.run(query=None)
    return templates.TemplateResponse(
        "partials/articles.html",
        {"request": request, "articles": articles},
    )


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    """Trigger a keyword / description-based search."""
    agent = AIResearchAgent()
    articles = await agent.run(query=query)
    return templates.TemplateResponse(
        "partials/articles.html",
        {"request": request, "articles": articles},
    )


@app.get("/articles", response_class=JSONResponse)
async def list_articles():
    """Return stored articles as JSON."""
    articles = get_articles(limit=100)
    return [a.model_dump(mode="json") for a in articles]


@app.post("/generate-blog", response_class=HTMLResponse)
async def generate_blog(request: Request, article_ids: List[int] = Form(...), llm_model: str = Form("gemini")):
    """Generate a blog article from selected research articles."""
    agent = AIResearchAgent()
    file_path = await agent.generate_blog_article(article_ids, model_provider=llm_model)
    
    if "Error" in file_path:
        return f"<div class='alert alert--danger'>{file_path}</div>"
    
    return f"<div class='alert alert--success'>文章已生成：{file_path}</div>"
