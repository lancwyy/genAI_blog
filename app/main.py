"""
main.py — FastAPI application entry point.

Defines routes and wires together the agent, database, and templates.
"""

from __future__ import annotations

import logging
import os
from typing import List, Dict
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.agent import AIResearchAgent
from app.database import (
    get_articles, 
    init_db, 
    get_unique_article_dates, 
    get_articles_by_date, 
    delete_article, 
    get_setting, 
    set_setting,
    get_fetch_queries,
    add_fetch_query,
    delete_fetch_query
)

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
async def fetch_news(request: Request, start_date: str = Form(None), end_date: str = Form(None)):
    """Trigger automatic search using default AI queries with date range filtering."""
    agent = AIResearchAgent()
    articles = await agent.run(query=None, start_date=start_date, end_date=end_date)
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
    result = await agent.generate_blog_article(article_ids, model_provider=llm_model)
    
    if result["status"] == "error":
        error_html = f"""
        <div class='alert alert--danger'>
            <strong>無法生成文章</strong><br><br>
            <strong>提示詞如下：</strong><br>
            <pre style='white-space: pre-wrap; background: #f8d7da; padding: 10px; border-radius: 4px;'>{result['prompt']}</pre><br>
            <strong>錯誤訊息如下：</strong><br>
            <code>{result['message']}</code>
        </div>
        """
        return error_html
    
    file_path = result["file_path"]
    return f"<div class='alert alert--success'>文章已生成：{file_path}</div>"


# ---------------------------------------------------------------------------
# Admin Routes
# ---------------------------------------------------------------------------

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """Render the management dashboard."""
    settings = {
        "GOOGLE_API_KEY": get_setting("GOOGLE_API_KEY"),
        "OPENAI_API_KEY": get_setting("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": get_setting("ANTHROPIC_API_KEY"),
        "GROQ_API_KEY": get_setting("GROQ_API_KEY"),
    }
    dates = get_unique_article_dates()
    
    # Blog file listing
    blog_base = Path("/app/blog_article")
    blog_structure = {}
    if blog_base.exists():
        # List directories (dates)
        for date_dir in sorted(blog_base.iterdir(), reverse=True):
            if date_dir.is_dir():
                files = [f.name for f in date_dir.glob("*") if f.is_file()]
                if files:
                    blog_structure[date_dir.name] = sorted(files, reverse=True)

    # Fetch managed keywords
    queries = get_fetch_queries()

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request, 
            "settings": settings, 
            "dates": dates,
            "blogs": blog_structure,
            "queries": queries
        },
    )


@app.post("/admin/settings", response_class=HTMLResponse)
async def update_settings(
    request: Request,
    google_key: str = Form(""),
    openai_key: str = Form(""),
    anthropic_key: str = Form(""),
    groq_key: str = Form(""),
):
    """Update LLM API keys in the database."""
    set_setting("GOOGLE_API_KEY", google_key)
    set_setting("OPENAI_API_KEY", openai_key)
    set_setting("ANTHROPIC_API_KEY", anthropic_key)
    set_setting("GROQ_API_KEY", groq_key)
    return "<div class='alert alert--success'>設定已更新成功！</div>"


@app.get("/admin/articles/date/{date_str}", response_class=HTMLResponse)
async def admin_articles_by_date(request: Request, date_str: str):
    """Fetch articles for a specific date for management."""
    articles = get_articles_by_date(date_str)
    # Simple partial for admin search history with bulk selection
    html = f"""
    <div style='margin-bottom: 1rem; display: flex; gap: 1rem; align-items: center;'>
        <label><input type='checkbox' onclick='toggleAll("{date_str}", this.checked)'> 全選</label>
        <button class='btn btn--danger btn--small' 
                hx-post='/admin/articles/bulk-delete' 
                hx-include='.chk-{date_str}'
                hx-vals='{{"date_str": "{date_str}"}}'
                hx-target='#articles-{date_str}'
                hx-confirm='確定要刪除選中的紀錄嗎？'>刪除選中項</button>
    </div>
    <table class='result-list'>
    """
    for a in articles:
        html += f"""
        <tr id='article-{a.id}'>
            <td style='width: 40px;'><input type='checkbox' name='article_ids' value='{a.id}' class='chk-{date_str}'></td>
            <td>{a.title}</td>
            <td>{a.source}</td>
            <td>
                <button class='btn btn--danger btn--small' 
                        hx-delete='/admin/articles/{a.id}' 
                        hx-target='#article-{a.id}' 
                        hx-swap='outerHTML'
                        hx-confirm='確定要刪除此紀錄嗎？'>刪除</button>
            </td>
        </tr>
        """
    html += "</table>"
    return html


@app.post("/admin/articles/bulk-delete", response_class=HTMLResponse)
async def admin_bulk_delete_articles(request: Request, date_str: str = Form(...), article_ids: List[int] = Form([])):
    """Bulk delete articles and re-render the list for the given date."""
    from app.database import delete_articles_bulk
    delete_articles_bulk(article_ids)
    return await admin_articles_by_date(request, date_str)


@app.delete("/admin/articles/{article_id}", response_class=HTMLResponse)
async def admin_delete_article(article_id: int):
    """Delete an article from history."""
    delete_article(article_id)
    return ""


@app.post("/admin/blogs/bulk-delete", response_class=HTMLResponse)
async def admin_bulk_delete_blogs(request: Request, date_str: str = Form(...), file_ptrs: List[str] = Form([])):
    """Bulk delete blog articles/logs and re-render the list."""
    for ptr in file_ptrs:
        if ".." in ptr: continue
        file_path = Path("/app/blog_article") / ptr
        if file_path.exists():
            file_path.unlink()
    
    # Re-render file list for this date
    blog_base = Path("/app/blog_article") / date_str
    files = []
    if blog_base.exists():
        files = sorted([f.name for f in blog_base.glob("*") if f.is_file()], reverse=True)
    
    if not files:
        if blog_base.exists() and not any(blog_base.iterdir()):
            try:
                blog_base.rmdir()
            except OSError:
                pass
        return "<p style='color: var(--text-muted); padding: 1rem;'>此日期的檔案已全數刪除。</p>"

    html = f"""
    <div style="margin-bottom: 1rem; display: flex; gap: 1rem; align-items: center;">
        <label><input type="checkbox" onclick="toggleAll('blog-{date_str}', this.checked)"> 全選</label>
        <button class="btn btn--danger btn--small" 
                hx-post="/admin/blogs/bulk-delete" 
                hx-include=".chk-blog-{date_str}"
                hx-vals='{{"date_str": "{date_str}"}}'
                hx-target="#articles-blog-{date_str}"
                hx-confirm="確定要刪除選中的檔案嗎？">刪除選中項</button>
    </div>
    <ul class="item-list">
    """
    for i, file in enumerate(files):
        html += f"""
        <li class="item-row" id="file-{date_str}-{i}">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <input type="checkbox" name="file_ptrs" value="{date_str}/{file}" class="chk-blog-{date_str}">
                <span style="font-family: monospace; font-size: 0.9rem;">{file}</span>
            </div>
            <button class="btn btn--danger btn--small"
                hx-delete="/admin/blogs/{date_str}/{file}"
                hx-target="#file-{date_str}-{i}" hx-swap="outerHTML"
                hx-confirm="確定要刪除此檔案嗎？">刪除</button>
        </li>
        """
    html += "</ul>"
    return html


@app.delete("/admin/blogs/{date_str}/{filename}", response_class=HTMLResponse)
async def admin_delete_blog(date_str: str, filename: str):
    """Delete a blog article or error file."""
    # Prevent directory traversal
    if ".." in date_str or ".." in filename:
        return HTMLResponse("Invalid path", status_code=400)
        
    file_path = Path("/app/blog_article") / date_str / filename
    if file_path.exists():
        file_path.unlink()
        
        # Check if dir is now empty
        dir_path = file_path.parent
        if dir_path.exists() and not any(dir_path.iterdir()):
            try:
                dir_path.rmdir()
            except OSError:
                pass
                
        return ""  # HTMX swaps out the list item
    return HTMLResponse("File not found", status_code=404)


# ---------------------------------------------------------------------------
# admin keyword routes
# ---------------------------------------------------------------------------

@app.post("/admin/queries", response_class=HTMLResponse)
async def admin_add_query(request: Request, query: str = Form(...)):
    """Add a new fetch keyword and return the updated row."""
    if not query.strip():
        return ""
    new_id = add_fetch_query(query.strip())
    
    # Return a single row to be appended or re-rendered
    return f"""
    <li class='item-row' id='query-{new_id}'>
        <span style='font-weight: 500;'>{query.strip()}</span>
        <button class='btn btn--danger btn--small' 
                hx-delete='/admin/queries/{new_id}' 
                hx-target='#query-{new_id}' 
                hx-swap='outerHTML'
                hx-confirm='確定要刪除此關鍵字嗎？'>刪除</button>
    </li>
    """


@app.delete("/admin/queries/{query_id}", response_class=HTMLResponse)
async def admin_delete_query(query_id: int):
    """Delete a fetch keyword."""
    delete_fetch_query(query_id)
    return ""
