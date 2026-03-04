"""
agent.py — AIResearchAgent: the core agent orchestrator.

Responsible for:
  1. Searching external sources (arXiv, Hacker News, RSS feeds)
  2. Summarizing articles via TextRank (No LLM for search results)
  3. Persisting results to SQLite
  4. Generating blog articles from selected sources via Google Gemini (Sequential)
"""

from __future__ import annotations

import logging
import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import google.generativeai as genai
import nltk
from openai import OpenAI
from anthropic import Anthropic
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

from app.database import save_article, get_articles_by_ids
from app.models import Article, RawArticle
from app.search_sources import search_all_sources

logger = logging.getLogger(__name__)

# Default queries used for the "automatic search" feature.
DEFAULT_QUERIES = ["AI", "large language model", "AI agent"]

# Ensure NLTK data is available for sumy
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class AIResearchAgent:
    """
    Core research agent.
    - Initial summaries: Local TextRank (No LLM usage for speed & cost)
    - Blog generation: Selectable LLM (Gemini, OpenAI, Claude)
    """

    def __init__(self) -> None:
        # Gemini setup
        gemini_key = os.getenv("GOOGLE_API_KEY", "")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self._gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        else:
            self._gemini_model = None

        # OpenAI setup
        openai_key = os.getenv("OPENAI_API_KEY", "")
        self._openai_client = OpenAI(api_key=openai_key) if openai_key else None

        # Anthropic setup
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._anthropic_client = Anthropic(api_key=anthropic_key) if anthropic_key else None
        
        # Simple lock to ensure sequential LLM requests if multiple generation calls occur
        self._llm_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, query: Optional[str] = None) -> List[Article]:
        """Main orchestration: search → summarize (Local TextRank) → save → return."""
        raw_articles = await self.search_sources(query)

        processed: List[Article] = []
        for raw in raw_articles:
            # Non-LLM summarization to save tokens and time during search
            summary = self.summarize_article(raw)
            article = Article(
                title=raw.title,
                summary=summary,
                url=raw.url,
                source=raw.source,
            )
            saved = self.save_article(article)
            processed.append(saved)

        logger.info(
            "Agent run complete — %d articles processed (query=%s)",
            len(processed),
            query,
        )
        return processed

    async def generate_blog_article(self, article_ids: List[int], model_provider: str = "gemini") -> str:
        """
        Research selected articles and generate a blog post.
        Uses the selected LLM provider (gemini, openai, claude) sequentially.
        """
        articles = get_articles_by_ids(article_ids)
        if not articles:
            return "Error: No articles found for the given IDs."

        # Prepare context
        context = "\n\n".join([
            f"Title: {a.title}\nSource: {a.source}\nSummary: {a.summary}\nURL: {a.url}"
            for a in articles
        ])

        prompt = (
            "You are a professional tech blogger. Research the following articles "
            "and write a high-quality, engaging blog post that synthesizes their "
            "key points. Quote from the articles where appropriate and provide "
            "insights into the technical trends they represent.\n\n"
            "Articles to reference:\n"
            f"{context}\n\n"
            "Requirements:\n"
            "1. Use a professional and engaging tone.\n"
            "2. Include a compelling title.\n"
            "3. Structure with headers (Markdown).\n"
            "4. Cite the sources using the provided URLs.\n"
            "5. Write in Traditional Chinese (繁體中文)."
        )

        async with self._llm_lock:
            try:
                content = ""
                if model_provider == "gemini":
                    if not self._gemini_model:
                        return "Error: GOOGLE_API_KEY not set."
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(None, lambda: self._gemini_model.generate_content(prompt))
                    content = response.text
                
                elif model_provider == "openai":
                    if not self._openai_client:
                        return "Error: OPENAI_API_KEY not set."
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(None, lambda: self._openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}]
                    ))
                    content = response.choices[0].message.content
                
                elif model_provider == "claude":
                    if not self._anthropic_client:
                        return "Error: ANTHROPIC_API_KEY not set."
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(None, lambda: self._anthropic_client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=4096,
                        messages=[{"role": "user", "content": prompt}]
                    ))
                    content = response.content[0].text
                
                else:
                    return f"Error: Unknown model provider '{model_provider}'"

                # Directory logic
                today_str = datetime.now().strftime("%Y-%m-%d")
                base_dir = Path("/app/blog_article") / today_str
                base_dir.mkdir(parents=True, exist_ok=True)

                timestamp = int(datetime.now().timestamp())
                filename = f"blog_{model_provider}_{timestamp}.md"
                prompt_filename = f"blog_{model_provider}_{timestamp}_prompt.txt"
                
                file_path = base_dir / filename
                prompt_path = base_dir / prompt_filename
                
                # Save the generated content
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                # Save the input prompt for audit
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(prompt)

                logger.info("Generated blog article saved to %s (and prompt to %s)", file_path, prompt_path)
                return str(file_path)

            except Exception as exc:
                logger.error("Failed to generate blog article: %s", exc)
                return f"Error during generation: {str(exc)}"

    # ------------------------------------------------------------------
    # Step 1: Search
    # ------------------------------------------------------------------

    async def search_sources(
        self, query: Optional[str] = None
    ) -> List[RawArticle]:
        """
        Retrieve articles from all configured sources.
        """
        queries = [query] if query else DEFAULT_QUERIES
        all_articles: List[RawArticle] = []
        seen_urls: set[str] = set()

        for q in queries:
            results = await search_all_sources(q, max_per_source=5)
            for article in results:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    all_articles.append(article)

        return all_articles

    # ------------------------------------------------------------------
    # Step 2: Summarize (Local TextRank)
    # ------------------------------------------------------------------

    def summarize_article(self, article: RawArticle) -> str:
        """
        Generate a concise summary without using any LLM.
        Priority: 
        1. Existing content/description (if long enough)
        2. TextRank extraction via sumy
        """
        # If the article already provides a decent summary or description
        content_len = len(article.content.strip())
        if 100 <= content_len <= 800:
            return article.content.strip()

        # If content is too long or non-existent, use TextRank
        if content_len > 100:
            return self._textrank_summarize(article.content, sentence_count=3)
        
        return article.content.strip() or "No summary available."

    @staticmethod
    def _textrank_summarize(text: str, sentence_count: int = 3) -> str:
        """Perform extractive summarization using TextRank algorithm."""
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary_sentences = summarizer(parser.document, sentence_count)
            return " ".join([str(s) for s in summary_sentences])
        except Exception as exc:
            logger.warning("TextRank summarization failed: %s", exc)
            # Simple fallback
            return text[:300].rsplit(" ", 1)[0] + "..." if len(text) > 300 else text

    # ------------------------------------------------------------------
    # Step 3: Save
    # ------------------------------------------------------------------

    @staticmethod
    def save_article(article: Article) -> Article:
        """Persist an article to SQLite."""
        return save_article(article)
