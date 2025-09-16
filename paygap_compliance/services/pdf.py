from __future__ import annotations

try:  # pragma: no cover - import guard exercised when dependency missing
    from playwright.async_api import async_playwright
except ModuleNotFoundError:  # pragma: no cover - executed during optional installs
    async_playwright = None  # type: ignore[assignment]


async def html_to_pdf(html: str) -> bytes:
    """Render HTML to PDF bytes using Playwright and headless Chromium."""
    if async_playwright is None:
        raise RuntimeError("Playwright is not installed. Run `playwright install chromium`.")

    async with async_playwright() as playwright:  # type: ignore[misc]
        browser = await playwright.chromium.launch()
        page = await browser.new_page()
        try:
            await page.set_content(html, wait_until="networkidle")
            pdf_bytes = await page.pdf(format="A4")
        finally:
            await page.close()
            await browser.close()
    return pdf_bytes
