from bs4 import BeautifulSoup


def extract_text_chunks(html: str) -> list[str]:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    return [chunk.strip() for chunk in text.split("。") if chunk.strip()]
