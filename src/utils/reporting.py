"""Helpers for markdown and PDF reporting artifacts."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .corpus import ensure_parent_dir


def write_text(path: Path, content: str) -> None:
    """Write plain text content with UTF-8 encoding."""

    ensure_parent_dir(path)
    path.write_text(content, encoding="utf-8")


def _wrap_paragraph(text: str, width: int = 100) -> str:
    """Wrap long text for PDF pages that use plain matplotlib text blocks."""

    return textwrap.fill(text, width=width)


def render_markdown_to_pdf(
    *,
    title: str,
    markdown_content: str,
    figure_paths: list[Path],
    output_path: Path,
) -> None:
    """Render a lightweight PDF report directly from markdown-like content and figures."""

    ensure_parent_dir(output_path)

    paragraphs = [segment.strip() for segment in markdown_content.split("\n\n") if segment.strip()]

    with PdfPages(output_path) as pdf:
        current_page = plt.figure(figsize=(8.27, 11.69))
        ax = current_page.add_subplot(111)
        ax.axis("off")
        y_position = 0.97

        ax.text(0.5, y_position, title, fontsize=18, fontweight="bold", ha="center", va="top")
        y_position -= 0.05

        for paragraph in paragraphs:
            wrapped = _wrap_paragraph(paragraph)
            estimated_height = max(0.045, 0.015 * (wrapped.count("\n") + 1))

            if y_position - estimated_height < 0.05:
                pdf.savefig(current_page, bbox_inches="tight")
                plt.close(current_page)
                current_page = plt.figure(figsize=(8.27, 11.69))
                ax = current_page.add_subplot(111)
                ax.axis("off")
                y_position = 0.97

            ax.text(0.05, y_position, wrapped, fontsize=10, ha="left", va="top")
            y_position -= estimated_height

        pdf.savefig(current_page, bbox_inches="tight")
        plt.close(current_page)

        for figure_path in figure_paths:
            image = mpimg.imread(figure_path)
            figure = plt.figure(figsize=(8.27, 11.69))
            ax = figure.add_subplot(111)
            ax.axis("off")
            ax.imshow(image)
            ax.set_title(figure_path.stem.replace("_", " ").title(), fontsize=14, pad=16)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
