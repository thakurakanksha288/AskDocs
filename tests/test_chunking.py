"""Tests for the text chunking logic."""
import pytest
from app.services.ingestion import chunk_text, count_tokens


def test_basic_chunk_splits_long_text():
    """A long text should produce multiple chunks."""
    long_text = "word " * 1500
    chunks = chunk_text(long_text, chunk_size=500, overlap=50)
    assert len(chunks) >= 3


def test_short_text_single_chunk():
    """Short text should produce exactly one chunk."""
    short = "Hello world. This is a short document."
    chunks = chunk_text(short, chunk_size=500, overlap=50)
    assert len(chunks) == 1


def test_chunk_indices_are_sequential():
    text = "token " * 1000
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


def test_overlap_means_some_text_repeated():
    """Adjacent chunks should share some tokens due to overlap."""
    text = "word " * 600
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    if len(chunks) >= 2:
        end_of_first = chunks[0]["text"][-100:]
        start_of_second = chunks[1]["text"][:100]
        assert len(set(end_of_first.split()) & set(start_of_second.split())) > 0


def test_empty_text_returns_empty():
    chunks = chunk_text("", chunk_size=500, overlap=50)
    assert chunks == []


def test_token_count_matches():
    text = "hello world foo bar baz"
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert chunks[0]["token_count"] == count_tokens(chunks[0]["text"])


def test_chunk_size_configurable():
    text = "word " * 400
    small_chunks = chunk_text(text, chunk_size=100, overlap=10)
    large_chunks = chunk_text(text, chunk_size=400, overlap=10)
    assert len(small_chunks) > len(large_chunks)
