import os

from fsspec.implementations.local import LocalFileSystem

from llama_parse import LlamaParse


def test_simple_page_text() -> None:
    parser = LlamaParse(result_type="text")

    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    result = parser.load_data(filepath)
    assert len(result) == 1
    assert len(result[0].text) > 0


def test_simple_page_markdown() -> None:
    parser = LlamaParse(result_type="markdown")

    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    result = parser.load_data(filepath)
    assert len(result) == 1
    assert len(result[0].text) > 0


def test_simple_page_with_custom_fs() -> None:
    parser = LlamaParse(result_type="markdown")
    fs = LocalFileSystem()
    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    result = parser.load_data(filepath, fs=fs)
    assert len(result) == 1


def test_simple_page_progress_workers() -> None:
    parser = LlamaParse(result_type="markdown", show_progress=True, verbose=True)

    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    result = parser.load_data([filepath, filepath])
    assert len(result) == 2
    assert len(result[0].text) > 0

    parser = LlamaParse(
        result_type="markdown", show_progress=True, num_workers=2, verbose=True
    )

    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    result = parser.load_data([filepath, filepath])
    assert len(result) == 2
    assert len(result[0].text) > 0
