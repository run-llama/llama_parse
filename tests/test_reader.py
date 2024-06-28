import os
import pytest
from llama_parse import LlamaParse


@pytest.mark.skipif(
    os.environ.get("LLAMA_CLOUD_API_KEY", "") == "",
    reason="LLAMA_CLOUD_API_KEY not set",
)
def test_simple_page_text() -> None:
    parser = LlamaParse(result_type="text")

    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    result = parser.load_data(filepath)
    assert len(result) == 1
    assert len(result[0].text) > 0

@pytest.fixture
def markdown_parser() -> LlamaParse:
    if os.environ.get("LLAMA_CLOUD_API_KEY", "") == "":
        pytest.skip("LLAMA_CLOUD_API_KEY not set")
    return LlamaParse(result_type="markdown", ignore_errors=False)


def test_simple_page_markdown(markdown_parser: LlamaParse) -> None:
    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    result = markdown_parser.load_data(filepath)
    assert len(result) == 1
    assert len(result[0].text) > 0

def test_simple_page_markdown_bytes(markdown_parser: LlamaParse) -> None:
    markdown_parser = LlamaParse(result_type="markdown", ignore_errors=False)

    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    # client must provide extra_info with file_name
    with pytest.raises(ValueError):
        result = markdown_parser.load_data(file_bytes)
    result = markdown_parser.load_data(file_bytes, extra_info={"file_name": "attention_is_all_you_need.pdf"})
    assert len(result) == 1
    assert len(result[0].text) > 0

def test_simple_page_markdown_buffer(markdown_parser: LlamaParse) -> None:
    markdown_parser = LlamaParse(result_type="markdown", ignore_errors=False)

    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    with open(filepath, "rb") as f:
        # client must provide extra_info with file_name
        with pytest.raises(ValueError):
            result = markdown_parser.load_data(f)
        result = markdown_parser.load_data(f, extra_info={"file_name": "attention_is_all_you_need.pdf"})
        assert len(result) == 1
        assert len(result[0].text) > 0


@pytest.mark.skipif(
    os.environ.get("LLAMA_CLOUD_API_KEY", "") == "",
    reason="LLAMA_CLOUD_API_KEY not set",
)
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
