import os
import pytest
import shutil
from fsspec.implementations.local import LocalFileSystem
from httpx import AsyncClient

from llama_cloud_services.parse import LlamaParse


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
    result = markdown_parser.load_data(
        file_bytes, extra_info={"file_name": "attention_is_all_you_need.pdf"}
    )
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
        result = markdown_parser.load_data(
            f, extra_info={"file_name": "attention_is_all_you_need.pdf"}
        )
        assert len(result) == 1
        assert len(result[0].text) > 0


@pytest.mark.skipif(
    os.environ.get("LLAMA_CLOUD_API_KEY", "") == "",
    reason="LLAMA_CLOUD_API_KEY not set",
)
@pytest.mark.asyncio
async def test_simple_page_with_custom_fs() -> None:
    parser = LlamaParse(result_type="markdown")
    fs = LocalFileSystem()
    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    result = await parser.aload_data(filepath, fs=fs)
    assert len(result) == 1


@pytest.mark.skipif(
    os.environ.get("LLAMA_CLOUD_API_KEY", "") == "",
    reason="LLAMA_CLOUD_API_KEY not set",
)
@pytest.mark.asyncio
async def test_simple_page_progress_workers() -> None:
    parser = LlamaParse(result_type="markdown", show_progress=True, verbose=True)

    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    result = await parser.aload_data([filepath, filepath])
    assert len(result) == 2
    assert len(result[0].text) > 0

    parser = LlamaParse(
        result_type="markdown", show_progress=True, num_workers=2, verbose=True
    )

    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    result = await parser.aload_data([filepath, filepath])
    assert len(result) == 2
    assert len(result[0].text) > 0


@pytest.mark.skipif(
    os.environ.get("LLAMA_CLOUD_API_KEY", "") == "",
    reason="LLAMA_CLOUD_API_KEY not set",
)
@pytest.mark.asyncio
async def test_custom_client() -> None:
    custom_client = AsyncClient(verify=False, timeout=10)
    parser = LlamaParse(result_type="markdown", custom_client=custom_client)
    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    result = await parser.aload_data(filepath)
    assert len(result) == 1
    assert len(result[0].text) > 0


@pytest.mark.skipif(
    os.environ.get("LLAMA_CLOUD_API_KEY", "") == "",
    reason="LLAMA_CLOUD_API_KEY not set",
)
@pytest.mark.asyncio
async def test_input_url() -> None:
    parser = LlamaParse(result_type="markdown")

    # links to a resume example
    input_url = "https://cdn-blog.novoresume.com/articles/google-docs-resume-templates/basic-google-docs-resume.png"
    result = await parser.aload_data(input_url)
    assert len(result) == 1
    assert "your name" in result[0].text.lower()


@pytest.mark.skipif(
    os.environ.get("LLAMA_CLOUD_API_KEY", "") == "",
    reason="LLAMA_CLOUD_API_KEY not set",
)
@pytest.mark.asyncio
async def test_input_url_with_website_input() -> None:
    parser = LlamaParse(result_type="markdown")
    input_url = "https://www.example.com"
    result = await parser.aload_data(input_url)
    assert len(result) == 1
    assert "example" in result[0].text.lower()


@pytest.mark.skipif(
    os.environ.get("LLAMA_CLOUD_API_KEY", "") == "",
    reason="LLAMA_CLOUD_API_KEY not set",
)
@pytest.mark.asyncio
async def test_mixing_input_types() -> None:
    parser = LlamaParse(result_type="markdown")
    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    input_url = "https://cdn-blog.novoresume.com/articles/google-docs-resume-templates/basic-google-docs-resume.png"
    result = await parser.aload_data([filepath, input_url])

    assert len(result) == 2


@pytest.mark.skipif(
    os.environ.get("LLAMA_CLOUD_API_KEY", "") == "",
    reason="LLAMA_CLOUD_API_KEY not set",
)
@pytest.mark.asyncio
async def test_download_images() -> None:
    parser = LlamaParse(result_type="markdown", take_screenshot=True)
    filepath = os.path.join(
        os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf"
    )
    json_result = await parser.aget_json([filepath])

    assert len(json_result) == 1
    assert len(json_result[0]["pages"][0]["images"]) > 0

    download_path = os.path.join(os.path.dirname(__file__), "test_files/images")
    shutil.rmtree(download_path, ignore_errors=True)

    await parser.aget_images(json_result, download_path)
    assert len(os.listdir(download_path)) == len(json_result[0]["pages"][0]["images"])
