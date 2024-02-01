import os
from llama_parser import LlamaParser

def test_simple_page_text():
    parser = LlamaParser(result_type="text")

    filepath = os.path.join(os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf")
    result = parser.load_data(filepath)
    assert len(result) == 1
    assert len(result[0].text) > 0

def test_simple_page_markdown():
    parser = LlamaParser(result_type="markdown")

    filepath = os.path.join(os.path.dirname(__file__), "test_files/attention_is_all_you_need.pdf")
    result = parser.load_data(filepath)
    assert len(result) == 1
    assert len(result[0].text) > 0
